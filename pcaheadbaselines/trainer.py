import argparse
import os
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.FC import FC
from models.PCA import PCA_Net
from src.models.PCAH import PCAH_Net
from src.models.modelUtils import Pool
from utils.dataset import LandmarksDataset, ToTensor, RandomScale, AugColor, \
    Rotate
from src.utils.datasetHeads import MeshHeadsDataset
from src.utils.utils import genMatrixesLungs, \
    genMatrixesLungsHeart, CrossVal
from src.utils.datasetHeads import SkullRandomHole


def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config[
        'val_batch_size'], num_workers=0)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    train_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []
    val_hd_avg = []

    tensorboard = "trained"

    folder = os.path.join(tensorboard, config['name'])
    os.makedirs(folder, exist_ok=True)
    writer = SummaryWriter(log_dir=folder)
    print(f"Tensorboard folder: {folder}")

    best = 1e12

    print('Training ...')

    scheduler = StepLR(
        optimizer, step_size=config['stepsize'], gamma=config['gamma']
    )
    pool = Pool()

    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        num_batches = 0

        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched[
                'landmarks'].to(device)
            out = model(image)

            optimizer.zero_grad()

            if type(out) is not tuple:
                # PCA and FC
                B = target.shape[0]
                outloss = F.mse_loss(out, target.reshape(B, -1))
                loss = outloss

            elif (len(out)) == 3:
                # HybridGNet 2 IGSC
                target_down = pool(target, model.downsample_matrices[0])

                out, pre1, pre2 = out
                # HybridGNet with 2 skip connections
                pre1loss = F.mse_loss(pre1, target_down)
                pre2loss = F.mse_loss(pre2, target)
                outloss = F.mse_loss(out, target)

                loss = outloss + pre1loss + pre2loss

                kld_loss = -0.5 * torch.mean(torch.mean(
                    1 + model.log_var - model.mu ** 2 - model.log_var.exp(),
                    dim=1), dim=0)
                loss += model.kld_weight * kld_loss

            else:
                raise Exception('Error unpacking outputs')

            train_rec_loss_avg[-1] += outloss.item()
            train_loss_avg[-1] += loss.item()

            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1

        train_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average reconstruction error: %f' % (
            epoch + 1, config['epochs'], train_rec_loss_avg[-1]))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_hd_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, target = sample_batched['image'].to(device), \
                                sample_batched['landmarks'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                out = out.reshape(-1, 3)
                target = target.reshape(-1, 3)

                loss_rec = mean_squared_error(out.cpu().numpy(),
                                              target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1

        val_loss_avg[-1] /= num_batches
        val_hd_avg[-1] /= num_batches

        print('Epoch [%d / %d] validation average reconstruction error: %f' % (
            epoch + 1, config['epochs'], val_loss_avg[-1]))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', train_rec_loss_avg[-1],
                          epoch)

        writer.add_scalar('Validation/MSE', val_loss_avg[-1],
                          epoch)

        if val_loss_avg[-1] < best:
            best = val_loss_avg[-1]
            print('Model Saved MSE')
            out = "bestMSE.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--model", default="HybridGNet", type=str)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--stepsize", default=50, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)

    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default=1, type=int)

    # Number of filters at low resolution for HybridGNet
    parser.add_argument("--f", default=32, type=int)


    # Number of latent variables
    parser.add_argument("--latents", default=64, type=int)

    # Number of filters at low resolution for HybridGNet
    parser.add_argument("--batch_size", default=4, type=int)

    # Define the output: only lungs, or lungs and heart by default
    parser.add_argument('--lungs', dest='Lungs', action='store_true')
    parser.set_defaults(Lungs=False)

    # Define the output: only lungs, or lungs and heart by default
    parser.add_argument('--heads', dest='Heads', action='store_true')
    parser.set_defaults(Heads=False)

    config = parser.parse_args()
    config = vars(config)

    # Transforms for lungs and heart
    transforms_train = transforms.Compose([RandomScale(), Rotate(3),
                                           AugColor(0.40), ToTensor()])
    transforms_val = ToTensor()
    heart = not config['Lungs']
    if config['Heads']:
        print('Organs: Skulls')
        # A, AD, D, U = genMatrixesLungs()  # Hybrid models not implemented yet
        train_heads_files = "heads_files/train_images_heads.txt"
        images = open(train_heads_files, 'r').read().splitlines()
        Dataset = MeshHeadsDataset
        transforms_train = transforms.Compose([SkullRandomHole()])
        transforms_val = transforms.Compose([SkullRandomHole()])
        heart = False
    elif config['Lungs']:
        print('Organs: Lungs')
        A, AD, D, U = genMatrixesLungs()
        images = open("train_images_lungs.txt", 'r').read().splitlines()
        Dataset = LandmarksDataset
    else:
        print('Organs: Lungs and Heart')
        A, AD, D, U = genMatrixesLungsHeart()
        images = open("train_images_heart.txt", 'r').read().splitlines()
        Dataset = LandmarksDataset

    print('Number of images: ', len(images))
    random.Random(13).shuffle(images)

    k = 1 if config['Heads'] else 5

    print('Fold %s' % config['fold'], 'of %s' % k)
    images_train, images_val = CrossVal(images, config['fold'], k)

    train_dataset = Dataset(
        images=images_train, img_path="../Chest-xray-landmark-dataset/Images",
        label_path="../Chest-xray-landmark-dataset/landmarks", heart=heart,
        transform=transforms_train
    )

    val_dataset = Dataset(
        images=images_val, img_path="../Chest-xray-landmark-dataset/Images",
        label_path="../Chest-xray-landmark-dataset/landmarks", heart=heart,
        transform=transforms_val
    )

    config['weight_decay'] = 1e-5
    config['val_batch_size'] = 1
    config['inputsize'] = 1024

    interp_factor = 0.85
    config['h'] = int(512 * interp_factor)
    config['w'] = int(512 * interp_factor)
    config['slices'] = int(233 * interp_factor)
    config['interpolate'] = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config['device'] = torch.device(device)

    if config['model'] == 'PCAH':  # Franco: PCA for heads.
        print('Model: PCA 3D')
        model = PCAH_Net(config)
    elif config['model'] == 'PCA':
        print('Model: PCA')
        model = PCA_Net(config)
    elif config['model'] == 'FC':
        print('Model: FC')
        model = FC(config)
    else:
        raise Exception('No valid model, choose between HybridGNet PCA or FC')

    trainer(train_dataset, val_dataset, model, config)
