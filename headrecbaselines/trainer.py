import os

import torch
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def trainer(train_dataset, val_dataset, model, config, seed=420):
    torch.manual_seed(seed)

    device = torch.device('cpu' if not torch.cuda.is_available() else
                          config['device'] if 'device' in config else 'cuda:0')
    print(f"running in {device}")

    model = model.to(device)

    train_loader = DataLoader(train_dataset, config['train_batch_size'], True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, config['val_batch_size'],
                            num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), config['lr'],
                                 weight_decay=config['weight_decay'])

    trn_loss_avg, trn_rec_loss_avg, val_loss_avg, val_hd_avg = [], [], [], []

    sv_fld = os.path.join('trained/', config['name'])  # model and log save fld
    os.makedirs(sv_fld, exist_ok=True)
    writer = SummaryWriter(log_dir=sv_fld)
    print(f"Tensorboard folder: {sv_fld}")

    best = 1e12

    print('Training ...')

    scheduler = StepLR(
        optimizer, step_size=config['step_size'], gamma=config['gamma']
    )

    for epoch in range(config['epochs']):
        model.train()

        trn_loss_avg.append(0)
        trn_rec_loss_avg.append(0)
        num_batches = 0

        print(f'Epoch [{epoch + 1}/{config["epochs"]}]. ')
        for i, sample_batched in enumerate(train_loader):
            print(f'\r  train batch [{i+1}/{len(train_loader)}]', end='')
            image, target = sample_batched['image'].to(device), \
                            sample_batched['landmarks'].to(device)
            out = model(image)

            optimizer.zero_grad()

            B = target.shape[0]
            out_loss = f.mse_loss(out, target.reshape(B, -1))
            loss = out_loss

            trn_rec_loss_avg[-1] += out_loss.item()
            trn_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1

        trn_loss_avg[-1] /= num_batches
        trn_rec_loss_avg[-1] /= num_batches

        print(f'\n    train avg reconstruction error: {trn_rec_loss_avg[-1]}')

        # Validation
        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_hd_avg.append(0)

        with torch.no_grad():
            for i, sample_batched in enumerate(val_loader):
                print(f'\r  val batch [{i + 1}/{len(val_loader)}]', end='')

                image, target = sample_batched['image'].to(device), \
                                sample_batched['landmarks'].to(device)

                out = model(image)

                out = out.reshape(-1, 3)
                target = target.reshape(-1, 3)

                loss_rec = mean_squared_error(out.cpu().numpy(),
                                              target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1

        val_loss_avg[-1] /= num_batches
        val_hd_avg[-1] /= num_batches

        print(f'\n    val avg reconstruction error: {val_loss_avg[-1]}')

        writer.add_scalar('Train/Loss', trn_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', trn_rec_loss_avg[-1], epoch)
        writer.add_scalar('Validation/MSE', val_loss_avg[-1], epoch)

        if val_loss_avg[-1] < best:
            best = val_loss_avg[-1]
            print(f'Model Saved MSE (epoch {epoch + 1})')
            out = "bestMSE.pt"
            torch.save(model.state_dict(), os.path.join(sv_fld, out))

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(sv_fld, "final.pt"))
