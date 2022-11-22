import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import vedo
import vtkmodules.all as vtk


def draw_organ(ax, array, color='b'):
    N = array.shape[0]
    for i in range(0, N):
        x, y = array[i, :]
        circ = plt.Circle((x, y), radius=3, color=color, fill=True)
        ax.add_patch(circ)
    return


def draw_lines(ax, array, color='b'):
    N = array.shape[0]
    for i in range(0, N):
        x1, y1 = array[i - 1, :]
        x2, y2 = array[i, :]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
    return


def drawOrgans(RL, LL, H=None, img=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if img is not None:
        plt.imshow(img, cmap='gray')
    else:
        img = np.zeros([1024, 1024])
        plt.imshow(img)

    plt.axis('off')

    draw_lines(ax, RL, 'r')
    draw_lines(ax, LL, 'g')

    draw_organ(ax, RL, 'r')
    draw_organ(ax, LL, 'g')

    if H is not None:
        draw_lines(ax, H, 'y')
        draw_organ(ax, H, 'y')

    return


import torch


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


## Adjacency Matrix
def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i - 1] = 1
        sub[i, (i + 1) % N] = 1
    return sub


## Downsampling Matrix
def mOrganD(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N2, N])

    for i in range(0, N2):
        if (2 * i + 1) == N:
            sub[i, 2 * i] = 1
        else:
            sub[i, 2 * i] = 1 / 2
            sub[i, 2 * i + 1] = 1 / 2

    return sub


## Upsampling Matrix
def mOrganU(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N, N2])

    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i // 2] = 1
        else:
            sub[i, i // 2] = 1 / 2
            sub[i, (i // 2 + 1) % N2] = 1 / 2

    return sub


## Generating Matrixes for every organ
def genMatrixesLungs():
    RLUNG = 44
    LLUNG = 50

    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)

    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))

    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)

    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)

    p1 = RLUNG
    p2 = p1 + LLUNG

    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))

    A = np.zeros([p2, p2])

    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2

    AD = np.zeros([p2_, p2_])

    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2

    D = np.zeros([p2_, p2])

    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2

    U = np.zeros([p2, p2_])

    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2

    return A, AD, D, U


def genMatrixesLungsHeart():
    RLUNG = 44
    LLUNG = 50
    HEART = 26

    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)

    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))

    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)

    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)

    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART

    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))

    A = np.zeros([p3, p3])

    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3

    AD = np.zeros([p3_, p3_])

    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3

    D = np.zeros([p3_, p3])

    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3

    U = np.zeros([p3, p3_])

    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3

    return A, AD, D, U


def CrossVal(all_files, iFold, k=5, p=0.783):
    if k > 1:
        total = len(all_files)
        val = int(total / k)

        indices = list(range(total))

        train_indices = indices[0:(iFold - 1) * val] + indices[iFold * val:]
        val_indices = indices[(iFold - 1) * val:iFold * val]

        train_paths = [all_files[i] for i in train_indices]
        val_paths = [all_files[i] for i in val_indices]
    else:
        train_paths = all_files[:int(p * len(all_files))]
        val_paths = all_files[int(p * len(all_files)):]

    return train_paths, val_paths


def check_files_exist(files, exception=True, warn=True):
    if type(files) is not list:
        files = [files]
    for f in files:
        if not os.path.exists(f):
            if exception:
                raise FileNotFoundError(f)
            elif warn:
                print(f"File {f} not found.")
                return False
            else:
                return False
    return True


def binarize_mesh(mesh, src_img, invert=False):
    """ Binarize a mesh using the src_img as a reference.

    :param mesh: vedo mesh to be binarized
    :param src_img: SimpleITK.Image used as a reference
    :param invert: if True, the mesh is inverted (background is 1)
    :return: binarized mesh, as a SimpleITK.Image
    """
    pd = mesh.polydata()  # get the mesh data

    # Define the volume
    whiteImage = vtk.vtkImageData()
    whiteImage.SetDirectionMatrix(-1, 0, 0, 0, -1, 0, 0, 0, 1)
    whiteImage.SetSpacing(src_img.GetSpacing())

    dim = src_img.GetSize()
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

    whiteImage.SetOrigin(src_img.GetOrigin())
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # fill the image with foreground voxels:
    inval = 0 if invert else 1

    # count = whiteImage.GetNumberOfPoints()
    # for i in range(count):
    #     whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

    # Check if it's the same image
    whiteImage.GetPointData().GetScalars().Fill(inval)

    # polygonal data --> image stencil:
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin(whiteImage.GetOrigin())
    pol2stenc.SetOutputSpacing(whiteImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    # cut the corresponding white image and set the background:
    outval = 1 if invert else 0

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.SetReverseStencil(invert)
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    # Convert to SimpleITK
    array = vedo.Volume(imgstenc.GetOutput()).tonumpy().astype(np.uint8)
    array = np.transpose(array, (2, 1, 0))

    o_img = sitk.GetImageFromArray(array)
    o_img.CopyInformation(src_img)
    return o_img


def mesh2vol(mesh_path, source_path, out_img_path=None, overwrite=False):
    """ Convert a mesh to a volume using the source image as a reference.

    :param mesh_path: path to the mesh
    :param source_path: path to the source image
    :param out_img_path: path to the output image
    :param overwrite: if True, overwrite the output image if it already exists
    :return: None
    """
    print(f"Binarizing {mesh_path}..", end='')

    mesh_ext = os.path.splitext(mesh_path)[1]
    if mesh_ext not in ['.vtk', '.stl']:
        print(f"Mesh {mesh_path} has extension {mesh_ext}. Trying to continue")

    out_img_path = out_img_path if out_img_path is not None \
        else mesh_path.replace(mesh_ext, "_binMesh.nii.gz")
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

    if not overwrite and os.path.exists(out_img_path):
        print(f"  file {out_img_path} already exists.")
        return out_img_path

    src_im = sitk.ReadImage(source_path)
    inp_mesh = vedo.Mesh(mesh_path)
    bin_vol = binarize_mesh(inp_mesh, src_im)  # Vedo mesh -> sitk.Image

    sitk.WriteImage(bin_vol, out_img_path)
    print("done.")
    return out_img_path


def stl_to_vtk(filename):
    """ Convert an STL file to a VTK file.

    :param filename: Path to the STL file.
    :return:
    """
    # Read the .stl file
    a = vtk.vtkSTLReader()
    a.SetFileName(filename)
    a.Update()
    a = a.GetOutput()

    # Write the .vtk file
    filename = filename.replace('.stl', '.vtk')
    b = vtk.vtkPolyDataWriter()
    b.SetFileName(filename)
    b.SetInputData(a)
    b.Update()
    return filename


def vtk_to_stl(filename):
    if type(filename) is list:
        files = []
        for file in filename:
            f = vtk_to_stl(file)
            files.append(f)
        if False in files:
            print(f"WARNING: {file} could not be converted to stl.")
        return files
    if os.path.isfile(filename):
        outfile = filename.replace('.vtk', '.stl')
        print(f"Creating {outfile}")
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(filename)
        reader.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(reader.GetOutputPort())
        writer.SetFileName(outfile)
        writer.Update()
        return outfile
    else:
        return False
