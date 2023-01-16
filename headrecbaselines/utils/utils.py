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


def mesh2vol(mesh_path, source_path, out_img_path=None, overwrite=False,
             spacing=None, origin=None, direction=None, fg_val=1, bg_val=0):
    """ Convert a mesh to a volume using the source image as a reference.

    :param mesh_path: path to the mesh
    :param source_path: path to the source image
    :param out_img_path: path to the output image
    :param overwrite: if True, overwrite the output image if it already exists
    :return: None
    """
    print(f"Binarizing {mesh_path}..", end='')

    mesh_ext = os.path.splitext(mesh_path)[1]

    out_img_path = out_img_path if out_img_path is not None \
        else mesh_path.replace(mesh_ext, "_binMesh.nii.gz")
    if os.path.dirname(out_img_path) == '':
        out_img_path = os.path.join(os.path.dirname(mesh_path), out_img_path)
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

    if not overwrite and os.path.exists(out_img_path):
        print(f"  file {out_img_path} already exists.")
        return out_img_path
    
    src_im = vedo.Volume(source_path)
    inp_mesh = vedo.Mesh(mesh_path)
    
    spacing = src_im._data.GetSpacing() if spacing is None else spacing
    origin = (-110.5, -135.6, -4) if origin is None else origin
    direction_matrix = (-1, 0, 0, 0, -1, 0, 0, 0, 1) if direction is None \
        else direction
    image_size = src_im.tonumpy().shape
    
    bin_vol = inp_mesh.binarize(spacing, False, direction_matrix, image_size, 
                                origin, fg_val, bg_val)
    
    if out_img_path.endswith('.nii.gz'):  # Cannot be saved with vedo
        tmp_im = tempfile.NamedTemporaryFile(suffix='.nii')
        bin_vol.write(tmp_im.name)
        sitk.WriteImage(sitk.ReadImage(tmp_im.name), out_img_path)
        tmp_im.close()
    else:  # ends with .nii
        bin_vol.write(out_img_path)
    
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

import math
import os
import os.path
import tempfile
from os.path import expanduser as eu
from typing import Optional

import SimpleITK as sitk
import numpy as np
import vedo
import vtk


def get_niigz_imgs(path: str, file_ids: Optional[dict] = None):
    """From a path get a list of nii.gz images.

    :param path: Input path. It could be a folder or a file.
    :param file_ids: Dictionary of file identifiers.
        example = {
            'image': '_image.nii.gz',
            'mask': '_mask.nii.gz',
        }
    :return: List of dicts of files.
        example = [
            {'image': 'path/to/image1.nii.gz',
             'mask': 'path/to/mask1.nii.gz'},
            {'image': 'path/to/image2.nii.gz',
             'mask': 'path/to/mask2.nii.gz'}
        ]
        If no file_ids are provided, the files are labelled as 'image'.
        example = [
            {'image': 'path/to/image1.nii.gz'},
            {'image': 'path/to/image2.nii.gz'}
        ]
    """
    print(f"Getting nii.gz images from {path}")
    paths, result = [], []
    if os.path.isdir(path):
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            if os.path.isfile(full_path) and f.endswith('.nii.gz'):
                paths.append(full_path)
    elif os.path.isfile(path) and path.endswith('.nii.gz'):
        paths.append(path)
    else:
        raise FileNotFoundError(
            f"The input path does not contain any valid files ({path})"
        )

    print(f"Found {len(paths)} images")
    list_not_empty = len(paths) > 0
    if file_ids is not None:  # In case we have image, mask (for example)
        while list_not_empty:
            path = paths[0]
            f_name = os.path.split(path)[1]
            for key, value in file_ids.items():  # For each type of image
                if value in f_name:  # Found a valid ID in the file name
                    dc = {}  # Here I'll store image, mask for a subject.
                    for kk in file_ids.keys():  # Grab all IDs and replace them
                        replaced = path.replace(value, file_ids[kk])
                        if not os.path.isfile(replaced):
                            raise FileNotFoundError(
                                f"The file {replaced} does not exists. Base "
                                f"path: {path}, file_ids: {file_ids}. You "
                                f"must rename the files in the folder making "
                                f"them differ only in the ids."
                            )
                        dc[kk] = replaced  # Add the file for that type
                        paths.remove(replaced)  # Remove the file from the list
                        list_not_empty = len(paths) > 0
                    result.append(dc)
    else:
        result = [{'image': p} for p in paths]  # No file_ids provided

    return result


def clip_int_sitk(image, imin, imax):
    """ Clip intensities in a SimpleITK image.

    :param image: Input image.
    :param imin: Minimum intensity value.
    :param imax: Maximum intensity value.
    :return: Clipped image.
    """
    arr = sitk.GetArrayFromImage(image)
    outimg = sitk.GetImageFromArray(np.clip(arr, imin, imax))
    outimg.SetSpacing(image.GetSpacing())
    outimg.SetOrigin(image.GetOrigin())
    outimg.SetDirection(image.GetDirection())
    return outimg


def min_max_normalization(image: sitk.Image):
    """  Map the image intensities to the [0, 1] range.

    :param image: SimpleITK image
    :return: normalized image
    """
    image = sitk.Cast(image, sitk.sitkFloat32)

    arr = sitk.GetArrayFromImage(image)
    return (image - arr.min()) / (arr.max() - arr.min())


def threshold(image: sitk.Image, value: float = 0):
    """ Threshold an image.

    :param image: SimpleITK image
    :param value: Threshold value
    :return: Thresholded image
    """
    pixel_id = image.GetPixelIDValue()
    image = image > value
    sitk.Cast(image, pixel_id)
    return image


def resample_sz_sp(image, mask=None, im_interp='linear', target_spacing=None,
                   direction=None, origin=None, target_size=None):
    """Resample the image size (without deforming the image).

    Resample the image size (without deforming the image) and spacing
    for matching the spacing given as parameter.

    :param target_spacing: desired spacing.
    :param direction: custom direction.
    :param origin: custom origin.
    :param target_size: desired image size.
    :return:
    """
    if not target_spacing:
        if not target_size:
            return None

    if direction is None:
        direction = image.GetDirection()
    if origin is None:
        origin = image.GetOrigin()

    orig_sz = image.GetSize()
    orig_sp = image.GetSpacing()

    t_sz = lambda osz, osp, tsp: int(math.ceil(osz * (osp / tsp)))
    t_sp = lambda osz, osp, tsz: osz * osp / tsz

    if not target_spacing and target_size:
        target_spacing = [t_sp(orig_sz[0], orig_sp[0], target_size[0]),
                          t_sp(orig_sz[1], orig_sp[1], target_size[1]),
                          t_sp(orig_sz[2], orig_sp[2], target_size[2])]

    if not target_size:
        target_size = [t_sz(orig_sz[0], orig_sp[0], target_spacing[0]),
                       t_sz(orig_sz[1], orig_sp[1], target_spacing[1]),
                       t_sz(orig_sz[2], orig_sp[2], target_spacing[2])]

    if im_interp.lower() == 'linear':
        interpolator = sitk.sitkLinear
    elif im_interp.lower() in ['nearest', 'nearestneighbor']:
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError(f"Unknown interpolator: {im_interp}")

    print("    Image")
    image = sitk.Resample(image, target_size, sitk.Transform(),
                          interpolator, origin, target_spacing,
                          direction, 0.0,
                          image.GetPixelIDValue())

    if mask:
        print("    Mask")
        mask = sitk.Resample(mask, target_size, sitk.Transform(),
                             sitk.sitkNearestNeighbor, origin,
                             target_spacing, direction, 0.0,
                             mask.GetPixelIDValue())

    return image, mask


def fixed_pad(v, final_img_size=None, mode="constant", constant_values=(0, 0),
              return_padding=False):
    """ Add fixed image size padding to an volume v

    :param v: Input 3D image.
    :param final_img_size: Desired image size.
    :param mode: One of the np.pad modes ('constant', 'edge', 'maximum',
    'mean', 'reflect', 'symmetric', 'wrap').
    :param constant_values: Used in 'constant'.  The values to set the padded
    values for each axis
    :param return_padding: Return added padding as output with the result.
    :return: Padded image or tuple containing the padded image with the
    padded size, according to the return_padding option.
    """
    if final_img_size is None:
        print("Desired image size not provided!")
        return None

    for i in range(0, len(final_img_size)):
        if v.shape[i] > final_img_size[i]:
            print("The input size is bigger than the output size!")
            print(v.shape, " vs ", final_img_size)
            return None

    padding = (
        (0, final_img_size[0] - v.shape[0]),
        (0, final_img_size[1] - v.shape[1]),
        (0, final_img_size[2] - v.shape[2]),
    )

    if not return_padding:
        return np.pad(v, padding, mode, constant_values=constant_values)
    else:
        return (
            np.pad(v, padding, mode, constant_values=constant_values),
            padding,
        )


def fixed_pad_sitk(sitk_img, pad):
    """ Add padding to a SimpleITK image.

    :param sitk_img: Input image.
    :param pad: Desired image size.
    :return: Padded image.
    """
    arr = sitk.GetArrayFromImage(sitk_img)
    out_img = sitk.GetImageFromArray(fixed_pad(arr, pad))
    out_img.SetSpacing(sitk_img.GetSpacing())
    out_img.SetOrigin(sitk_img.GetOrigin())
    out_img.SetDirection(sitk_img.GetDirection())
    return out_img


def sort_dict_by_val(labels: dict, filt: float = 0., top=-1):
    """ Sort dict based on the values and remove the smaller items

    It will sort the whole dict and then remove those key-value pairs whose
    value is smaller than filt * the_first_value.

    The top parameter keeps the N biggest values. This option overrides the
    filt parameter.
    """
    # Sort descending by value
    labels = {k: v for k, v in sorted(labels.items(),
                                      key=lambda item: item[1],
                                      reverse=True)}
    if filt == 0:
        return labels

    i = 0
    if top == -1:
        for i, val in enumerate(labels.values()):  # Determine cut index (i)
            if val < filt * list(labels.values())[0]:
                break
    elif top == 0:
        return {}
    elif top < 0:
        raise AttributeError(f"A negative value for the 'top' parameter was"
                             f"provided ({top}).")
    else:
        i = top

    # Create the dict with the first i elements only
    return {k: v for k, v in zip(list(labels.keys())[:i],
                                 list(labels.values())[:i])}


def get_largest_cc(image, rel=1, top=-1):
    """ Retains the biggest components of a mask.

    It obtains the largest connected components, and according to the rel
    parameter, it draws the components that are at least rel % of the size
    of the biggest CC.

    If the top parameter is set (different than -1), it will override the rel
    parameter and take the biggest N connected components.
    """

    image = sitk.Cast(image, sitk.sitkUInt32)  # Cast to uint32

    connected_component_filter = sitk.ConnectedComponentImageFilter()
    objects = connected_component_filter.Execute(image)

    labels = {}  # Save label id -> size
    # If there is more than one connected component
    if connected_component_filter.GetObjectCount() > 1:
        objects_data = sitk.GetArrayFromImage(objects)

        # Detect the largest connected component
        for i in range(1, connected_component_filter.GetObjectCount() + 1):
            component_data = objects_data[objects_data == i]
            labels[i] = len(component_data.flatten())  # Voxel count

        f_labels = sort_dict_by_val(labels, filt=rel, top=top)
        data_aux = np.zeros(objects_data.shape, dtype=np.uint8)
        for label in f_labels.keys():
            data_aux[objects_data == label] = 1

        # Save edited image
        output = sitk.GetImageFromArray(data_aux)
        output.SetSpacing(image.GetSpacing())
        output.SetOrigin(image.GetOrigin())
        output.SetDirection(image.GetDirection())
    else:
        output = image

    return output


def save_img(img, fpath):
    """ Save an image in the given path.

    :param img: Image to save.
    :param fpath: Path to save the image.
    :return:
    """
    os.makedirs(os.path.split(fpath)[0], exist_ok=True)
    sitk.WriteImage(img, fpath)
    print(f"    Saved image in {fpath}")


def nii_to_stl_marching_cubes_folder(input_f_path, output_folder=None,
                                     overwrite=True):
    """ Convert a folder of nii files to stl files using marching cubes.

    :param input_f_path: Path to the folder containing the nii files.
    :param output_folder: Path to the folder where the stl files will be
    saved.
    :param overwrite: If True, it will overwrite the existing files.
    :return:
    """
    if output_folder is None:
        output_folder = input_f_path

    saved_files = []
    for f in os.listdir(input_f_path):
        if f.endswith(".nii.gz"):
            f = nii_to_stl_marching_cubes(os.path.join(input_f_path, f),
                                          output_folder, overwrite)
            saved_files.append(f)
    return saved_files


def nii_to_stl_marching_cubes(input_im_path, output_file=None, overwrite=True):
    if type(input_im_path) is list:
        saved_files = []
        for im in input_im_path:
            f = nii_to_stl_marching_cubes(im, output_file, overwrite)
            saved_files.append(f)
        return saved_files
    elif os.path.isdir(input_im_path):
        nii_to_stl_marching_cubes_folder(input_im_path, output_file, overwrite)
        return

    # If no output path provided
    alt_out_path = input_im_path.replace("nii.gz", "stl")

    if output_file is None:
        output_file = alt_out_path
    elif os.path.isdir(output_file):
        output_file = os.path.join(output_file, os.path.split(alt_out_path)[1])

    if not overwrite and os.path.exists(output_file):
        print(f"    File {output_file} already exists. Skipping...")
        return output_file

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(input_im_path)
    reader.Update()

    # Marching Cubes
    dmc = vtk.vtkMarchingCubes()
    dmc.SetInputData(reader.GetOutput())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    transform = vtk.vtkTransform()
    qfm = reader.GetQFormMatrix()
    t1 = qfm.GetElement(0, 0) * qfm.GetElement(0, 3)
    t2 = qfm.GetElement(1, 1) * qfm.GetElement(1, 3)
    t3 = qfm.GetElement(2, 2) * qfm.GetElement(2, 3)
    transform.Translate(t1, t2, t3)

    transformPoly = vtk.vtkTransformPolyDataFilter()
    transformPoly.SetInputConnection(dmc.GetOutputPort())
    transformPoly.SetTransform(transform)
    transformPoly.Update()

    # Save the mesh
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transformPoly.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_file)
    writer.Write()

    print(f"    Saved mesh in {output_file}")

    return output_file


def sitk_to_stl_marching_cubes(sitk_img, output_file=None):
    im = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    im_path = im.name
    sitk.WriteImage(sitk_img, im_path)
    output_file = nii_to_stl_marching_cubes(im_path, output_file)

    return output_file


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


def decimate_meshes(meshes_folder, decimate_factor, overwrite=True,
                    ending='.stl'):
    """ Decimate the meshes in the given folder.

    :param meshes_folder: Path to the folder containing the meshes.
    :param decimate_factor: Decimation factor.
    :param overwrite: If True, it will overwrite the existing meshes.
    :return: List of paths to the decimated meshes.
    :param ending: Ending of the files to be decimated.
    """
    print('Decimating meshes...')

    meshes_folder = eu(meshes_folder)
    print(f'  Input folder: {meshes_folder}')
    meshes_paths = [os.path.join(meshes_folder, f) for f in
                    os.listdir(meshes_folder) if
                    f.endswith(ending) and 'decimated' not in f]
    saved_meshes = []
    print(f'  Found {len(meshes_paths)} meshes')
    for i, mesh in enumerate(meshes_paths):
        if ending and not mesh.endswith(ending):
            continue
        print(
            f'    [{i + 1}/{len(meshes_paths)}] '
            f'Input mesh: {os.path.split(mesh)[1]}'
        )
        decimated_path = mesh.replace(
            '.stl',
            f'_decimated_{int(decimate_factor * 100)}perc.stl'
        )
        if not overwrite and os.path.exists(decimated_path):
            print(f'    Mesh already exists, skipping...')
            saved_meshes.append(decimated_path)
            continue
        v_mesh = vedo.Mesh(mesh).decimate(decimate_factor)
        v_mesh.write(decimated_path)
        saved_meshes.append(decimated_path)
        print(f'      Decimated mesh saved in {decimated_path}')

    saved_meshes = [stl_to_vtk(f) for f in saved_meshes]
    return saved_meshes
