from .utils import stl_to_vtk
from .deformetrica import estimate_registration
import os

def fit_meshes(preds_folder, inps_folder, pred_suffix='', vol_suffix='.nii', overwrite=False):
    """ Fit prediction meshes with their corresponding input mesh.

    Use deformetrica to fit the meshes in preds_folder to their corresponding
    input volume mesh in vols_folder. The output is saved in output_folder.

    :param preds_folder: Folder with the predicted meshes.
    :param inps_folder: Folder with the corresponding volume meshes.
    :param pred_suffix: Suffix of the predicted meshes.
    :param vol_suffix: Suffix of the corresponding volumes.
    :param overwrite: If True, overwrite the output files.
    :return: output_folder: Folder where the fitted meshes are saved.
    """
    preds_folder = os.path.expanduser(preds_folder)
    inps_folder = os.path.expanduser(inps_folder)

    fitted_mshs_folders = []  # List of the output folders

    for pred in os.listdir(preds_folder):
        pred_suffix = os.path.splitext(pred)[1] if pred_suffix == '' \
            else pred_suffix
        has_exts = any([pred.endswith(e) for e in ['.vtk', '.stl']])
        if not pred.endswith(pred_suffix) or not has_exts:
            continue
        pred_path = os.path.join(preds_folder, pred)
        vol_name = pred.replace(pred_suffix, vol_suffix)
        vol_path = os.path.join(inps_folder, vol_name)
        if not os.path.isfile(vol_path):
            print(f"Volume {vol_path} not found.")
            continue
        print(f'Fitting mesh {pred_path} to volume {vol_path}')
        fitted = fit_mesh_to_mesh(pred_path, vol_path, overwrite)
        fitted_mshs_folders.append(fitted)

    return fitted_mshs_folders


def fit_mesh_to_mesh(pred_path, fixedmesh_path, owrt=False):
    """ Fit a mesh to another using deformetrica.

    :param pred_path: Path of the predicted mesh.
    :param fixedmesh_path: Path of the corresponding volume mesh.
    :param owrt: If True, overwrite the output file.
    """
    if not pred_path.endswith('.vtk'):  # TODO Do this with a temp file
        pred_path = stl_to_vtk(pred_path)
    pred_folder = os.path.dirname(pred_path)
    vol_fname_noext = os.path.splitext(os.path.split(fixedmesh_path)[1])[0]
    out_path = os.path.join(pred_folder, 'reg_' + vol_fname_noext)
    estimate_registration(pred_path, [fixedmesh_path], out_path, overwrite=owrt)
    return out_path
