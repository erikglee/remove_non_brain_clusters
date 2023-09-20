import numpy as np
import nibabel as nib
from scipy import ndimage #'1.8.0'

def remove_extra_clusters_from_mask(path_to_mask, path_to_aseg = None):
    '''Function that removes smaller/unconnected clusters from brain mask
    
    Parameters
    ----------
    
    path_to_mask : str
        Path to the binary (0/1) brain mask file to be edited.
    path_to_aseg : str or None (default None)
        Optional path to the corresponding aseg image. If provided,
        the areas from small clusters (defined in mask space) will
        be set to zero in a new copy of this image
        
    Returns
    -------
    None
    
    Makes new copies that replace the input mask file and optionally
    aseg file. These new nifti images will have smaller non-brain regions
    (defined based on the mask image) set to zero.
    
    '''

    mask_img = nib.load(path_to_mask)
    #seg_img = nib.load(path_to_seg)
    
    temp_data = mask_img.get_fdata()
    labels, nb = ndimage.label(temp_data)
    largest_label_size = 0
    largest_label = 0
    for i in range(nb + 1):
        if i == 0:
            continue
        label_size = np.sum(labels == i)
        if label_size > largest_label_size:
            largest_label_size = label_size
            largest_label = i
    new_mask_data = np.zeros(temp_data.shape)
    new_mask_data[labels == largest_label] = 1
    new_mask = nib.nifti1.Nifti1Image(new_mask_data.astype(np.uint8), affine=mask_img.affine, header=mask_img.header)
    nib.save(new_mask, path_to_mask)
    
    if type(path_to_aseg) != type(None):
        aseg_img = nib.load(path_to_aseg)
        aseg_data = aseg_img.get_fdata()
        aseg_data[new_mask_data != 1] = 0
        new_aseg = nib.nifti1.Nifti1Image(aseg_data.astype(np.uint8), affine=aseg_img.affine, header=aseg_img.header)
        nib.save(new_aseg, path_to_aseg)

    return
