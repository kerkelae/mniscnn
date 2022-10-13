import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion


bvals = np.loadtxt("train-subject/dwi.bval")
mask = nib.load("train-subject/brain_mask.nii.gz").get_fdata().astype(bool)
mask = binary_erosion(mask, iterations=3)
b0_data = nib.load("train-subject/dwi.nii.gz").get_fdata()[mask][
    :, np.where(bvals == 0)[0]
]
b0_data /= np.mean(b0_data, axis=1)[:, np.newaxis]
print(f"SNR = {1 / np.nanstd(b0_data)} (2-shell HARDI)")

bvals = np.loadtxt("tensor-valued/lte-pte.bval")
mask = nib.load("tensor-valued/brain_mask.nii.gz").get_fdata().astype(bool)
mask = binary_erosion(mask, iterations=3)
b0_data = nib.load("tensor-valued/lte-pte.nii.gz").get_fdata()[mask][
    :, np.where(bvals == 0)[0]
]
b0_data /= np.mean(b0_data, axis=1)[:, np.newaxis]
print(f"SNR = {1 / np.nanstd(b0_data)} (tensor-valued)")
