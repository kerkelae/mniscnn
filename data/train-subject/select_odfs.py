import nibabel as nib
import numpy as np

from mniscnn import isft, sft


mask_img = nib.load("brain_mask.nii.gz")
mask = mask_img.get_fdata().astype(bool)
odfs_sh = nib.load("wmfod.nii.gz").get_fdata()[mask]

odfs = isft.numpy() @ odfs_sh[..., np.newaxis]
odfs[odfs < 0] = 0
odfs /= 4 * np.pi * np.sum(odfs, axis=1)[:, np.newaxis, :] / isft.size(0)  # normalize
odfs_sh = (sft.numpy() @ odfs)[:, :, 0]
odfs_sh = odfs_sh[~np.sum(np.isnan(odfs_sh), axis=1).astype(bool)]

np.savetxt("odfs_sh.txt", odfs_sh)
