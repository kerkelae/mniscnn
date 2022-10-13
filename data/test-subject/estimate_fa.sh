dwi2tensor dwi.mif dt.mif
tensor2metric -fa fa.nii.gz -mask brain_mask.nii.gz dt.mif -force
