mrconvert -fslgrad dwi.bvec dwi.bval dwi.nii.gz dwi.mif

dwi2response dhollander dwi.mif wm_response.txt gm_response.txt csf_response.txt

dwi2fod -mask brain_mask.nii.gz msmt_csd dwi.mif wm_response.txt wmfod.nii.gz gm_response.txt gm.nii.gz csf_response.txt csf.nii.gz
