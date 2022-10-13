# fitmcmicro is available at https://github.com/ekaden/smt

./fitmcmicro --bvals dwi.bval --bvecs dwi.bvec test_signals.nii.gz smt_results.nii.gz

./fitmcmicro --mask brain_mask.nii.gz --bvals dwi.bval --bvecs dwi.bvec dwi.nii.gz smt_maps.nii.gz
