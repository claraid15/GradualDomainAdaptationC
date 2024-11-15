Dataset instructions
- Code for the paper "Understanding Self-Training for Gradual Domain Adaptation"
- The portraits dataset is an existing dataset, and can be downloaded from
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0
- After downloading, extract the tar file, and copy the "M" and "F" folders inside a folder called dataset_32x32 inside the
code folder (current folder, where the README is). Then run "python create_dataset.py"
- Create a folder called "saved_files" in the current code folder.
- Experiments on other datasets should work without downloading additional datasets.
- For the final camera-ready code, we will include virtualenv instructions, and a more automated process for processing the dataset.

Main files
- gradual_shift_better.py contains scripts for experiments in Section 5.1 and 5.3.
- regularization_helps.py contains scripts for experiments in Section 5.2.
