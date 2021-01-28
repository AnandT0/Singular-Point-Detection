Step 0:
  Ensure that the dataset directory contains 'data_used' folder for the images, 'gTruth' for the ground truths of the dataset,
  'Mask_gt' as an empty folder for the generated masks

  Maintain a model directory, which contains 'Val_sample', 'Plot' for the loss plot, 'Weights' for the saved weights and
  'Loss_files' for the losses.
Step 1:
	Replace the paths to the ground truths as well as the destination file in 'Grountruth_in_one_file.py'
  Make sure to create the destination file in the same directory as the dataset to avoid confusion.
  Run 'Grountruth_in_one_file.py'
Step 2:
  Replace the paths for the ground truth with the .txt file that was just created as well as the location of the images of the dataset
  in 'Code_for_mask.py'. Also Replace the output folder path.
  Run 'Code_for_mask.py'
Step 3:
  Run 'Finger_ae_reg_hourglass2.py' after setting the number of epochs. follow the prompts given by the script.
