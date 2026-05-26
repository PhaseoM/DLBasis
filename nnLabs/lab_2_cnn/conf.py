block_n = 3
epochs = 160
batchsize = 128
weight_decay = 0.0001
momentum = 0.9
learning_rate = 0.1  # divide it by 10 at 32k and 48k iterations
is_residual = True
is_data_augment = True
info_output_filepath = "./data_analysis/lab_2_cnn/"
fig_output_filepath = "./data_analysis/lab_2_cnn/graph/"
pkl_dump_filepath = "./data_analysis/lab_2_cnn/pkldump/"
# data augmentation
# training: 4 pixels are padded , and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
# testing: the original 32×32 image.
