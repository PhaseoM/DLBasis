block_n = 3
epochs = 160
batchsize = 128
weight_decay = 0.0001
momentum = 0.9
learning_rate = 0.1  # divide it by 10 at 32k and 48k iterations
is_resiual = True

# 45k/5k train/val split

# data augmentation
# training: 4 pixels are padded , and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
# testing: the original 32×32 image.
