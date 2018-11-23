import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
from tflearn.metrics import R2

#%%
import numpy as np
from numpy.random import randint

#%%

def defineArchitecture():

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = convnet = input_data(shape=[None, 128, 128, 1], name='input')

    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    # Step 3: Convolution again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 4: Convolution yet again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 5: Max pooling again
    network = max_pool_2d(network, 2)

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)

    network = fully_connected(network, 5, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=0.01, loss='mean_square', metric=R2(), name='targets')

    return network
#%%


model = tflearn.DNN(defineArchitecture())

# Load path/class_id image file:
# train_dataset_file = 'training_images.txt'
# eval_dataset_file = 'eval_images.txt'

# train_GT_file = 'training_annotations.txt'
# eval_GT_file = 'eval_annotations.txt'

# with open(train_GT_file) as inFile:
    # lines = inFile.readlines()
    # Y = list(map(lambda x: float(x), lines))
    # Y = np.asarray(Y).reshape(len(Y), 1)


# with open(eval_GT_file) as inFile:
    # lines = inFile.readlines()
    # test_y = list(map(lambda x: float(x), lines))
    # test_y = np.asarray(test_y).reshape(len(test_y), 1)

# X, _ = image_preloader(train_dataset_file, image_shape=(256, 256),   mode='file', categorical_labels=False, normalize=False, grayscale=True)
# test_x, _ = image_preloader(eval_dataset_file, image_shape=(256, 256),   mode='file', categorical_labels=False, normalize=False, grayscale=True)


# model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_epoch=True, snapshot_step=500, show_metric=True, run_id='optimus_simulated')

# model.save('optimus.model')

# Load path/class_id image file:
train_dataset_file = 'pictures2.txt'

train_GT_file = 'demodata/datadescription.txt'

with open(train_GT_file) as inFile:
	list = []
	for line in inFile:
		parts = line.split('\t')       
		list.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
	Y = np.array(list)

X, _ = image_preloader(train_dataset_file, image_shape=(128, 128),   mode='file', categorical_labels=False, normalize=False, grayscale=True)
X = np.reshape(X, (-1, 128, 128, 1))

model.fit({'input': X}, {'targets': Y}, n_epoch=10000, validation_set=0.1, snapshot_epoch=True, snapshot_step=500, show_metric=True, run_id='dickbutt_simulated')

model.save('dickbutt.model')



