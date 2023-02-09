# Usage:
# For skin segmentation: python train_batch2.py --seg_type skin --train_dir your_dir --prefix your_prefix
# For eczema segmentation: python train_batch2.py --seg_type ad --train_dir your_dir --prefix your_prefix

import os
# select GPU automatically
# import setGPU

os.environ["SM_FRAMEWORK"] = "keras"
import cv2
import tensorflow as tf
import keras
from keras.utils import generic_utils

import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('keras')
import ad_seg_utils as seg_utils
import pickle
import sys
from datetime import datetime


def main():
	strategy = tf.distribute.MirroredStrategy()
	print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	# display input arguments (i.e., type of segmentation) in the log
	SEGMENTATION = str(sys.argv[2])
	TRAINING_DIR = str(sys.argv[4])
	MODEL_PREFIX = str(sys.argv[6])

	MODEL_NAME = '/' + MODEL_PREFIX + '_' + SEGMENTATION + '_nloss_distributed.h5'
	LC_NAME = '/' + MODEL_PREFIX + '_' + SEGMENTATION + '_nloss_distributed.png'
	print("Program initiating...")
	print("Segmentation: " + SEGMENTATION)
	print("Model prefix: " + MODEL_PREFIX)

	"""# Segmentation model training"""

	BACKBONE = 'efficientnetb3'
	BATCH_SIZE = 2
	LR = 0.0001
	EPOCHS = 50
	BIN_SEG = True

	preprocess_input = sm.get_preprocessing(BACKBONE)
	# set parameters based on the type of segmentation
	if SEGMENTATION == 'SKIN' or SEGMENTATION == 'skin':
		CLASSES = ['background', 'skin']
		WEIGHTS = np.array([1, 1])
	elif SEGMENTATION == 'AD' or SEGMENTATION == 'ad':
		CLASSES = ['background', 'skin', 'eczema']
		WEIGHTS = np.array([1, 1, 1])
		BIN_SEG = False
	else:
		print('Unexpected type of segmentation, should be either skin or ad\n program terminated')
		return -1

	# Please edit your directory to make it consistent with the swet dataset
	PROJ_DIR = "/path_to_project_dir"
	DATA_DIR = os.path.join(PROJ_DIR, 'data')
	MODEL_DIR = os.path.join(PROJ_DIR, 'output/models/' + SEGMENTATION + '_' + MODEL_PREFIX + "_distribute_saveAll")
	PRED_DIR = os.path.join(PROJ_DIR, 'output/predictions')

	# create necessary directories
	print("checking directory...")
	# create directory if not exist
	if not os.path.exists(PRED_DIR):
		os.system("mkdir -p " + PRED_DIR)
	if not os.path.exists(MODEL_DIR):
		os.system("mkdir -p " + MODEL_DIR)

	# debug
	print('Model full directory: ' + MODEL_DIR + MODEL_NAME)
	print('Learning curve full directory: ' + MODEL_DIR + LC_NAME)

	# training set directories
	x_train_dir = os.path.join(TRAINING_DIR, 'reals')
	y_train_dir = os.path.join(TRAINING_DIR, 'labels')

	x_valid_dir = os.path.join(DATA_DIR, 'validation_set_corrected/reals')
	y_valid_dir = os.path.join(DATA_DIR, 'validation_set_corrected/labels')

	x_test_dir = os.path.join(DATA_DIR, 'test_set/reals')
	y_test_dir = os.path.join(DATA_DIR, 'test_set/labels')

	print('checking existing models...')
	if os.path.exists(MODEL_DIR + MODEL_NAME):
		print('checkpoint is found, loading trained models...')
		# with strategy.scope():
		model = seg_utils.load_model(dir=MODEL_DIR+MODEL_NAME, classes=CLASSES, weights=WEIGHTS)
		print('trained model loaded')
	else:
		print('checkpoint is not found, preparing to train new models...')
		# define network parameters
		n_classes = len(CLASSES)
		# select training mode
		activation = 'sigmoid' if n_classes == 1 else 'softmax'
		#create model
		model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
		# define optomizer
		optim = keras.optimizers.Adam(LR)
		metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
		# compile keras model with defined optimozer, loss and metrics
		with strategy.scope():
			model.compile(optim, loss="categorical_crossentropy", metrics=metrics)

	# Dataset for train images
	train_dataset = seg_utils.Dataset(
		x_train_dir,
		y_train_dir,
		classes=CLASSES,
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=True,
		binary_seg=BIN_SEG,
	)

	# Dataset for validation images
	valid_dataset = seg_utils.Dataset(
		x_valid_dir,
		y_valid_dir,
		classes=CLASSES,
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=True,
		binary_seg=BIN_SEG,
	)

	# Output the learning curve
	test_dataset = seg_utils.Dataset(
		x_test_dir,
		y_test_dir,
		classes=CLASSES,
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=True,
		binary_seg=BIN_SEG,
	)

	train_dataloader = seg_utils.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	valid_dataloader = seg_utils.Dataloder(valid_dataset, batch_size=1, shuffle=False)
	test_dataloader = seg_utils.Dataloder(test_dataset, batch_size=1, shuffle=False)
	# check shapes for errors
	# assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
	# assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, n_classes)

	# define callbacks for learning rate scheduling and best checkpoints saving
	today_str = datetime.today().strftime('%Y-%m-%d')
	callbacks = [
		keras.callbacks.ModelCheckpoint(MODEL_DIR + MODEL_NAME, save_weights_only=False, monitor='loss', save_best_only=True, mode='min'),
		keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_delta=0, min_lr=0.4*LR, mode='min'),
		keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, "tensorboard_log/" + str(today_str)))
	]
	print('tensorboard log path: ' + os.path.join(MODEL_DIR, "tensorboard_log/" + str(today_str)))

	print('training starts...')
	# train model
	history = model.fit_generator(
		train_dataloader,
		steps_per_epoch=len(train_dataloader),
		epochs=EPOCHS,
		callbacks=callbacks,
		validation_data=valid_dataloader,
		validation_steps=len(valid_dataloader),
	)

	# save training history performance as a dictionary
	# with open('/trainHistoryDict', 'wb') as file_pi:
		# pickle.dump(history.history, file_pi)

	# print general evaluation metrics
	scores = model.evaluate_generator(test_dataloader)

	print("Loss: {:.5}".format(scores[0]))
	for metric, value in zip(metrics, scores[1:]):
		print("mean {}: {:.5}".format(metric.__name__, value))

if __name__ == "__main__":
	main()