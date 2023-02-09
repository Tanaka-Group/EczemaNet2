# @Author:  zihaowang
# @Email:   zihao.wang20@alumni.imperial.ac.uk
# @Website: www.wangzihao.org
# @Date:    2021-01-20 09:35:50
# @Last Modified by:   zihaowang
# @Last Modified time: 2023-02-11 12:09:13

# Usage:
# For skin segmentation: python train.py --option "skin"
# For eczema segmentation: python train.py --option "ad"
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('keras')
import ad_seg_utils as seg_utils
import pickle
import sys

def main():
	# display input arguments (i.e., type of segmentation) in the log
	SEGMENTATION = str(sys.argv[2])
	MODEL_NAME = '/' + SEGMENTATION + '_seg_da_batch8.h5'
	print("Program initiating... type of segmentation: " + SEGMENTATION)

	"""# Segmentation model training"""
	BACKBONE = 'efficientnetb3'
	BATCH_SIZE = 8
	LR = 0.00025
	EPOCHS = 15
	CLASSES = []
	WEIGHTS = []
	BIN_SEG = True

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
	MODEL_DIR = os.path.join(PROJ_DIR, 'output/models')
	PRED_DIR = os.path.join(PROJ_DIR, 'output/predictions')
	# debug
	print(MODEL_DIR + MODEL_NAME)
	print(MODEL_DIR + '/learning_curve_' + SEGMENTATION + 'batch8.png')

	x_train_dir = os.path.join(DATA_DIR, 'augmented_training_set_corrected/reals')
	y_train_dir = os.path.join(DATA_DIR, 'augmented_training_set_corrected/labels')

	x_valid_dir = os.path.join(DATA_DIR, 'validation_set_corrected/reals')
	y_valid_dir = os.path.join(DATA_DIR, 'validation_set_corrected/labels')

	x_test_dir = os.path.join(DATA_DIR, 'test_set/reals')
	y_test_dir = os.path.join(DATA_DIR, 'test_set/labels')

	# directory debug
	# print("training reals: " + str(x_train_dir))
	# print("training labels: " + str(y_train_dir))
	#
	# print("testing reals: " + str(x_test_dir))
	# print("testing labels: " + str(y_test_dir))
	#
	# print("val reals: " + str(x_valid_dir))
	# print("val labels: " + str(y_valid_dir))

	# create necessary directory
	print("checking directory...")
	# create directory if not exist
	if not os.path.exists(PRED_DIR):
	  os.system("mkdir -p " + PRED_DIR)
	print("done!")
	
	# fixed size of input are deprecation after EczemaNet meeting Jan. 27
	# IMAGE_WIDTH = 480
	# IMAGE_HEIGHT = 352
	preprocess_input = sm.get_preprocessing(BACKBONE)

	# define network parameters
	n_classes = len(CLASSES)
	# select training mode
	activation = 'sigmoid' if n_classes == 1 else 'softmax'
	#create model
	model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

	# define optomizer
	optim = keras.optimizers.Adam(LR)

	# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
	# Set class weights for diss_loss (background: 1, skin: 1, eczema: 1)

	dice_loss = sm.losses.DiceLoss(class_weights=WEIGHTS)
	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	total_loss = dice_loss + (1 * focal_loss)

	# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
	# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

	metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

	# compile keras model with defined optimozer, loss and metrics
	model.compile(optim, total_loss, metrics)

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

	train_dataloader = seg_utils.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	valid_dataloader = seg_utils.Dataloder(valid_dataset, batch_size=1, shuffle=False)

	# debug
	# print(train_dataloader[0][0].shape)
	# print(train_dataloader[0][1].shape)
	# print((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, n_classes))

	# check shapes for errors
	# assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
	# assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, n_classes)

	# define callbacks for learning rate scheduling and best checkpoints saving
	callbacks = [
		keras.callbacks.ModelCheckpoint(MODEL_DIR + MODEL_NAME, save_weights_only=False, save_best_only=True, mode='min'),
		keras.callbacks.ReduceLROnPlateau(),
	]

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

	test_dataloader = seg_utils.Dataloder(test_dataset, batch_size=1, shuffle=False)
	# print general evaluation metrics
	scores = model.evaluate_generator(test_dataloader)

	print("Loss: {:.5}".format(scores[0]))
	for metric, value in zip(metrics, scores[1:]):
		print("mean {}: {:.5}".format(metric.__name__, value))

	# Plot training & validation iou_score values
	plt.figure(figsize=(30, 5))
	plt.subplot(121)
	plt.plot(history.history['iou_score'])
	plt.plot(history.history['val_iou_score'])
	plt.title('Model iou_score')
	plt.ylabel('iou_score')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')

	# Plot training & validation loss values
	plt.subplot(122)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	# plt.show()
	plt.savefig(MODEL_DIR + '/learning_curve_' + SEGMENTATION + '_batch8.png')

if __name__ == "__main__":
	main()