import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('keras')
import ad_seg_utils as seg_utils
import bb_eval_utils as eval_utils
import csv
import math
import sys

def main():
	################################ Model loading (this part will be replaced with new data type soon) ################################
	SEGMENTATION = str(sys.argv[2])
	PERTURBATION = str(sys.argv[4])
	MODEL_PREFIX = str(sys.argv[6])
	MODEL_NAME = str(sys.argv[8])

	print("Program initiating... type of segmentation: " + SEGMENTATION + ", type of perturbation: " + PERTURBATION)

	BIN_SEG = True
	CLASSES = ['background', 'skin']
	WEIGHTS = np.array([1, 1])
	target_idx = 1

	BACKBONE = 'efficientnetb3'
	preprocess_input = sm.get_preprocessing(BACKBONE)

	# set parameters based on the type of segmentation
	if SEGMENTATION == 'SKIN' or SEGMENTATION == 'skin':
		pass
	elif SEGMENTATION == 'AD' or SEGMENTATION == 'ad':
		BIN_SEG = False
		target_idx = 2
		CLASSES = ['background', 'skin', 'eczema']
		WEIGHTS = np.array([1, 1, 1])
	else:
		print('Unexpected type of segmentation, should be either skin or ad\n program terminated')
		return -1


	"""# Model Evaluation"""
	# config PROJ_DIR according to your environment
	PROJ_DIR = "/path_to_project_dir"
	PRED_DIR = os.path.join(PROJ_DIR, 'output/predictions/' + SEGMENTATION + '_' + MODEL_PREFIX + '_' + PERTURBATION + '/crops')
	BB_DIR = os.path.join(PROJ_DIR, 'output/predictions/' + SEGMENTATION + '_' + MODEL_PREFIX + '_' + PERTURBATION + '/boxes')
	EVAL_DIR = os.path.join(PROJ_DIR, 'output/evaluations')
	MODEL_DIR = os.path.join(PROJ_DIR, 'output/models')
	DATA_DIR = os.path.join(PROJ_DIR, 'data')

	# new dataset
	x_ref_dir = os.path.join(DATA_DIR, 'test_set/reals')
	y_ref_dir = os.path.join(DATA_DIR, 'test_set/labels')
	x_perturb_dir = os.path.join(DATA_DIR, 'perturbed_test_sets/adversarial_test_set_' + PERTURBATION)
	print('reading ref images from: ' + str(x_ref_dir))
	print('reading perturbed images from: ' + str(x_perturb_dir))

	reference_dataset = seg_utils.Dataset(
		x_ref_dir,
		y_ref_dir,
		classes=CLASSES,
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=False,
		use_full_resolution=False,
		binary_seg=BIN_SEG,
	)

	perturbed_dataset = seg_utils.Dataset(
		x_perturb_dir,
		y_ref_dir,
		classes=CLASSES, 
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=False,
		use_full_resolution=False,
		binary_seg=BIN_SEG,
	)

	model = seg_utils.load_model(dir=MODEL_DIR + MODEL_NAME, classes=CLASSES, weights=WEIGHTS)
	print('Trained model loaded!')

	# # define network parameters
	# n_classes = len(CLASSES)
	# # select training mode
	# activation = 'sigmoid' if n_classes == 1 else 'softmax'
	# #create model
	# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
	# # define optomizer
	# optim = keras.optimizers.Adam(LR)
	# # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
	# # Set class weights for diss_loss (background: 1, skin: 1, eczema: 1)
	# dice_loss = sm.losses.DiceLoss(class_weights=WEIGHTS)
	# focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	# total_loss = dice_loss + (1 * focal_loss)
	# # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
	# # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
	# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
	# # compile keras model with defined optimozer, loss and metrics
	# model.compile(optim, total_loss, metrics)
	# # load trained segmentation model
	# model.load_weights(MODEL_DIR + MODEL_NAME)

	################################ Mask prediction and evaluation ################################
	"""# Saving Masks Predictions"""
	# save all predictions 
	# clear previous predictions
	print('Creating directories and clearing previous masks...')
	os.system("mkdir -p " + PRED_DIR)
	os.system("mkdir -p " + BB_DIR)
	os.system("rm " + PRED_DIR + "/*.jpg")
	os.system("rm " + PRED_DIR + "/*.JPG")
	os.system("rm " + BB_DIR + "/*.jpg")
	os.system("rm " + BB_DIR + "/*.JPG")
	os.system("rm " + EVAL_DIR + "/robustness_evaluation_" + SEGMENTATION + "_" + MODEL_PREFIX + "_" + PERTURBATION + ".csv")
	print('Done! Now saving new prediction masks...')
	
	# create a list to store a series of IoU values
	iou = []

	with open(EVAL_DIR + "/robustness_evaluation_" + SEGMENTATION + "_" + MODEL_PREFIX + "_" + PERTURBATION + ".csv", 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Reference file name", "Perturbation file name", "IoU"])
		for i in range(len(reference_dataset)):
			# prediction for reference and perturbed images
			ref_image, gt_mask = reference_dataset[i]
			ref_image = np.expand_dims(ref_image, axis=0)
			ref_pred = model.predict(ref_image)

			perturbed_image, _ = perturbed_dataset[i]
			perturbed_image = np.expand_dims(perturbed_image, axis=0)
			perturbed_pred = model.predict(perturbed_image)

			# extract the last index to decide which mask to output. [0: background; 1: skin; 2: eczema]
			ref_pred_img = ref_pred[0, :, :, target_idx]
			ref_pred_img = (ref_pred_img * 255).astype(np.uint8)

			perturbed_pred_img = perturbed_pred[0,:,:,target_idx]
			perturbed_pred_img = (perturbed_pred_img * 255).astype(np.uint8)

			# save the images
			cv2.imwrite(os.path.join(PRED_DIR, "pred_" + reference_dataset.images_ids[i]), ref_pred_img)
			cv2.imwrite(os.path.join(PRED_DIR, "pred_" + perturbed_dataset.images_ids[i]), perturbed_pred_img)
			print("saving", reference_dataset.images_ids[i])
			print("saving", perturbed_dataset.images_ids[i])

			# generate bounding boxes for each predicted mask
			perturbed_boxes = []
			perturbed_pred_img = cv2.imread(os.path.join(PRED_DIR, "pred_" + perturbed_dataset.images_ids[i]))
			gray = cv2.cvtColor(perturbed_pred_img,cv2.COLOR_BGR2GRAY)
			thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
			perturbed_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			perturbed_contours = perturbed_contours[0] if len(perturbed_contours) == 2 else perturbed_contours[1]

			ref_boxes = []
			ref_pred_img = cv2.imread(os.path.join(PRED_DIR, "pred_" + reference_dataset.images_ids[i]))
			gray = cv2.cvtColor(ref_pred_img, cv2.COLOR_BGR2GRAY)
			thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
			ref_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			ref_contours = ref_contours[0] if len(ref_contours) == 2 else ref_contours[1]

			# extract contours from perturbed predictions
			for cntr in perturbed_contours:
				rect = cv2.minAreaRect(cntr)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				area = cv2.contourArea(cntr)
				# Abandon boxes with too small area
				if seg_utils.crop_filter(area):
					perturbed_boxes.append(box)

			# extract contours from reference predictions
			for cntr in ref_contours:
				rect = cv2.minAreaRect(cntr)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				area = cv2.contourArea(cntr)
				# Abandon boxes with too small area
				if seg_utils.crop_filter(area):
					ref_boxes.append(box)

			# append IoU to the list
			iou_per_image = eval_utils.compute_robustness_iou(gt_mask, ref_boxes, perturbed_boxes)
			iou.append(iou_per_image)
			writer.writerow([reference_dataset.images_ids[i], perturbed_dataset.images_ids[i], iou_per_image])

		# Append the mean performance to the end of csv
		writer.writerow(['mean', '', np.mean(iou)])
		writer.writerow(['se', '', np.std(iou) / np.sqrt(len(iou))])

		print('Done!')

if __name__ == "__main__":
	main()