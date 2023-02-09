import os
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
	SEGMENTATION  = str(sys.argv[2])
	BIN_SEG = True if SEGMENTATION == 'skin' else False
	MODEL_NAME = str(sys.argv[4])
	SUFFIX = str(sys.argv[6])

	CLASSES = ['background', 'skin', 'eczema']
	WEIGHTS = np.array([1, 1, 1])
	target_idx = 2

	if BIN_SEG:
		CLASSES = ['background', 'skin']
		WEIGHTS = np.array([1, 1])
		target_idx = 1

	BACKBONE = 'efficientnetb3'
	preprocess_input = sm.get_preprocessing(BACKBONE)

	"""# Model Evaluation"""
	# config PROJ_DIR according to your environment
	PROJ_DIR = "/path_to_project_dir"
	PRED_DIR = os.path.join(PROJ_DIR, 'output/predictions/' + SEGMENTATION + '_base_' + SUFFIX + '/masks')
	BB_DIR = os.path.join(PROJ_DIR, 'output/predictions/' + SEGMENTATION + '_base_' + SUFFIX + '/boxes')
	EVAL_DIR = os.path.join(PROJ_DIR, 'output/evaluations')
	MODEL_DIR = os.path.join(PROJ_DIR, 'output/models')
	DATA_DIR = os.path.join(PROJ_DIR, 'data')
    
	# new dataset
	x_test_dir = os.path.join(DATA_DIR, 'test_set/reals')
	y_test_dir = os.path.join(DATA_DIR, 'test_set/labels')

	print('reading test images from: ' + str(x_test_dir))
	print('reading test masks from: ' + str(y_test_dir))

	test_dataset = seg_utils.Dataset(
		x_test_dir, 
		y_test_dir, 
		classes=CLASSES, 
		augmentation=None,
		preprocessing=seg_utils.get_preprocessing(preprocess_input),
		is_train=False,
		use_full_resolution=False,
		binary_seg=BIN_SEG,
	)

	model = seg_utils.load_model(dir=MODEL_DIR + MODEL_NAME, classes=CLASSES, weights=WEIGHTS)
	print('Trained model loaded!')

	################################ Mask prediction and evaluation ################################
	"""# Saving Masks Predictions"""
	# save all predictions 
	# clear previous predictions
	print('Clearing previous masks...')
	os.system("mkdir -p " + PRED_DIR)
	os.system("mkdir -p " + BB_DIR)
	os.system("rm " + PRED_DIR + "/*.jpg")
	os.system("rm " + PRED_DIR + "/*.JPG")
	os.system("rm " + BB_DIR + "/*.jpg")
	os.system("rm " + BB_DIR + "/*.JPG")
	os.system("rm " + EVAL_DIR + "/bb_evaluation_" + SEGMENTATION + "_base_" + SUFFIX + ".csv")
	print('Done! Now saving new prediction masks...')

	# Feb 8: export evaluation result as csv file 
	cov = []
	prec = []
	f1 = []
	with open(EVAL_DIR + "/bb_evaluation_" + SEGMENTATION + "_base_" + SUFFIX + ".csv", 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["file_name", "coverage", "precision", "f1_score"])
		for i in range(len(test_dataset)):
			# save predicted masks
			image, gt_mask = test_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = model.predict(image)
			# change the last number to decide which mask to output. [0: background; 1: skin; 2: eczema]
			pr_img = pr_mask[0,:,:,target_idx]
			pr_img = (pr_img * 255).astype(np.uint8)
			cv2.imwrite(os.path.join(PRED_DIR, "pred_" + test_dataset.images_ids[i]), pr_img)
			# generate bounding boxes for each predicted mask
			boxes = []
			img = cv2.imread(os.path.join(PRED_DIR, "pred_" + test_dataset.images_ids[i]))
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
			result = img.copy()
			contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contours = contours[0] if len(contours) == 2 else contours[1]
			for cntr in contours:
				rect = cv2.minAreaRect(cntr)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				area = cv2.contourArea(cntr)
				# Abandon boxes with too small area
				if(area > 10000.0):
					boxes.append(box)
					result = cv2.drawContours(result,[box],0,(0,0,255),2)

			# Feb 8: compute performance of bounding boxes
			coverage_per_image, precision_per_image, f1_per_image = eval_utils.compute_coverage_precision(gt_mask, boxes, target_idx=target_idx)
			cov.append(coverage_per_image)
			prec.append(precision_per_image)
			f1.append(f1_per_image)

			writer.writerow([test_dataset.images_ids[i], coverage_per_image, precision_per_image, f1_per_image])
			# save bounding boxes
			cv2.imwrite(os.path.join(BB_DIR, "bb_" + test_dataset.images_ids[i]), result)

		# Feb 8: append the mean and se at the end of the csv file
		writer.writerow(['mean', np.mean(cov), np.mean(prec), np.mean(f1)])
		writer.writerow(['se', np.std(cov) / np.sqrt(len(cov)), np.std(prec) / np.sqrt(len(prec)), np.std(f1) / np.sqrt(len(f1))])

if __name__ == "__main__":
	main()