"""
README

This file contains all necessary classes and functions for evaluating the segmentation network.
The generated (skin/AD) crops can then be used as the inputs for severity network.

The document of this project is available here:
    https://github.ic.ac.uk/tanaka-group/EczemaNet-DeepLearning-Segmentation/blob/master/README.md

"""

import matplotlib.path as pltPath
import numpy as np
import math
from matplotlib import pyplot as plt

def compute_coverage_precision(true_mask, boxes, target_idx=2):
	""" compute the precision, recall (coverage) and the F1-score for the given prediction.

	# Arguments
		true_mask: the ground truth mask for the given image
		boxes: the labelled anchors produced by border following

		target_idx: indicate whether we are working on skin or AD segmentation.
		* this idx will be used to retrieve correct mask from ground truth, where:
		0 - background
		1 - skin
		2 - eczema

	# Returns
		return the metrics in the following order: recall, precision, F1-score.

	"""

	boxes_area = 0
	pixels_in_box_and_mask = 0
	pixels_in_mask = 0
	
	# coordinate transformation
	for box in boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]

	# iterate over all pixels to find if it is contained in box(es)
	for i in range(true_mask.shape[0]):
		for j in range(true_mask.shape[1]):
			# count the number of eczema (or skin) pixels in mask (E_m)
			if true_mask[i, j, target_idx] > 0:
				pixels_in_mask += 1

			for box in boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j,true_mask.shape[0] - i]]):
					# count pixels in both ground truth mask and boxes (TP)
					if true_mask[i, j, target_idx] > 0:
						pixels_in_box_and_mask += 1
					# debug
					# pixel_box_num[i,j] += 50

					# accumulate to calculate the union region of boxes (A_b)
					boxes_area += 1
					break
	# debug
	# plt.imshow(pixel_box_num)

	# define evaluation metrics
	coverage = 0 if pixels_in_mask==0 else pixels_in_box_and_mask / pixels_in_mask
	precision = 0 if boxes_area==0 else pixels_in_box_and_mask / boxes_area
	f1 = 0 if precision+coverage == 0 else 2 * precision * coverage / (precision + coverage)

	# print(pixels_in_mask / area)
	print('current coverage=', coverage)
	print('current precision=', precision)
	print('current f1=', f1)
	return coverage, precision, f1


def compute_robustness_iou(true_mask, ref_boxes, perturbed_boxes):
	""" compute the IoU score for robustness analysis

	# Arguments
		true_mask: the ground truth mask for the given image
		ref_boxes: the labelled boxes for predictions on the unperturbed images
		perturbed_boxes: the labelled boxes for predictions on the perturbed images

	# Returns
		return the IoU score as a float number

		"""
	ref_region = np.zeros(shape=true_mask.shape[0:2], dtype=np.int8)
	perturbed_region = np.zeros(shape=true_mask.shape[0:2], dtype=np.int8)

	# coordinate transformation
	for box in ref_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]
	for box in perturbed_boxes:
		for point in box:
			point[1] = true_mask.shape[0] - point[1]

	# iterate over all pixels to find which of them are contained in reference prediction
	# and which of them are contained in perturbed prediction
	for i in range(true_mask.shape[0]):
		for j in range(true_mask.shape[1]):
			# iterate through all the boxes to check if pixel is inside any of them
			for box in ref_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j, true_mask.shape[0] - i]]):
					# count pixels in both ground truth mask and boxes (TP)
					ref_region[i, j] += 1
					break
			for box in perturbed_boxes:
				path = pltPath.Path([box[0], box[1], box[2], box[3]])
				if path.contains_points([[j, true_mask.shape[0] - i]]):
					# count pixels in both ground truth mask and boxes (TP)
					perturbed_region[i, j] += 1
					break

	intersection = np.multiply(ref_region, perturbed_region)
	union = np.add(ref_region, perturbed_region)
	# 	print("max: ", np.max(intersection))
	nb_intersection = np.sum([pixel > 0 for pixel in intersection])
	nb_union = np.sum([pixel > 0 for pixel in union])
	IoU = 0 if nb_union == 0 else nb_intersection / nb_union

	print("IoU: ", IoU)

	# Code for visualisation

	# 	plt.figure(dpi=200)
	# 	plt.tight_layout()

	# 	plt.subplot(141)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(ref_region)
	# 	plt.title("ref pred")

	# 	plt.subplot(142)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(perturbed_region)
	# 	plt.title("perturbed pred")

	# 	plt.subplot(143)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(union)
	# 	plt.title("union")

	# 	plt.subplot(144)
	# 	plt.box(False)
	# 	plt.axis('off')
	# 	plt.imshow(intersection)
	# 	plt.title("intersection")

	# 	plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=0.2, dpi=200)
	# 	plt.show()

	return IoU

