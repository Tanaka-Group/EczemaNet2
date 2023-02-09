# @Author:  zihaowang
# @Email:   zihao.wang20@alumni.imperial.ac.uk
# @Website: www.wangzihao.org
# @Date:    2021-01-19 15:29:02
# @Last Modified by:   zihaowang
# @Last Modified time: 2023-02-11 12:08:42

"""
README

This file contains the class for reading dataset and all other necessary functions
(data preprocessing, etc) for training segmentation model. The predicted segmentation
model is then used to generate (skin or AD) crops based on boarder following algorithm.

"""

import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import glob
import segmentation_models as sm

def load_model(dir, classes, weights):
	# focal loss and dice loss
	# n_classes = len(classes)
	# dice_loss = sm.losses.DiceLoss(class_weights=weights)
	# focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	# total_loss = dice_loss + (1 * focal_loss)
	# keras.losses.custom_loss = total_loss
	# dependencies = {
	# 	'dice_loss_plus_1focal_loss': total_loss,
	# 	'iou_score': sm.metrics.IOUScore(threshold=0.5),
	# 	'f1-score': sm.metrics.FScore(threshold=0.5)
	# }

	# cross entropy loss
	dependencies = {
		'iou_score': sm.metrics.IOUScore(threshold=0.5),
		'f1-score': sm.metrics.FScore(threshold=0.5)
	}
	loaded_model = keras.models.load_model(dir, custom_objects=dependencies)

	return loaded_model

def visualize(**images):
	""" Plot images in one row """
	n = len(images)
	plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		plt.subplot(1, n, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		plt.imshow(image)
	plt.show()

# helper function for data visualization    
def denormalize(x):
	"""Scale image to range 0..1 for correct plot"""
	x_max = np.percentile(x, 98)
	x_min = np.percentile(x, 2)
	x = (x - x_min) / (x_max - x_min)
	x = x.clip(0, 1)
	return x

def crop_rect(image, rect, box):
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def crop_filter(area):
	if area > 10000.0:
		return True
	else:
		return False

def run_sigle_pred(model, input_dir, output_dir, file_name, preprocessing=None, target_idx=2, resize_ratio=0.4, refno=None, visno=None):
    # config path
    mask_dir = os.path.join(output_dir, "masks")
    crop_dir = os.path.join(output_dir, "crops")
    box_dir = os.path.join(output_dir, "boxes")

    # create folder if not exist
    if not os.path.exists(mask_dir):
        os.system("mkdir -p " + mask_dir)
    if not os.path.exists(crop_dir):
        os.system("mkdir -p " + crop_dir)
    if not os.path.exists(box_dir):
        os.system("mkdir -p " + box_dir)
#     print("directory created!")

    # clear previous output for same image
#     print("checking previous duplicate croppings...")
    cmd = "rm " + crop_dir + "/" + file_name.split(".")[0] + "*.jpg"
    os.system(cmd)

    # read and resize image
    image = cv2.imread(input_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width, channels = image.shape
    target_height = int((image_height / 32) * resize_ratio) * 32
    target_width = int((image_width / 32) * resize_ratio) * 32

    image = cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    image_cp = image.copy()

    # preprocessing (lambda transformation) for prediction
    if preprocessing:
        sample = preprocessing(image=image)
        image = sample['image']

    image = np.expand_dims(image, axis=0)

    # get prediction
    pr_mask = model.predict(image)

    # change the index to decide which mask to output. [0: background; 1: skin; 2: eczema]
    pr_img = pr_mask[0,:,:,target_idx]
    pr_img = (pr_img * 255).astype(np.uint8)

    # output predicted mask
    cv2.imwrite(os.path.join(mask_dir, "pred_" + refno + "_" + visno + "_" + file_name), pr_img)
    # debug
    # plt.figure()
    # plt.imshow(pr_img)
    # print("mask saved!")

    # generate bounding boxes for each predicted mask
    mask = cv2.imread(os.path.join(mask_dir, "pred_" + refno + "_" + visno + "_" + file_name))
    result = mask.copy()
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    num_crops = 0
    crop_fns = []

    image_cp = cv2.bitwise_and(image_cp, image_cp, mask = thresh)
    # debug crops
#     plt.figure()
#     plt.imshow(image_cp)

    # iterate through boxes
    for cntr in contours:
        rect = cv2.minAreaRect(cntr)

        # print(rect)
        # print(rect[1])

        # recalculate the left corner coordinates (rect[0]) and height&width (rect[1]) for generating full-resolution crops
        # rect = (tuple(x / 1 for x in rect[0]), tuple(x / 1 for x in rect[1]), rect[2])

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cntr)

        # filter that excludes small meaningless small boxes
        if crop_filter(area):
            # extract crops from the coordinate of boxes
            img_crop = crop_rect(image_cp, rect, box)
            crop_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_crop-" + str(num_crops) + ".jpg"
            plt.imsave(os.path.join(crop_dir, crop_file_name), img_crop)

            crop_fns.append(crop_file_name)
            result = cv2.drawContours(result,[box],0,(0,0,255),2)
            num_crops += 1


    # save the figure with labelled bounding boxes 
    box_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_UNet.jpg"
    cv2.imwrite(os.path.join(box_dir, box_file_name), result)

    print(file_name, ": cropping done!")

    return num_crops, crop_fns

def run_sigle_pred_whole_img(model, input_dir, output_dir, file_name, preprocessing=None, target_idx=2, resize_ratio=0.4, refno=None, visno=None):
    # config path
    mask_dir = os.path.join(output_dir, "masks")
    crop_dir = os.path.join(output_dir, "crops")
    box_dir = os.path.join(output_dir, "boxes")
    crop_fns = []

    # create folder if not exist
    if not os.path.exists(mask_dir):
        os.system("mkdir -p " + mask_dir)
    if not os.path.exists(crop_dir):
        os.system("mkdir -p " + crop_dir)
    if not os.path.exists(box_dir):
        os.system("mkdir -p " + box_dir)
    print("directory created!")

    # clear previous output for same image
    print("checking previous duplicate croppings...")
    cmd = "rm " + crop_dir + "/" + file_name.split(".")[0] + "*.jpg"
    os.system(cmd)

    # read and resize image
    image = cv2.imread(input_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width, channels = image.shape
    target_height = int((image_height / 32) * resize_ratio) * 32
    target_width = int((image_width / 32) * resize_ratio) * 32

    image = cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    image_cp = image.copy()

    # preprocessing (lambda transformation) for prediction
    if preprocessing:
        sample = preprocessing(image=image)
        image = sample['image']

    image = np.expand_dims(image, axis=0)

    # get prediction
    pr_mask = model.predict(image)

    # change the index to decide which mask to output. [0: background; 1: skin; 2: eczema]
    pr_img = pr_mask[0,:,:,target_idx]
    pr_img = (pr_img * 255).astype(np.uint8)

    # output predicted mask
    cv2.imwrite(os.path.join(mask_dir, "pred_" + refno + "_" + visno + "_" + file_name), pr_img)
    mask = cv2.imread(os.path.join(mask_dir, "pred_" + refno + "_" + visno + "_" + file_name))
    result = mask.copy()
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]

    # apply the predicted mask on the whole image
    image_cp = cv2.bitwise_and(image_cp, image_cp, mask=thresh)
    crop_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_crop-0" + ".jpg"
    crop_fns.append(crop_file_name)
    plt.imsave(os.path.join(crop_dir, crop_file_name), image_cp)

    print(file_name, ": whole image BR done!")

    return 1, crop_fns

# def listdir_nohidden(path):
#     return glob.glob(os.path.join(path, '*.*'))
def listdir_nohidden(path, is_train=True):

	file_list = [file for file in os.listdir(path) if not file.startswith('.')]
	# print('before', file_list)
	if not is_train:
		if any("_aug" in string for string in file_list):
			file_list = [f.replace('_aug', '') for f in file_list]
			file_list = sorted(file_list)
			# print('after', file_list)
			file_list = [f.replace('.jpg', '_aug.jpg') for f in file_list]
			# print("aug detected!")
		else:
			file_list = sorted(file_list)

	# print('after', file_list)

	return file_list

# classes for data loading and preprocessing
class Dataset:
	""" Read images, apply augmentation and preprocessing transformations.

	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing
			(e.g. noralization, shape manipulation, etc.)

	"""

	CLASSES = ['background', 'skin', 'eczema']

	def __init__(
			self,
			images_dir,
			masks_dir,
			classes=None,
			augmentation=None,
			preprocessing=None,
			is_train=True,
			use_full_resolution=False,
			binary_seg=False,
	):
		self.images_ids = listdir_nohidden(images_dir, is_train)
		self.masks_ids = listdir_nohidden(masks_dir, is_train)
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.images_ids]
		self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.masks_ids]

		# convert str names to class values on masks
		# self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
		# self.class_values = [0, 127, 255]
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

		self.augmentation = augmentation
		self.preprocessing = preprocessing
		self.use_full_resolution = use_full_resolution
		self.is_train = is_train
		self.binary_seg = binary_seg

	def __getitem__(self, i):

		# read data
		image = cv2.imread(self.images_fps[i])

		# convert BGR to RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# define whether or not to resize the input stream to save program running memory
		# note: the size of input has to be divisible by 32 for tensorflow to process
		if not self.use_full_resolution:
			target_ratio = 0.4
		else:
			target_ratio = 1

		image_height, image_width, channels = image.shape
		target_height = 768 if self.is_train else int((image_height / 32) * target_ratio) * 32
		target_width = 1024 if self.is_train else int((image_width / 32) * target_ratio) * 32
		image = cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)

		# load the mask in gray scale, RGB -> single value
		mask = cv2.imread(self.masks_fps[i], 0)
		# 21 Jan, 2021: resize the mask to save memory
		mask = cv2.resize(mask, (target_width, target_height), interpolation = cv2.INTER_AREA)
		# reorganize the RGB value in masks

		if self.binary_seg:
			# binary segmentation
			mask[mask > 1] = 1
		else:
			# multi-class classfication
			mask[mask == 127] = 1
			mask[mask == 255] = 2
			mask[mask > 2] = 1

		# print(np.unique(mask))

		# extract certain classes from mask (e.g. cars)
		masks = [(mask == v) for v in self.class_values]
		mask = np.stack(masks, axis=-1).astype('float')
		# debug msg
		# print(mask.shape)

		# # apply augmentations
		# if self.augmentation:
		#     sample = self.augmentation(image=image, mask=mask)
		#     image, mask = sample['image'], sample['mask']

		# apply preprocessing
		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']

		return image, mask

	def __len__(self):
		return len(self.images_ids)


class Dataloder(keras.utils.Sequence):
	"""Load data from dataset and form batches

	Args:
		dataset: instance of Dataset class for image loading and preprocessing.
		batch_size: Integet number of images in batch.
		shuffle: Boolean, if `True` shuffle image indexes each epoch.
	"""

	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(dataset))

		self.on_epoch_end()

	def __getitem__(self, i):

		# collect batch data
		start = i * self.batch_size
		stop = (i + 1) * self.batch_size
		data = []
		for j in range(start, stop):
			data.append(self.dataset[j])

		# transpose list of lists
		batch = [np.stack(samples, axis=0) for samples in zip(*data)]

		return batch

	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.indexes) // self.batch_size

	def on_epoch_end(self):
		"""Callback function to shuffle indexes each epoch"""
		if self.shuffle:
			self.indexes = np.random.permutation(self.indexes)


def round_clip_0_1(x, **kwargs):
	return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
	train_transform = [

		A.HorizontalFlip(p=0.5),

		A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

		A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
		A.RandomCrop(height=320, width=320, always_apply=True),

		A.IAAAdditiveGaussianNoise(p=0.2),
		A.IAAPerspective(p=0.5),

		A.OneOf(
			[
				A.CLAHE(p=1),
				A.RandomBrightness(p=1),
				A.RandomGamma(p=1),
			],
			p=0.9,
		),

		A.OneOf(
			[
				A.IAASharpen(p=1),
				A.Blur(blur_limit=3, p=1),
				A.MotionBlur(blur_limit=3, p=1),
			],
			p=0.9,
		),

		A.OneOf(
			[
				A.RandomContrast(p=1),
				A.HueSaturationValue(p=1),
			],
			p=0.9,
		),
		A.Lambda(mask=round_clip_0_1)
	]
	return A.Compose(train_transform)


def get_validation_augmentation():
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
		A.PadIfNeeded(384, 480)
	]
	return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform

	Args:
		preprocessing_fn (callbale): data normalization function
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose

	"""

	_transform = [
		A.Lambda(image=preprocessing_fn),
	]
	return A.Compose(_transform)