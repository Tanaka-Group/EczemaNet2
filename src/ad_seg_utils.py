"""
README

This file contains all necessary classes and functions for AD image segmentation.
The generated (skin/AD) crops can then be used as the inputs for severity network.

The installation document is available here:
    https://github.ic.ac.uk/tanaka-group/EczemaNet-DeepLearning-Segmentation/blob/master/README.md

"""

# Import libraries
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import glob
import segmentation_models as sm
import random


def load_model(dir, classes=None, weights=None):
    """ Load the trained segmentation model from the given direction

    # Arguments
        dir: relative or absolute directory of where the (.h5) model stores
        classes: [deprecated feature], used for loading (skin/AD) model
        weights: [deprecated feature], used for loading (skin/AD) model

    # Returns
        return the model on the given directory

    """

    # cross entropy loss
    dependencies = {
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'f1-score': sm.metrics.FScore(threshold=0.5)
    }
    loaded_model = keras.models.load_model(dir, custom_objects=dependencies)

    return loaded_model

def visualize(**images):
    """ Plot images in one row

    # Arguments
        images: image(s) stored as numpy array

    # Returns
        plot image(s) in a row, return void

    """
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
    """Scale image to range [0, 1] for correct plot

    # Arguments
        x: image in numpy array

    # Returns
        return the normalized image in numpy array

    """
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def random_color():
    """generate a random color which could be either 'red', 'blue' or 'green'

    # Arguments
        void

    # Returns
        return the random rgb tuple

    """
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def crop_visualisation(image, x0, y0, x1, y1):
    """Visualize square cropping to check if the crops are segmented
    (into squares) correctly

    # Arguments
        image: image read by OpenCV library

        x0, y0, x1, y1: represent the four corners of the square.
        where (x0, y0) and (x1, y1) should be the coordinates of
        the two diagonal points

    # Returns
        add a color layer on the image, each color represent a crop,
        so we can check if the cropping algorithm work as we expected.
        return the image with the color layers

    """
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    zeros_mask = cv2.rectangle(zeros1, (x0, y0), (x1, y1),
                                color=random_color(), thickness=-1)

    mask = np.array(zeros_mask)
    try:
        # transparency of the background picture
        alpha = 1
        # transparency of the masks
        beta = 0.5
        gamma = 0
        # cv2.addWeighted combined image and mask together
        mask_img = cv2.addWeighted(image, alpha, mask, beta, gamma)
        # cv2.imwrite(os.path.join(output_fold, 'masked_img.jpg'), mask_img)

    except:
        print('Exception captured.')

    return mask_img

def crop_square(image, rect, box):
    """Crop the AD image into square images in order to reduce
    distortion during the resizing stage in severity network.

    # Arguments
        image: image read by OpenCV library
        rec: min area rectangular returned from border following (returned by cv2.minAreaRect())
        box = box points returned from border following (returned by cv2.boxPoints(rect))

    # Returns
        return a list contains all the cropped square crops, each crop is stored in a numpy array

    """
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
    crop_list = []

    x0 = 0
    y0 = 0
    threshold = 0.5

    if(width > height):
        L = height
        x1 = x0 + L
        y1 = L
        while x1 <= width - 1:
            crop_list.append(warped[y0:y1, x0:x1])
            # visualisation for debugging
            # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # warped = crop_visualisation(warped, x0, y0, x1, y1)

            if x1 == width - 1:
                break
            x1 = x1 + L if x1 + L < width else (width - 1 if width - x1 >= L * threshold else width)
            x0 = x1 - L
    else:
        L = width
        x1 = L
        y1 = y0 + L
        while y1 <= height - 1:
            crop_list.append(warped[y0:y1, x0:x1])
            # visualisation for debugging
            # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # warped = crop_visualisation(warped, x0, y0, x1, y1)

            if y1 == height - 1:
                break
            y1 = y1 + L if y1 + L < height else (height - 1 if height - y1 >= L * threshold else height)
            y0 = y1 - L
    # visualisation for debugging
    # plt.imsave('masked_crop.jpg', warped)

    return crop_list


def crop_square_quality(image, rect, box, target_idx=1):
    """Similar functions as crop_square(image, rect, box) but added exclusion
    rules to remove low quality crops, setting the limitation of side length
    (in pixel) to adjust crop qualities

    # Arguments
        image: image read by OpenCV library
        rec: min area rectangular returned from border following (returned by cv2.minAreaRect())
        box: box points returned from border following (returned by cv2.boxPoints(rect))
        target_idx: [deprecated feature] can be ignored

    # Returns
        return a list contains all the cropped square crops, each crop is stored in a numpy array

    """
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
    crop_list = []

    x0 = 0
    y0 = 0
    len_threshold = 0.5
    side_threshold = 200

    if(width > height):
        L = height
        if L >= side_threshold:
            x1 = x0 + L
            y1 = L
            while x1 <= width - 1:
                crop_list.append(warped[y0:y1, x0:x1])
                # visualisation for debugging
                # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if x1 == width - 1:
                    break
                x1 = x1 + L if x1 + L < width else (width - 1 if width - x1 >= L * len_threshold else width)
                x0 = x1 - L
    else:
        L = width
        if L >= side_threshold:
            x1 = L
            y1 = y0 + L
            while y1 <= height - 1:
                crop_list.append(warped[y0:y1, x0:x1])
                # visualisation for debugging
                # cv2.rectangle(warped, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if y1 == height - 1:
                    break
                y1 = y1 + L if y1 + L < height else (height - 1 if height - y1 >= L * len_threshold else height)
                y0 = y1 - L
    # visualisation for debugging debugging
    # plt.figure(dpi=200)
    # plt.imshow(warped)

    return crop_list


def crop_filter(area):
    """Exclude low quality crops only based on their area

    # Arguments
        area: the computed area of the desired image
        area = a * b pixel square, where a and b are height and width respectively.

    # Returns
        return a boolean representing whether the image should be removed

    """
    if area >= 10000.0:
        return True
    else:
        return False
######################### temp workspace starts
def run_gt_image_combination(file_name, image_dir, mask_dir, output_dir, refno=None, visno=None):
    """Combine SWET image with their corresponding ground truth mask to
    remove skin and background pixels (noises)

    # Arguments
        file_name: file name of the target image
        image_dir: directory of the folder which stores the SWET image
        mask_dir: directory of the folder which stores the ground truth mask
        output_dir: desired output directory
        refno: the reference no of the diagnosis
        visno: the visual no of the diagnosis

    # Returns
        return
        1. the number of crops generated from the given image
        2. a list with the full directory of the generated crops

    """
    weekno = file_name.split(".")[0]
    weekno = weekno[6:]

    # config path
    crop_dir = os.path.join(output_dir, "crops")
    box_dir = os.path.join(output_dir, "boxes")
    whole_dir = os.path.join(output_dir, "images")

    # create folder if not exist
    if not os.path.exists(crop_dir):
        os.system("mkdir -p " + crop_dir)
    if not os.path.exists(box_dir):
        os.system("mkdir -p " + box_dir)
    if not os.path.exists(whole_dir):
        os.system("mkdir -p " + whole_dir)

    # read and resize image
    image = cv2.imread(image_dir)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cp = image.copy()

    mask = cv2.imread(os.path.join(mask_dir, file_name))
    result = mask.copy()
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    num_crops = 0
    crop_fns = []

    image_cp = cv2.bitwise_and(image_cp, image_cp, mask=thresh)
    whole_img_name = refno + "_vis-" + visno + "_" + weekno + ".jpg"
    plt.imsave(os.path.join(whole_dir, whole_img_name), image_cp)

    # iterate through boxes
    for cntr in contours:
        rect = cv2.minAreaRect(cntr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cntr)

        # filter that excludes small meaningless small boxes
        if crop_filter(area):
            # extract crops from the coordinate of boxes
            img_crops = crop_square_quality(image_cp, rect, box)
            for img in img_crops:
                crop_file_name = refno + "_vis-" + visno + "_" + weekno + "_crop-" + str(
                    num_crops) + ".jpg"
                plt.imsave(os.path.join(crop_dir, crop_file_name), img)

                crop_fns.append(crop_file_name)
                result = cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
                num_crops += 1

    # save the figure with labelled bounding boxes
    box_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_UNet.jpg"
    cv2.imwrite(os.path.join(box_dir, box_file_name), result)
    print(file_name, ": cropping done!")

    return num_crops, crop_fns

######################### temp workspace starts
def run_gt_image_combination_only_remove_background(file_name, image_dir, mask_dir, output_dir, refno=None, visno=None):
    """Similar version as run_gt_image_combination, but only remove the background pixels
    (in other words, skin pixels will not be removed by this function)

    # Arguments
        file_name: file name of the target image
        image_dir: directory of the folder which stores the SWET image
        mask_dir: directory of the folder which stores the ground truth mask
        output_dir: desired output directory
        refno: the reference no of the diagnosis
        visno: the visual no of the diagnosis

    # Returns
        return
        1. the number of crops generated from the given image
        2. a list with the full directory of the generated crops

    """

    weekno = file_name.split(".")[0]
    weekno = weekno[6:]

    # config path
    crop_dir = os.path.join(output_dir, "crops")
    box_dir = os.path.join(output_dir, "boxes")
    whole_dir = os.path.join(output_dir, "images")

    # create folder if not exist
    if not os.path.exists(crop_dir):
        os.system("mkdir -p " + crop_dir)
    if not os.path.exists(box_dir):
        os.system("mkdir -p " + box_dir)
    if not os.path.exists(whole_dir):
        os.system("mkdir -p " + whole_dir)

    # cmd = "rm " + crop_dir + "/" + file_name.split(".")[0] + "*.jpg"
    # os.system(cmd)

    # read and resize image
    image = cv2.imread(image_dir)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cp = image.copy()

    # print(os.path.join(mask_dir, file_name))

    mask = cv2.imread(os.path.join(mask_dir, file_name))
    result = mask.copy()
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh_skin = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)[1]
    thresh_ad = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_ad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    num_crops = 0
    crop_fns = []
    image_cp = cv2.bitwise_and(image_cp, image_cp, mask=thresh_skin)
    whole_img_name = refno + "_vis-" + visno + "_" + weekno + ".jpg"
    plt.imsave(os.path.join(whole_dir, whole_img_name), image_cp)

    # iterate through boxes
    for cntr in contours:
        rect = cv2.minAreaRect(cntr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cntr)

        # filter that excludes small meaningless small boxes
        if crop_filter(area):
            # extract crops from the coordinate of boxes
            img_crops = crop_square_quality(image_cp, rect, box)
            for img in img_crops:
                # weekno = file_name.split(".")[0]
                # weekno = weekno[6:]
                crop_file_name = refno + "_vis-" + visno + "_" + weekno + "_crop-" + str(
                    num_crops) + ".jpg"
                plt.imsave(os.path.join(crop_dir, crop_file_name), img)

                crop_fns.append(crop_file_name)
                result = cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
                num_crops += 1

    # save the figure with labelled bounding boxes
    box_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_UNet.jpg"
    cv2.imwrite(os.path.join(box_dir, box_file_name), result)
    print(file_name, ": cropping done!")

    return num_crops, crop_fns

######################### temp workspace ends
def run_sigle_pred(model, input_dir, output_dir, file_name, preprocessing=None, target_idx=2, resize_ratio=0.4, refno=None, visno=None):
    """Run U-net segmentation + border following + square cropping on a given image,
    generate and store the predicted mask, labelled boxes and square crops.

    # Arguments
        model: the loaded tensorflow model
        input_dir: input directory for the target image
        output_dir: directory to store the outputs
        file_name: the file name of the input image

        target_idx: indicates which type of segmentation you want to apply,
        * target_idx = 1 -> skin segmentation
        * target_idx = 2 -> AD segmentation
        * target_idx = 0 -> background segmentation

        resize_ratio: to what extent would you like to resize the original images.
        * resize_ratio = 1 -> using the original resolution of the image.
        Note due to the large dimension of SWET images, unlimited size may lead to a slow inference.
        

    # Returns
        return
        1. the number of crops generated from the given image
        2. a list with the full directory of the generated crops

    """

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
    print("directory created!")

    # clear previous output for same image
    #     print("checking previous duplicate croppings...")
    # cmd = "rm " + crop_dir + "/" + file_name.split(".")[0] + "*.jpg"
    # os.system(cmd)

    # read and resize image
    image = cv2.imread(input_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width, channels = image.shape
    target_height = int((image_height / 32) * resize_ratio) * 32
    target_width = int((image_width / 32) * resize_ratio) * 32

    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    image_cp = image.copy()

    # preprocessing (lambda transformation) for prediction
    if preprocessing:
        sample = preprocessing(image=image)
        image = sample['image']

    image = np.expand_dims(image, axis=0)

    # get prediction
    pr_mask = model.predict(image)

    # change the index to decide which mask to output. [0: background; 1: skin; 2: eczema]
    pr_img = pr_mask[0, :, :, target_idx]
    pr_img = (pr_img * 255).astype(np.uint8)

    pr_bg = pr_mask[0, :, :, 0]
    pr_bg = (pr_bg * 255).astype(np.uint8)

    # output predicted mask
    cv2.imwrite(os.path.join(mask_dir, "pred_" + refno + "_" + visno + "_" + file_name), pr_img)

    cv2.imwrite(os.path.join(mask_dir, "bg_pred_" + refno + "_" + visno + "_" + file_name), pr_bg)
    # debug (visualisation)
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

    # crop AD regions but only remove background pixels
    mask_bg = cv2.imread(os.path.join(mask_dir, "bg_pred_" + refno + "_" + visno + "_" + file_name))
    gray_bg = cv2.cvtColor(mask_bg, cv2.COLOR_BGR2GRAY)
    thresh_bg = cv2.threshold(gray_bg, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # image_cp = cv2.bitwise_and(image_cp, image_cp, mask=thresh)
    image_cp = cv2.bitwise_and(image_cp, image_cp, mask=thresh_bg)

    # debug (visualisation)
    #     plt.figure()
    #     plt.imshow(image_cp)

    # iterate through boxes
    for cntr in contours:
        rect = cv2.minAreaRect(cntr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cntr)

        # filter that excludes small meaningless small boxes
        if crop_filter(area):
            # extract crops from the coordinate of boxes
            # crop_rect(image_cp, rect, box)
            img_crops = crop_square_quality(image_cp, rect, box, target_idx)
            for img in img_crops:
                crop_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_crop-" + str(num_crops) + ".jpg"
                plt.imsave(os.path.join(crop_dir, crop_file_name), img)

                crop_fns.append(crop_file_name)
                result = cv2.drawContours(result,[box],0,(0,0,255),2)
                num_crops += 1


    # save the figure with labelled bounding boxes
    box_file_name = refno + "_vis-" + visno + "_" + file_name.split(".")[0] + "_UNet.jpg"
    cv2.imwrite(os.path.join(box_dir, box_file_name), result)

    print(file_name, ": cropping done!")

    return num_crops, crop_fns

def run_sigle_pred_whole_img(model, input_dir, output_dir, file_name, preprocessing=None, target_idx=2, resize_ratio=0.4, refno=None, visno=None):
    """Run U-net segmentation without border following on a given image,
    generate and store the whole predicted mask, labelled boxes and the segmented image.

    # Arguments
        model: the loaded tensorflow model
        input_dir: input directory for the target image
        output_dir: directory to store the outputs
        file_name: the file name of the input image

        target_idx: indicates which type of segmentation you want to apply,
        * target_idx = 1 -> skin segmentation
        * target_idx = 2 -> AD segmentation
        * target_idx = 0 -> background segmentation

        resize_ratio: to what extent would you like to resize the original images.
        * resize_ratio = 1 -> using the original resolution of the image.
        Note: due to the large dimensions of SWET images, unlimited size may lead to a slow inference.

    # Returns
        return
        1. the number of crops generated from the given image
        2. a list with the full directory of the generated crops

    """

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
    # cmd = "rm " + crop_dir + "/" + file_name.split(".")[0] + "*.jpg"
    # os.system(cmd)

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


def listdir_nohidden(path, is_train=True):
    """Generate a list contains all the non-hidden files in the given path,
    this is to exclude system hidden files such as .DS_Store
    - feature added: sort the list to prevent images and masks are not 1-to-1 corresponded.

    # Arguments
        path: directory to the image folder

        is_train: boolean variable shows whether this function is used in model training

    # Returns
        return a sorted list which contains all the non-hidden files in the given path

    """
    file_list = [file for file in os.listdir(path) if not file.startswith('.')]
    # print('before', file_list)
    if not is_train:
        if any("_aug" in string for string in file_list):
            file_list = [f.replace('_aug', '') for f in file_list]
            file_list = sorted(file_list)
            file_list = [f.replace('.jpg', '_aug.jpg') for f in file_list]
        # print("aug detected!")
        else:
            file_list = sorted(file_list)


    return file_list

# classes for data loading and preprocessing
class Dataset:
    """ Read images, apply augmentation and preprocessing transformations.

    # Arguments
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

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # debug: check the dimension of the mask
        # print(mask.shape)

        # # apply traditional augmentations (deprecated)
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

    # Arguments
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
    """Get the augmented training set (deprecated)"""
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

    # Arguments
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    # Return
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)