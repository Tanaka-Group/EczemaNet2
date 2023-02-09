# @Author:  zihaowang
# @Email:   zihao.wang20@alumni.imperial.ac.uk
# @Website: www.wangzihao.org
# @Date:    2021-01-21 23:43:14
# @Last Modified by:   zihaowang
# @Last Modified time: 2023-02-11 12:08:54

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


################################ Model loading (this part will be replaced with new data type soon) ################################
BIN_SEG = True

MODEL_NAME = '/mul_seg_model.h5'
CLASSES = ['background', 'skin', 'eczema']
WEIGHTS = np.array([1, 1, 1])
target_idx = 2

if BIN_SEG:
    MODEL_NAME = '/bin_seg_model.h5'
    CLASSES = ['background', 'skin']
    WEIGHTS = np.array([1, 1])
    target_idx = 1

BACKBONE = 'efficientnetb3'
LR = 0.0001
preprocess_input = sm.get_preprocessing(BACKBONE)

"""# Model Evaluation"""
# config PROJ_DIR according to your environment
PROJ_DIR = "/path_to_project_dir"PRED_DIR = os.path.join(PROJ_DIR, 'output/predictions/skin_seg_ad_iden')
BB_DIR = os.path.join(PROJ_DIR, 'output/bounding_boxes')
EVAL_DIR = os.path.join(PROJ_DIR, 'output/evaluations')
MODEL_DIR = os.path.join(PROJ_DIR, 'output')
DATA_DIR = os.path.join(PROJ_DIR, 'data')

# old dataset
# x_test_dir = os.path.join(DATA_DIR, 'old_data/test')
# y_test_dir = os.path.join(DATA_DIR, 'old_data/testannot')

# new dataset
x_test_dir = os.path.join(DATA_DIR, 'perturbed_test_sets/adversarial_test_set_blur')
y_test_dir = os.path.join(DATA_DIR, 'test_set/labels')

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
eczema_dataset = seg_utils.Dataset(
    x_test_dir,
    y_test_dir,
    classes=['background', 'skin', 'eczema'],
    augmentation=None,
    preprocessing=seg_utils.get_preprocessing(preprocess_input),
    is_train=False,
    use_full_resolution=False,
    binary_seg=0,
)

test_dataloader = seg_utils.Dataloder(test_dataset, batch_size=1, shuffle=False)

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

# load trained segmentation model
model.load_weights(MODEL_DIR + MODEL_NAME)


################################ Mask prediction and evaluation ################################
"""# Saving Masks Predictions"""
# save all predictions
# clear previous predictions
print('Clearing previous masks...')
os.system("rm " + PRED_DIR + "/*.jpg")
os.system("rm " + PRED_DIR + "/*.JPG")
os.system("rm " + BB_DIR + "/*.jpg")
os.system("rm " + BB_DIR + "/*.JPG")
os.system("rm " + EVAL_DIR + "/bb_evaluation_SKSEG_DA_BLUR.csv")
print('Done! Now saving new prediction masks...')

# Feb 8: export evaluation result as csv file
cov = []
prec = []
f1 = []
with open(EVAL_DIR + '/bb_evaluation_SKSEG_DA_BLUR.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name", "coverage", "precision", "f1_score"])
    for i in range(len(test_dataset)):
        # save predicted masks
        image, _ = test_dataset[i]
        _, gt_mask = eczema_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)
        # change the last number to decide which mask to output. [0: background; 1: skin; 2: eczema]
        pr_img = pr_mask[0,:,:,target_idx]
        pr_img = (pr_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(PRED_DIR, "pred_" + test_dataset.ids[i]), pr_img)
        # generate bounding boxes for each predicted mask
        boxes = []
        img = cv2.imread(os.path.join(PRED_DIR, "pred_" + test_dataset.ids[i]))
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
        coverage_per_image, precision_per_image, f1_per_image = eval_utils.compute_coverage_precision(gt_mask, boxes, target_idx=2)
        cov.append(coverage_per_image)
        prec.append(precision_per_image)
        f1.append(f1_per_image)

        writer.writerow([test_dataset.ids[i], coverage_per_image, precision_per_image, f1_per_image])
        # save bounding boxes
        cv2.imwrite(os.path.join(BB_DIR, "bb_" + test_dataset.ids[i]), result)
    # Feb 8: append the mean performance to the end of csv
    # f1, se = eval_utils.compute_f1(cov, prec)
    writer.writerow(['mean', np.mean(cov), np.mean(prec), np.mean(f1)])
    writer.writerow(['se', np.std(cov)/np.sqrt(len(cov)), np.std(prec)/np.sqrt(len(prec)), np.std(f1)/np.sqrt(len(f1))])
    print('Done!')