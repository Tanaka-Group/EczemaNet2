# @Author:  zihaowang
# @Email:   zihao.wang20@alumni.imperial.ac.uk
# @Website: www.wangzihao.org
# @Date:    2021-01-19 14:16:58
# @Last Modified by:   zihaowang
# @Last Modified time: 2023-02-11 12:08:46

import os
import shutil
import numpy as np

TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VAL_RATIO = 0.2

# config PROJ_DIR according to your environment
PROJ_DIR = "/path_to_project_dir"
DATA_DIR = os.path.join(PROJ_DIR, 'data');
IMAGE_DIR = os.path.join(DATA_DIR, 'images/');
MASK_DIR = os.path.join(DATA_DIR, 'masks/');
TRAIN_DIR = os.path.join(DATA_DIR, 'train');

def main():
	np.random.seed(2021)
	print("configuring project directories...")
	if not os.path.exists(TRAIN_DIR):
		# os.system("mkdir -p " + PROJ_DIR + "{data/{train,trainannot,test,testannot,val,valannot}}")
		os.system("mkdir " + DATA_DIR + "/train " + DATA_DIR + "/test " + DATA_DIR + "/val "
			+ DATA_DIR + "/trainannot " + DATA_DIR + "/testannot " + DATA_DIR + "/valannot")
	print("done!")

	file_list = os.listdir(IMAGE_DIR)
	np.random.shuffle(file_list)

	file_list_len = len(file_list)
	train_files, test_files, val_files = np.split(np.array(file_list),[int(file_list_len*TRAIN_RATIO), int(file_list_len*(1-VAL_RATIO))])

	print("\ntraining set: " + str(len(train_files)) + " images")
	print("test set: " + str(len(test_files)) + " images")
	print("validation set: " + str(len(val_files)) + " images \n")

	# Copy-pasting images
	print("dividing dataset...")

	for file in train_files:
		shutil.copy(IMAGE_DIR+file, DATA_DIR + "/train")
		shutil.copy(MASK_DIR+file, DATA_DIR + "/trainannot")
	for file in test_files:
		shutil.copy(IMAGE_DIR+file, DATA_DIR + "/test")
		shutil.copy(MASK_DIR+file, DATA_DIR + "/testannot")

	for file in val_files:
		shutil.copy(IMAGE_DIR+file, DATA_DIR + "/val")
		shutil.copy(MASK_DIR+file, DATA_DIR + "/valannot")

	print("done!")

if __name__ == "__main__":
	main()