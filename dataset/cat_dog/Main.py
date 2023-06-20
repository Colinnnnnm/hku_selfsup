import shutil
import os
import random
random.seed(7600)
folder_path = "dataset/cat_dog/"
images = os.listdir(folder_path+"data")

train_set = random.sample(images, k = int(len(images)*0.8))
test_set = [t for t in images if t not in train_set]



# creating directories for each class
try:
    shutil.rmtree(f'{folder_path}train_set')
except FileNotFoundError:
    pass
try:
    shutil.rmtree(f'{folder_path}test_set')
except FileNotFoundError:
    pass
os.makedirs(f"{folder_path}train_set/cat")
os.makedirs(f"{folder_path}train_set/dog")
os.makedirs(f"{folder_path}test_set/cat")
os.makedirs(f"{folder_path}test_set/dog")
for file in train_set:
    # getting category
    split = file.split(".")
    dir_name = split[0].lower()
    file_name = dir_name+"_"+split[1]+"."+split[2]
    # copying to correct folder
    shutil.copy(f"{folder_path}data/{file}",
              f"{folder_path}/train_set/{dir_name}/{file_name}")
for file in test_set:
    # getting category
    split = file.split(".")
    dir_name = split[0].lower()
    file_name = dir_name+"_"+split[1]+"."+split[2]
    # copying to correct folder
    shutil.copy(f"{folder_path}data/{file}",
              f"{folder_path}/test_set/{dir_name}/{file_name}")