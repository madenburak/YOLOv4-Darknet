"""
Divides raw_data folder as val_data and train data  
Folder structre is : 
raw_data:
    |___01.jpg
    |___01.xml
    |___02.jpg
    |___02.xml
    |___...
    |___... 
Output:
    output/test_data:
        |___02.jpg
        |___02.xml
        |___17.jpg
        |___17.xml
        |___...
        |___...   
    output/train_data:
        |___01.jpg
        |___01.xml
        |___03.jpg
        |___03.xml
        |___...
        |___... 
"""

import glob
import os 
import random
import shutil

def get_filepaths(raw_data_path,extention_type):
    """
    Collect data's names as a list.
    
    Input:
        raw_data_path: string. Main data path that has .jpg and .xml files.
        extention_type: string. Ex: ".jpg", ".xml"
    Return:
        filename_list: list. list of the data's path.
    """
    filename_list = list()

    filename_list = glob.glob(raw_data_path + '/*{}'.format(extention_type))
    #for dirpath, dirnames, filenames in os.walk(raw_data_path):
    #    for fl in filenames:
    #        filename_list.append(fl)

    return filename_list



def split(split_value,paths_list):
    """
    Divide the folder as much as the partition value. 
    Input: 
        split_value: Int. 20 ,10 
        paths_list: List. 
    
    Return: 
        val_list: List. Splitted list. List len is %split_value of the paths_list.
        train_list: List. paths_list - val_list.
    """
    val_list = list()

    random.shuffle(paths_list)
    splitted_value = int((len(paths_list)/100) * split_value)
    val_list = paths_list[:splitted_value]
    train_list = paths_list[splitted_value:]

    return val_list, train_list

def create_folder(folder_path):
    """
    Create folder safety. 
    Input:
        foler_path: String. folder_path/folder_name
    """
    try:
        os.mkdir(folder_path)
        print("Successfuly created the directory", folder_path)
    except FileExistsError:
        print("Folder is already exist", folder_path)

def get_only_filename(file_list):
    """
    Get filename from file's path and return list that has only filename.
    Input: 
        file_list: List. file's paths list.
    Attribute:
        file_name: String. "01.jpg"
        file_name_without_ext: String. "01"
    Return: 
        filename_list: Only filename list.
    """

    filename_list = list()

    for file_path in file_list:
        file_name = file_path.split("/")[-1]
        file_name_without_ext = file_name.split(".")[0]
        filename_list.append(file_name_without_ext)
    
    return filename_list

def copy_files_from_list(main_list,target_folder):
    """
    Moves files to the target path.
    Input:
        main_list: List. Source of the file's path in the list. 
        target_folder: String. Target folder.
    """

    for file_path in main_list:
        file_name = file_path.split("/")[-1]
        target_path = target_folder + file_name
        shutil.copy(file_path,target_path)


PATH = "D:/Dataset/raw_data"
OUTPUT_PATH = "D:/Dataset/output"
VAL_PATH = "D:/Dataset/output/val_data"
TRAIN_PATH = "D:/Dataset/output/train_data"

raw_data_path = os.path.join(os.getcwd(), (PATH))
create_folder(OUTPUT_PATH)
create_folder(TRAIN_PATH)
create_folder(VAL_PATH)

img_list = get_filepaths(raw_data_path,".jpg")
label_list = get_filepaths(raw_data_path,".xml")

val_img_list, train_img_list = split(20,img_list)

val_imgname_list = get_only_filename(val_img_list)
all_labelnames_list = get_only_filename(label_list)

val_label_list = list()
train_label_list = list()

for val_imgname in val_imgname_list:
    if val_imgname in all_labelnames_list:
        index_val = all_labelnames_list.index(val_imgname)
        val_label_list.append(label_list[index_val])
    
    else:
        print("{} images hasn't label.".format(val_imgname))
        exit(1)

train_label_list = [i for i in label_list if i not in val_label_list]

all_list = [train_img_list, train_label_list, val_img_list, val_label_list]

for i in all_list:
    if i == train_img_list or i == train_label_list:
        target_path = TRAIN_PATH + "/"
    else:
        target_path = VAL_PATH + "/"
    
    copy_files_from_list(i,target_path)



print("Done")