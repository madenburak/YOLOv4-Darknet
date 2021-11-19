# we will import all required libraries for this tutorial in advance
#labels must be csv

import imgaug as ia

ia.seed(1)

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil

def grouped_df(csv_file_path):
    labels_df = pd.read_csv(csv_file_path)

    return labels_df

def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

def image_augment(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
                             )
    grouped = df.groupby('filename')
    
    for filename in df['filename'].unique():
    #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)   
    #   read the image
        image = imageio.imread(images_path+filename)
    #   get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    #   disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
    #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
    #   don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        
    #   otherwise continue
        else:
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])            
    
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


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


def create_folder(folder_path):
    """
    Create folder safety. 
    Input:
        folder_path: String. folder_path/folder_name
    """
    try:
        os.mkdir(folder_path)
        print("Successfuly created the directory", folder_path)
    except FileExistsError:
        print("Folder is already exist", folder_path)

path = "D:/Dataset/train/"

LABEL_PATH = path + "train_data_labels.csv"
IMG_PATH = path# + "train_data/"
AUG_IMG_PATH = path + "train_data_aug/"
ALL_IMG_PATH = path + "train_data_all/"

create_folder(AUG_IMG_PATH)
create_folder(ALL_IMG_PATH)

labels_df =grouped_df(LABEL_PATH)

method_list = [iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
                   iaa.Affine(scale=(1.0, 1.5)),
                   iaa.Fliplr(1),
                   iaa.Flipud(1),
                   iaa.Affine(rotate=(-7, 7)),
                   iaa.Multiply((0.90, 1.05))
                   ]

frames = list()

for i, method in enumerate(method_list):

    aug_method = iaa.OneOf([method])
    output_name = "aug{}_".format(str(i))
    augmented_images_df = image_augment(labels_df,IMG_PATH,AUG_IMG_PATH,output_name,aug_method)
    frames.append(augmented_images_df)

aug_labels_df = pd.concat(frames)
all_labels_df = pd.concat([labels_df,aug_labels_df])
all_labels_df.to_csv(path + "data_all_labels.csv", index=False)

main_img_list = get_filepaths(IMG_PATH,".bmp")
aug_img_list = get_filepaths(AUG_IMG_PATH, ".bmp")

copy_files_from_list(main_img_list,ALL_IMG_PATH)
copy_files_from_list(aug_img_list,ALL_IMG_PATH)