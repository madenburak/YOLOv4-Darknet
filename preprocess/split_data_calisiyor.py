import splitfolders

#There should be two folders inside. Displays without labels, one labels to one folder (junk, deleted after processing) to the other folder. They are separated and labels are added to them.

splitfolders.ratio("D:/Dataset", output="D:/Dataset/output", seed=42, ratio=(.8, .2, .0), group_prefix=None)#train_val_test

# import cv2
 
# # read image
# img = cv2.imread('10595.png', cv2.IMREAD_UNCHANGED)
 
# # get dimensions of image
# dimensions = img.shape
 
# # height, width, number of channels in image
# height = img.shape[0]
# width = img.shape[1]
# channels = img.shape[2]
 
# print('Image Dimension    : ',dimensions)
# print('Image Height       : ',height)
# print('Image Width        : ',width)
# print('Number of Channels : ',channels)
