"""
This Module contains functions,
supporting you with the usual image operations,
loading and saving.
"""
import os
import numpy as np
import cv2

def load_images_labels_and_paths(root_path):
    """
    Returns the images from the subfolders which represents the labels.
    Returns also those labels and full paths.

    Keyword arguments:
    root_path -- the root path which contains images organized in folders by their classes.
    """


    image_data = []
    image_labels = []
    corresponding_paths = []

    root_sub_dirs = os.listdir(root_path)
    for root_sub_dir in root_sub_dirs:
        root_sub_path = os.path.join(root_path, root_sub_dir)
        if os.path.isdir(root_sub_path):
            picture_names = os.listdir(root_sub_path)
            for picture_name in picture_names:
                if ".DS_Store" not in picture_name:
                    picture_path = os.path.join(root_sub_path, picture_name)
                    corresponding_paths.append(picture_path)
                    image_data.append(cv2.imread(picture_path))
                    image_labels.append(root_sub_dir)

    return image_data, image_labels, corresponding_paths

def save_images_by_label(root_path, images, labels):
    """
    Saves images categorized by labels.

    Keyword arguments:
    root_path -- location where the folderstructure should be created.
    images -- array of images matrix of shape (?,?,3).
    labels -- array of labels corresponding to "images".
    """

    for i in range(len(labels)):
        label_string = str(labels[i])
        subfolder_path = os.path.join(root_path, label_string)
        full_path = os.path.join(subfolder_path, label_string + "_" + str(i) + ".jpg")

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        cv2.imwrite(full_path, images[i])

def get_and_preprocess_image(image_path, long_edge_size=299):
    """
    Loads and preprocesses an image for inception v3 model.
    Not exactly a square, but should be ok.
    Resizes without altering aspect ratio.

    Keyword arguments:
    image_path -- Path of the image to process
    long_edge_size -- size of the long edge
    """
    image = cv2.imread(image_path)

    if image.shape[0] < long_edge_size and image.shape[1] < long_edge_size:
        return image

    height = long_edge_size
    width = long_edge_size
    if image.shape[0] < image.shape[1]:
        width = int(float(image.shape[0])/image.shape[1] * long_edge_size)
    else:
        height = int(float(image.shape[1])/image.shape[0] * long_edge_size)
    resized_image = cv2.resize(image, (height, width)) 
    return resized_image
