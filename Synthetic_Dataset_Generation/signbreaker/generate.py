"""Module of functions for generating a synthetic manipulated sign dataset."""

import os
from utils import load_paths, dir_split, overlay, overlay_new
from datetime import datetime
import imutils
import cv2
import random
import numpy as np

from synth_image import SynthImage

def __has_opaque_pixel(line):
    """Checks if a line of pixels contains a pixel above a transparency threshold."""
    opaque = False
    for pixel in line:
        # Check if pixel is opaque
        if pixel[3] > 200:
            opaque = True
            break  # Stop searching if one is found
    return opaque

def __bounding_axes(img):
    """Returns the bounding axes of an image with a transparent background."""
    # Top axis
    y_top = 0
    for row in img:  # Iterate through each row of pixels, starting at top-left
        if __has_opaque_pixel(row) is False:  # Check if the row has an opaque pixel
            y_top += 1  # If not, move to the next row
        else:
            break  # If so, break, leaving y_top as the bounding axis

    # Bottom axis
    height = img.shape[0]
    y_bottom = height - 1
    for row in reversed(img):  # Iterate from the bottom row up
        if __has_opaque_pixel(row) is False:
            y_bottom -= 1
        else:
            break

    # Left axis
    # Rotate 90 degrees to iterate through what were originally columns
    r_img = imutils.rotate_bound(img, 90)
    x_left = 0
    for column in r_img:
        if __has_opaque_pixel(column) is False:
            x_left += 1
        else:
            break

    # Right axis
    r_height = r_img.shape[0]
    x_right = r_height - 1
    for column in reversed(r_img):
        if __has_opaque_pixel(column) is False:
            x_right -= 1
        else:
            break

    ## For debug
    # img[y_top, :] = (255, 0, 0, 255)
    # img[y_bottom, :] = (255, 0, 0, 255)
    # img[:, x_left] = (255, 0, 0, 255)
    # img[:, x_right] = (255, 0, 0, 255)
    # cv2.imwrite(image_dir, img)
    ##

    ##
    # For debug (place outside this function)
    # for img in load_paths("Traffic_Signs_Templates/4_Transformed_Images/0/0_ORIGINAL"):
    #     bounding_axes(img)  
    ##  

    return [x_left, x_right, y_top, y_bottom]

def new_data(synth_image, online=False):
    """Blends a synthetic sign with its corresponding background."""
    bg_path = synth_image.bg_path
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    assert bg is not None, "Background image not found"

    fg_path = synth_image.fg_path
    if online is True:
        fg = synth_image.fg_image
    else:
        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    assert fg is not None, "Foreground image not found"

    if synth_image.fg_coords is not None and synth_image.fg_size is not None:
        x, y = synth_image.fg_coords
        new_size = synth_image.fg_size
    else:
        x, y, new_size = SynthImage.gen_sign_coords(bg.shape[:2], fg.shape[:2])

    # pad = 40  # pixels to pad on each side of the image
    # fg = cv2.copyMakeBorder(fg, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    # if(np.random.randint(5) < 10):  # 50% chance of rotating
    #     angle = int(np.random.normal(0,0.5)*180)  # Normal distribution of rotation angle
    #     # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point  
    #     image_center = tuple(np.array(fg.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #     fg = cv2.warpAffine(fg, rot_mat, fg.shape[1::-1], flags=cv2.INTER_LINEAR)

    image = overlay_new(fg, bg, new_size, x, y)
    fg = cv2.resize(fg, (new_size, new_size))
    axes = __bounding_axes(fg)  # Retrieve bounding axes of the sign image
    axes[0] += x  # Adjusting bounding axis to make it relative to the whole bg image
    axes[1] += x
    axes[2] += y
    axes[3] += y
    synth_image.bounding_axes = axes
    return image


#TODO: These two functions should be one function always include background using classes?
# List of paths for all SGTSD relevant files using exposure_manipulation
def paths_list(imgs_directory, bg_directory):
    directories = []
    for places in load_paths(imgs_directory):  # List of places: originally either UK_rural or UK_urban
        for imgs in load_paths(places):  # Folder for each bg image: eg. IMG_0
            dr = dir_split(imgs)
            bg = os.path.join(bg_directory, dr[-2], dr[-1] + ".png")  # Retrieving relevant bg image
            for signs in load_paths(imgs):  # Folder for each sign type: eg. SIGN_9
                for dmgs in load_paths(signs):  # Folder for each damage type: eg. 0_HOLES
                    for png in load_paths(dmgs):
                        directories.append([png, bg])
    return directories  # Directory for every single FILE and it's relevant bg FILE

# List of paths for all SGTSD relevant files using fade_manipulation; backgrounds are assigned to 
def assigned_paths_list(imgs_directory, bg_directory):  #TODO: is this the same as above?
    directories = []
    for places in load_paths(bg_directory):  # Folder for each place: eg. GTSDB
        for imgs in load_paths(places):  # Iterate through each b.g. image: eg. IMG_0
            for signs in load_paths(imgs_directory):  # Folder for each sign type: eg. SIGN_9
                for dmgs in load_paths(signs):  # Folder for each damage type: eg. 9_HOLES
                    for png in load_paths(dmgs):
                        directories.append([png, imgs])
    return directories  # Directory for every single FILE and its relevant bg FILE

# Paths of images needed to generate examples for 'sign' with damage 'dmg'
def list_for_sign_x(sign, dmg, directories):
    l = []
    for elements in directories:
        foreground = dir_split(elements[0])
        if (foreground[-2] == sign + dmg):  # Eg. if (9_YELLOW == 4_ORIGINAL)
            l.append(elements)
    return l  # Directory for every single sign and its relevant background image