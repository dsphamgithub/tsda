"""Module of functions to apply damage to signs."""
# Extended from original functions by Jack Downes: https://github.com/ai-research-students-at-curtin/sign_augmentation

import os
import random as rand
import math
import ntpath
from pathlib import Path

import imutils
import numpy as np
import cv2 as cv
from skimage import draw
import scipy.stats as stats

from utils import overlay, calc_damage, calc_damage_ssim, calc_damage_sectors, sectors_no_damage, calc_ratio, remove_padding, pad, get_truncated_normal
from synth_image import SynthImage

attributes = {
    "damage_type"  : "None",
    "tag"          : "-1",   # Set of parameters used to generate damage as string 
    "damage_ratio" : "0.0",  # Quantity of damage (0 for no damage, 1 for all damage)
    }

dmg_measure = "pixel_wise"
num_sectors = 4


def damage_image(synth_img, output_dir, config, backgrounds=[], single_image=False):
    """Applies all the different types of damage to the imported image, saving each one"""
    damaged_images = []
    img = cv.imread(synth_img.fg_path, cv.IMREAD_UNCHANGED)
    img = img.astype('uint8')
    
    # Create file writing info: filename, class number, output directory, and labels directory
    _, filename = ntpath.split(synth_img.fg_path)  # Remove parent directories to retrieve the image filename
    class_num, _ = filename.rsplit('.', 1)  # Remove extension to get the sign/class number

    output_path = os.path.join(output_dir, class_num)
    # Create the output directory if it doesn't exist already
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_dmgs = config['num_damages']
    global dmg_measure
    global num_sectors
    dmg_measure = config['damage_measure_method']
    num_sectors = config['num_damage_sectors']

    def apply_damage(dmg, att):
        """Helper function to avoid repetition."""
        # print('old path', synth_img.fg_path)
        # print('output_path', output_path)
        dmg_path = os.path.join(output_path, f"{class_num}_{att['damage_type']}")
        if att['tag']:
            dmg_path += f"_{att['tag']}.png"
        else:
            dmg_path += ".png"
        synth_dmg = synth_img.clone()
        synth_dmg.set_damage(
            att["damage_type"],
            att["tag"],
            float(att["damage_ratio"]),
            att["sector_damage"]
        )
        synth_dmg.set_fg_image(dmg)
        if single_image:
            return synth_dmg
        else:
            synth_dmg.set_fg_path(dmg_path)
            cv.imwrite(dmg_path, dmg)
            damaged_images.append(synth_dmg)

    total_p = 0
    if single_image is True:
        for dmg_type in config['num_damages']:
            if dmg_type != 'online':
                total_p += config['num_damages'][dmg_type]
    p = rand.random()
    p_thresh = 0.0

    # ORIGINAL UNDAMAGED
    if single_image:
        p_thresh += n_dmgs['no_damage'] / total_p
        if p < p_thresh:
            dmg, att = no_damage(img)
            return apply_damage(dmg, att)
    elif n_dmgs['no_damage'] > 0 or n_dmgs['online'] is True:
        dmg, att = no_damage(img)
        apply_damage(dmg, att)

    # Only return undamaged sign so that flow of program continues
    if n_dmgs['online'] is True and single_image is False:
        return damaged_images

    # QUADRANT
    if single_image:
        p_thresh += n_dmgs['quadrant'] / total_p
        if p < p_thresh:
            dmg, att = remove_quadrant(img, -1)
            return apply_damage(dmg, att)
    elif n_dmgs['quadrant'] > 0:
        quad_nums = rand.sample(range(1, 5), min(n_dmgs['quadrant'], 4))
        for n in quad_nums:
            dmg, att = remove_quadrant(img, n)
            apply_damage(dmg, att)
    
    # BIG HOLE
    if single_image:
        p_thresh += n_dmgs['big_hole'] / total_p
        if p < p_thresh:
            dmg, att = remove_hole(img, -1)
            return apply_damage(dmg, att)
    elif n_dmgs['big_hole'] > 0:
        angles = rand.sample(
            range(0, 360, 20), n_dmgs['big_hole'])
        for a in angles:
            dmg, att = remove_hole(img, a)
            apply_damage(dmg, att)

    # GRAFFITI
    g_conf = config['graffiti']
    if single_image:
        p_thresh += n_dmgs['graffiti'] / total_p
        if p < p_thresh:
            l, u = 0.0, g_conf['max']
            mu, sigma = g_conf['max']/4, g_conf['max']/3
            X = stats.truncnorm(
                (l - mu) / sigma, (u - mu) / sigma, loc=mu, scale=sigma)
            dmg, att = graffiti(img, target=X.rvs(1)[0], color=(0,0,0), solid=g_conf['solid'])
            return apply_damage(dmg, att)
    elif n_dmgs['graffiti'] > 0:
        targets = np.linspace(g_conf['initial'], g_conf['final'], n_dmgs['graffiti'])
        for t in targets:
            dmg, att = graffiti(img, target=t, color=(0,0,0), solid=g_conf['solid'])
            apply_damage(dmg, att)

    return damaged_images


def validate_sign(img):
    """Ensure sign image fits within the parameters of valid signs."""
    has_alpha = (img.shape[2] == 4)
    if has_alpha:
        return img
    else:
        raise ValueError


def no_damage(img):
    """Return the image as is, along with its attributes."""
    dmg = validate_sign(img)
    # dmg = cv.bitwise_and(img, img, mask=dmg[:,:,3])

    # Assign labels
    att = attributes
    att["damage_type"]   = "no_damage"
    att["tag"]           = ""
    att["damage_ratio"]  = "0.0"  # This should be 0.0
    att["sector_damage"] = sectors_no_damage(num_sectors)

    return dmg, att

def remove_quadrant(img, quad_num=-1):
    """Make one random quandrant transparent."""
    dmg = validate_sign(img)
    quadrant = dmg[:,:,3].copy()

    height, width, _ = img.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    if quad_num == -1:
        quad_num = rand.randint(1, 4)
    # Remove the quadrant: -1 offset is necessary to avoid damaging part of a wrong quadrant
    if quad_num == 1:         # top-right          centre
        cv.rectangle(quadrant, (width, 0), (centre_x, centre_y-1), 0, thickness=-1)
    elif quad_num == 2:       # top-left           centre
        cv.rectangle(quadrant, (0, 0), (centre_x-1, centre_y-1), 0, thickness=-1)
    elif quad_num == 3:       # bottom-left        centre
        cv.rectangle(quadrant, (0, height), (centre_x-1, centre_y), 0, thickness=-1)
    elif quad_num == 4:       # bottom-right       centre
        cv.rectangle(quadrant, (width, height), (centre_x, centre_y), 0, thickness=-1)
    
    dmg = cv.bitwise_and(img, img, mask=quadrant)

    # Assign labels
    att = attributes
    att["damage_type"]   = "quadrant"
    att["tag"]           = str(quad_num)
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))  # This should be around 0.25
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def remove_hole(img, angle=-1):
    """Remove a circle-shaped region from the edge of the sign."""
    dmg = validate_sign(img)
    hole = dmg[:,:,3].copy()
    att = attributes

    height, width, _ = dmg.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    if angle == -1:
        angle = rand.randint(0, 359)
    radius = int(2 * height / 5)
    rad = -(angle * math.pi / 180)  # Radians
    x = centre_x + int(radius * math.cos(rad))  # x-coordinate of centre
    y = centre_y + int(radius * math.sin(rad))  # y-coordinate of centre

    cv.circle(hole, (x,y), radius, (0,0,0), -1)  # -1 to create a filled circle
    dmg = cv.bitwise_and(dmg, dmg, mask=hole)

    # Assign labels
    att = attributes
    att["damage_type"]   = "big_hole"
    att["tag"]           = str(int(angle))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att


### The following methods are all for the 'graffiti' damage type ###

def in_bounds(img, xx, yy):
    """Return true if (xx,yy) is in bounds of the sign."""
    height, width, _ = img.shape
    alpha_ch = cv.split(img)[3]
    in_bounds = False

    # Check that the points are within the image
    if (xx > 0) and (xx < width) and (yy > 0) and (yy < height):
        # Test the opacity to check if it's in bounds of the sign
        if alpha_ch[yy][xx] > 0:
            in_bounds = True
    return in_bounds

def calc_points(img, x0, y0, offset):
    """Caclulate a random midpoint and endpoint for the bezier curve."""
    # Choose the midpoint for the bezier curve
    x1, y1 = 0, 0
    while not in_bounds(img, x1, y1):
        x1 = x0 + rand.randint(-offset, offset)
        y1 = y0 + rand.randint(-offset, offset)

    # Choose the end point
    x2, y2 = 0, 0
    while not in_bounds(img, x2, y2):
        x2 = x1 + rand.randint(-offset, offset)
        y2 = y1 + rand.randint(-offset, offset)
    
    return x1, y1, x2, y2

def draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color, solid):
    """Draw a single bezier curve of desired thickness and colour."""
    from skimage import morphology

    if solid:
        start = -(thickness // 2)
        end = thickness // 2
        for ii in range(start, end+1):
            # Draw curves, shifting vertically and adjusting x-coordinates to round the edges
            yy, xx = draw.bezier_curve(y0+ii, x0+abs(ii), y1+ii, x1, y2+ii, x2-abs(ii), weight=1, shape=grft.shape)
            
            grft[yy, xx] = 255  # Modify the colour of the pixels that belong to the curve
        # Dilate the image to fill in the gaps between the bezier lines
        # grft = morphology.dilation(grft, morphology.disk(radius=1))
    else:
        alpha = rand.randint(240, 255)  # Random level of transparency
        # skimage.draw.bezier_curve() only draws curves of thickness 1 pixel,
        # We layer multiple curves to produce a curve of desired thickness
        start = -(thickness // 2)
        end = thickness // 2
        for ii in range(start, end+1):
            # Draw curves, shifting vertically and adjusting x-coordinates to round the edges
            yy, xx = draw.bezier_curve(y0+ii, x0+abs(ii), y1+ii, x1, y2+ii, x2-abs(ii), weight=1, shape=grft.shape)
            
            grft[yy, xx] = color + (alpha,)  # Modify the colour of the pixels that belong to the curve
    return grft

def graffiti(img, target=0.2, color=(0,0,0), solid=True):
    """Apply graffiti damage to sign.
       :param initial: the first target level of obscurity (0-1)
       :param final: the level of obscurity to stop at (0-1)
       :returns: a list containing the damaged images, and a list with the corresponding attributes
    """
    from skimage import morphology

    validate_sign(img)
    height, width, _ = img.shape
    if solid:
        grft = np.zeros((height, width), dtype=np.uint8)  # New blank image for drawing the graffiti on.
    else:
        grft = np.zeros((height, width, 4), dtype=np.uint8)  # New blank image for drawing the graffiti on.

    ratio = 0.0
    x0, y0 = width//2, height//2  # Start drawing in the centre of the image
    
    # Keep drawing bezier curves until the obscurity hits the target
    while ratio < target:
        radius = width // 5  # Radius of max distance to the next point
        x1, y1, x2, y2 = calc_points(img, x0, y0, radius)
        thickness = int(round(width // 20))
        grft = draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color, solid)
        # Make the end point the starting point for the next curve
        x0, y0 = x2, y2
        if solid:
            grft_dilated = morphology.dilation(grft, morphology.disk(radius=1))
            ratio = round(calc_ratio(grft_dilated, img), 4)
        else:
            ratio = round(calc_ratio(grft, img), 4)
    # Add a copy to the list and continue layering more graffiti
    
    if solid:
        grft = grft_dilated
        grft = cv.cvtColor(grft, cv.COLOR_GRAY2BGRA)

    grft[:,:,3] = cv.bitwise_and(grft[:,:,3], img[:,:,3])  # Combine with original alpha to remove any sign spillover

    if solid:
        grft = cv.bitwise_and(img, img, mask=grft_dilated)
        alpha = cv.split(grft)[3]
        grft[alpha == 255] = color + (255,)

    # Apply a Gaussian blur to each image, to smooth the edges
    k = (int(round( width/30 )) // 2) * 2 + 1  # Kernel size must be odd
    grft = cv.GaussianBlur(grft, (k,k), 0)
    dmg = overlay(grft, img)
    
    # Assign labels
    att = attributes
    att["damage_type"]   = "graffiti"
    att["tag"]           = str(round(target, 3))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)
    return dmg, att
