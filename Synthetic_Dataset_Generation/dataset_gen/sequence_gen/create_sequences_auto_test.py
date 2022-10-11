"""Testing module to view a sequence of foregrounds interpolated over the same background, and interactively change
parameters.
"""

import os
import sys
import cv2
import math
from dataset_gen.sequence_gen.create_sequences_auto import produce_anchors, get_world_coords, SIGN_COORDS

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

from signbreaker.utils import overlay

X_TRACKBAR_MAX = 200
Y_TRACKBAR_MAX = 200 
SIZE = 0.15


class ShowAnchors(object):
    """[summary]
    A class which is a wrapper for the functions needed to show num_frames projected
    signs at varying distances from the camera, using cv2.imshow
    """
    def __init__(self, bg_path, fg_path, min_dist, max_dist, num_frames):
        self.bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        self.fg_img = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_frames = num_frames
        
        self.display_img = self.bg_img.copy()
        height, width, _ = self.bg_img.shape
        self.aspect_ratio = width / height
        
        self.x_prop = 0.75
        self.y_prop = 0.45
        
        self.__draw_anchors(0.75, 0.45)
        
        cv2.imwrite('overlayed_sequence_auto.png', self.display_img)
        cv2.createTrackbar('x', 'image', 0, X_TRACKBAR_MAX, self.__x_on_change)
        cv2.createTrackbar('y', 'image', 0, Y_TRACKBAR_MAX, self.__y_on_change)
        cv2.waitKey(0)


    def __draw_anchors(self, x, y):
        self.display_img = self.bg_img.copy()
        res = get_world_coords(self.aspect_ratio, x, y, self.min_dist, (SIZE, SIZE))
        world_x, world_y, x_wsize, y_wsize = res
        anchors = produce_anchors(self.bg_img.shape, world_x, world_y, (x_wsize, y_wsize), 
                                  self.min_dist, self.max_dist, self.num_frames)
        
        for anchor in anchors:
            sign_img_scaled = cv2.resize(self.fg_img, (anchor.height, anchor.width))
            self.display_img = overlay(sign_img_scaled, self.display_img, anchor.screen_x, anchor.screen_y)
        cv2.imshow('image', self.display_img)
        
        
    def __x_on_change(self, val):
        self.x_prop = val / X_TRACKBAR_MAX
        self.__draw_anchors(self.x_prop, self.y_prop)
        
        
    def __y_on_change(self, val):
        self.y_prop = val / Y_TRACKBAR_MAX
        self.__draw_anchors(self.x_prop, self.y_prop)   
       

if __name__ == '__main__':
    os.chdir(os.path.join(current_dir, '../../'))
    
    bg_path = './signbreaker/Backgrounds/GTSDB/00014.png'
    fg_path = './signbreaker/Sign_Templates/1_Input/1.png'
    anchors = ShowAnchors(bg_path, fg_path, min_dist=4, max_dist=20, num_frames=8)