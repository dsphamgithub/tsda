"""
Extended from create_sequences.py. A module containing functions used to generate a sequence of images through
interpolation of a foreground over a background via perspective projection. Creates the illusion of changing distance.

Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.

The distance of the near and far anchor points are passed as command line arguments, and a set of foregrounds, with
evenly distributed distances over the near and far anchor distances, are produced via perspective projection. 

The camera's focal length or field of view is also passed as a command line argument to apply perspective projection.
"""

MIN_ANCHOR_SIZE = 10
MAX_ANCHOR_SIZE = 200
FOVY = 60
NEAR_CLIPPING_PLANE_DIST = 2
FAR_CLIPPING_PLANE_DIST = 50
VANISHING_POINT_TRANSLATION = 0.55

SIGN_COORDS = {'x':1.5, 'y':1, 'height':0.5}  # Example world coordinates for rendered sign objects

import numpy as np
import math


class Anchor(object):
    """
    A class that stores OpenCV pixel coordinates of the top left corner of the sign, 
    as well as the width and height of the sign in pixels.
    """
    def __init__(self, bg_size, NDC_x, NDC_y, x_size, y_size):
        height, width, _ = bg_size
        x_size, y_size = x_size / 2, y_size / 2
        
        # Converting from normalized device coordinates to 0-1 range
        x_prop = (NDC_x + 1) / 2
        x_prop = min(max(x_prop, 0), 1 - x_size)
        
        # Converting from normalized device coordinates to 0-1 range and shifting
        # vanishing point vertically
        y_prop = 1 - ((NDC_y + 1 - VANISHING_POINT_TRANSLATION) / (2 - VANISHING_POINT_TRANSLATION))
        y_prop = min(max(y_prop, 0), 1 - y_size)
        
        # Converting from 0-1 range to pixel coordinates
        self.width = int(x_size * width)
        self.height = int(y_size * height)
        self.screen_x = int(x_prop * width)
        self.screen_y = int(y_prop * height)
        
    def __str__(self):
        return f"Anchor: {self.screen_x}, {self.screen_y}, {self.height}, {self.width}"
      
        
class SignObject(object):
    """
    A class that stores the world coordinates of a sign object, applying perspective projection via
    the method 'perspective_transform'
    """
    def __init__(self, x, y, z, dims):
        width, height = dims
        # Top left corner of sign
        self.x1 = x 
        self.y1 = y 
        # Bottom right corner of sign
        self.x2 = x + width
        self.y2 = y - height
        # Virtual distance from camera to sign
        self.z = z
        
    def perspective_transform(self, bg_size, proj_matrix):
        """
        Applies perspective projection to the sign object to produce normalised device coordinates (NDC), 
        and initialises and returns an anchor object using these new coordinates.
        """
        # Scale the sign coordinates by the perspective projection matrix
        x1, y1, _, w = proj_matrix.dot(np.array([self.x1, self.y1, self.z, 1]))
        x2, y2, _, w = proj_matrix.dot(np.array([self.x2, self.y2, self.z, 1]))
        
        # Transform to normalized device coordinates
        x1f, y1f, x2f, y2f = x1/w, y1/w, x2/w, y2/w
        
        x_size = abs(x2f - x1f)
        y_size = abs(y2f - y1f)
        return Anchor(bg_size, NDC_x=x1f, NDC_y=y1f, x_size=x_size, y_size=y_size)
    
    
def create_perspective(fovy, aspect, near, far):
    """[summary]
    Creates a perspective projection matrix using vertical fov, aspect ratio, and near and
    far clipping distances. 
    Code adapted from http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """
    if (fovy <= 0 or fovy >= 180 or aspect <= 0 or near >= far or near <= 0):
        raise ValueError('Invalid parameters to createPerspective')
    half_fovy = math.radians(fovy) / 2
    
    top = near * math.tan(half_fovy)
    bottom = -top
    right = top * aspect
    left = -right
    return create_frustrum(left, right, bottom, top, near, far)


def create_frustrum(left, right, bottom, top, near, far):
    """[summary]
    Creates a perspective projection matrix using the left, right, bottom, top, near and far
    bounds of the near clipping plane.
    Code adapted from http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """
    if (left == right or bottom == top or near == far):
        raise ValueError('Invalid parameters to createFrustrum')
    if (near <= 0 or far <= 0):
        raise ValueError('Near and far must be positive')
    
    perspective_matrix =  np.array([[near, 0, 0, 0],
                                    [0, near, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, -1, 0]], 
                                    dtype=np.float32) 
    NDC_matrix =  np.array([[2/(right-left), 0, 0, 0],
                            [0, 2/(top-bottom), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            dtype=np.float32)
    return np.dot(NDC_matrix, perspective_matrix)
    

def produce_anchors(bg_shape, x, y, sign_dims, min_dist, max_dist, num_frames):
    """[summary]
    Generates a list of anchor objects, by applying perspective projection on the constant
    world coordinates, depending on virtual distance from the camera. Using np.linspace, the 
    distances are evenly distributed over min_dist to max_dist to create the anchor objects.
    """
    anchors = []
    height, width, _ = bg_shape
    aspect_ratio = width / height
    proj_matrix = create_perspective(FOVY, aspect_ratio, NEAR_CLIPPING_PLANE_DIST, FAR_CLIPPING_PLANE_DIST)
    
    for dist in np.linspace(max_dist, min_dist, num=num_frames, endpoint=True):
        sign_z = -1 * dist  # Projection matrix assumes negative z values in front of camera
        sign_near = SignObject(x=x, y=y, z=sign_z, dims=sign_dims)
        anchor = sign_near.perspective_transform(bg_shape, proj_matrix)
        anchors.append(anchor)
    return anchors


def get_world_coords(aspect, x_prop, y_prop, z, fg_dims):
    """[summary]
    Get the location in word coordinates of the sign located z distance from the camera.
    Args:
        fovy: vertical field of view
        aspect: width / height of the screen
        coords: x proportion, y proportion, z value => where x and y coordinates are proportion to screen
        fg_dims: y size, x size => where sizes are proportional to screen
    """
    x_size, y_size = fg_dims
    
    half_fovy = math.radians(FOVY) / 2
    top = z * math.tan(half_fovy)
    right = top * aspect
    
    NDC_x = x_prop * 2 - 1  # [0, 1] --> [-1, 1]
    NDC_y = (1 - y_prop) * (2 - VANISHING_POINT_TRANSLATION) + VANISHING_POINT_TRANSLATION - 1
    
    x_world = right * NDC_x
    y_world = top * NDC_y
    
    y_wsize = 2 * top * y_size
    x_wsize = y_wsize * (x_size / y_size)
    
    return x_world, y_world, x_wsize, y_wsize
    
    