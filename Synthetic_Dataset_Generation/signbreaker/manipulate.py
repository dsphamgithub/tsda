"""Module of functions to apply transformations to sign images."""
# Extended from original code by Alexandros Stergiou:
# (https://github.com/alexandrosstergiou/Traffic-Sign-Recognition-basd-on-Synthesised-Training-Data)

from abc import ABC, abstractmethod
import cv2
import numpy as np
import ntpath
import os
import math
from utils import load_paths, dir_split, get_truncated_normal
from PIL import Image, ImageStat, ImageEnhance
import random
from synth_image import SynthImage

# Open and validate config file
import yaml
with open("config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

##################################
###  TRANSFORMATION FUNCTIONS  ###
##################################
class AbstractTransform(ABC):
    """Image transformation template pattern abstract class."""
    def transform(self, input_image, output_path, num_transform):
        image_path = input_image.fg_path
        if output_path is None:
            img = input_image.fg_image
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.height, self.width, _ = img.shape

        self.num_transformed = 0
        tform_imgs = []  # Collection of transformed images
        for ii in range(num_transform):
            tform_img, descriptor = self.transformation(img)
            tform_imgs.append((tform_img, descriptor))
        assert self.num_transformed == num_transform, f"There were meant to be " \
            "{num_transform} transformations, but {self.num_transformed} occured"

        # Retrieve the filename to save as
        _, tail  = ntpath.split(image_path)  # Filename of img, parent directories removed
        title, _ = tail.rsplit('.', 1)  # Discard extension
        
        transformed_images = []  # Collection of transformed SynthImage objects
        for ii, tform_img_tuple in enumerate(tform_imgs):
            descriptor = tform_img_tuple[1]
            # Only return the transformed images
            if output_path is None:
                transformed_image = input_image.clone()
                transformed_image.set_fg_image(tform_img_tuple[0])
                transformed_image.set_transformation(descriptor)
                transformed_images.append(transformed_image)
            # Save to disk and return the transformed images
            else:
                save_dir = os.path.join(output_path, title)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # TODO: Make sure no two rotations are the same (or just add ii to filename)
                if descriptor is None:
                    descriptor = ii
                    save_path = os.path.join(save_dir, f"{descriptor}.png")
                else:
                    save_path = os.path.join(save_dir, f"{ii}_{descriptor}.png")
                cv2.imwrite(save_path, tform_img_tuple[0])

                transformed_image = input_image.clone()
                transformed_image.set_fg_path(save_path)
                transformed_image.set_transformation(descriptor)
                transformed_images.append(transformed_image)
        return transformed_images

    @abstractmethod
    def transformation(self, img):
        pass


class RotationTransform(AbstractTransform):
    """Author: Prasanna Asokan"""
    def transformation(self, img):
        SD_tilt = config['transforms']['tilt_SD']
        tilt_range = config['transforms']['tilt_range']
        SD_Y = config['transforms']['Z_SD']
        Z_range = config['transforms']['Z_range']
        angle = np.zeros(3)
        tilt = get_truncated_normal(mean=0, sd=SD_tilt, low=(-1)*tilt_range,
                                    upp=tilt_range)
        angle[0:2] = tilt.rvs(2)
        Z = get_truncated_normal(mean=0, sd=SD_Y, low=(-1)*Z_range, upp=Z_range)
        angle[2] = Z.rvs(1)
        
        if(img.shape[0] > 320):
            dz = img.shape[0]
            f = 200 + img.shape[0]/5
        else:
            dz = 320 - 0.25*(320-img.shape[0])
            f = 200
        f *= 2
        dz *= 2

        dest = self.rotate_image(img, angle[0], angle[1], angle[2], 0, 0, dz, f)
        self.num_transformed += 1
        return dest, f"{angle[0]:.2f}_{angle[1]:.2f}_{angle[2]:.2f}"
        
    def rotate_image(self, src, alpha, beta, gamma, dx, dy, dz, f):
        """Rotates any input image by a given angle in the x, y and z planes.
        :param src: the input image
        :param alpha: rotation around the x axis in degrees
        :param beta: rotation around the y axis in degrees
        :param gamma: rotation around the z axis (2d rotation) in degrees
        :param dx: translation around the x axis
        :param dy: translation around the y axis
        :param dz: translation around the z axis (distance to image)
        :param f: focal distance (distance between camera and image)
        referenced from
        http://jepsonsblog.blogspot.com/2012/11/rotation-in-3d-using-opencvs.html
        """
        # Convert to radians and start on x axis?
        alpha = math.radians(alpha)
        beta  = math.radians(beta)
        gamma = math.radians(gamma)

        # Get width and height for ease of use in matrices
        h, w = src.shape[:2]

        # Projection 2D -> 3D matrix
        A1 = np.array(
            [[1, 0, -w/2],
            [ 0, 1, -h/2],
            [ 0, 0, 1   ],
            [ 0, 0, 1   ]])

        # Rotation matrices around the X, Y, and Z axis
        xa1 = math.cos(alpha)
        xa2 = math.sin(alpha)
        RX = np.array(
            [[1, 0,   0,    0],
            [ 0, xa1, -xa2, 0],
            [ 0, xa2, xa1,  0],
            [ 0, 0,   0,    1]])
        ya1 = math.cos(beta)
        ya2 = math.sin(beta)
        RY = np.array(
            [[ya1, 0, -ya2, 0],
            [ 0,   1, 0,    0],
            [ ya2, 0, ya1,  0],
            [ 0,   0, 0,    1]])
        za1 = math.cos(gamma)
        za2 = math.sin(gamma)
        RZ = np.array(
            [[za1, -za2, 0, 0],
            [ za2,  za1, 0, 0],
            [ 0,    0,   1, 0],
            [ 0,    0,   0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation Matrix
        T = np.array(
            [[1, 0, 0, dx],
            [ 0, 1, 0, dy],
            [ 0, 0, 1, dz],
            [ 0, 0, 0, 1 ]])

        # 3D -> 2D matrix
        A2 = np.array(
            [[f, 0, w/2, 0],
            [ 0, f, h/2, 0],
            [ 0, 0, 1,   0]])

        # Final tranformation matrix
        trans = np.dot(A2, np.dot(T, np.dot(R, A1)))

        # Apply matrix transformation
        return cv2.warpPerspective(src, M=trans, dsize=(w,h), flags=cv2.INTER_LANCZOS4)


class FixedAffineTransform(AbstractTransform):
    def transformation(self, img):
        """Creates and saves different angles of the imported image.
        Adapted from code originally authored by Alexandros Stergiou.
        """
        width, height = self.width, self.height

        # Transform function names are numbered in order of (my subjective) significance in visual difference
        #[0] 0 FORWARD FACING
        def t0():
            return img

        #[3] 1 EAST FACING
        def t3():
            pts1 = np.float32( [[width/10,height/10], [width/2,height/10], [width/10,height/2]] )
            pts2 = np.float32( [[width/5,height/5], [width/2,height/8], [width/5,height/1.8]] )
            M = cv2.getAffineTransform(pts1,pts2)
            return cv2.warpAffine(img,M,(width,height))
        
        #[2] 2 NORTH-WEST FACING
        def t2():
            pts3 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts4 = np.float32( [[width*9/10,height/5], [width/2,height/8], [width*9/10,height/1.8]] )
            M = cv2.getAffineTransform(pts3,pts4)
            return cv2.warpAffine(img,M,(width,height))
        
        #[8] 3 LEFT TILTED FORWARD FACING
        def t8():
            pts5 = np.float32( [[width/10,height/10], [width/2,height/10], [width/10,height/2]] )
            pts6 = np.float32( [[width/12,height/6], [width/2.1,height/8], [width/10,height/1.8]] )
            M = cv2.getAffineTransform(pts5,pts6)
            return cv2.warpAffine(img,M,(width,height))
        
        #4 RIGHT TILTED FORWARD FACING (disabled: consistently gets cut off for UK templates)
        # pts7 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        # pts8 = np.float32( [[width*10/12,height/6], [width/2.2,height/8], [width*8.4/10,height/1.8]] )
        # M = cv2.getAffineTransform(pts7,pts8)
        # return cv2.warpAffine(img,M,(width,height))
        
        #[7] 5 WEST FACING (disabled: consistently gets cut off for GTSDB Wikipedia templates)
        # def t7():
        #     pts9  = np.float32( [[width/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        #     pts10 = np.float32( [[width/9.95,height/10], [width/2.05,height/9.95], [width*9/10,height/2.05]] )
        #     M = cv2.getAffineTransform(pts9,pts10)
        #     return cv2.warpAffine(img,M,(width,height))
        
        #[12] 6 RIGHT TILTED FORWARD FACING
        def t12():
            pts11 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts12 = np.float32( [[width*9/10,height/10], [width/2,height/9], [width*8.95/10,height/2.05]] )
            M = cv2.getAffineTransform(pts11,pts12)
            return cv2.warpAffine(img,M,(width,height))
        
        #[10] 7 FORWARD FACING W/ DISTORTION (disabled: consistently gets cut off for GTSDB Wikipedia templates)
        # def t10():
        #     pts13 = np.float32( [[width/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        #     pts14 = np.float32( [[width/9.8,height/9.8], [width/2,height/9.8], [width*8.8/10,height/2.05]] )
        #     M = cv2.getAffineTransform(pts13,pts14)
        #     return cv2.warpAffine(img,M,(width,height))
        
        #8 FORWARD FACING W/ DISTORTION 2 (disabled: consistently gets cut off for UK templates)
        # pts15 = np.float32( [[width/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        # pts16 = np.float32( [[width/11,height/10], [width/2.1,height/10], [width*8.5/10,height/1.95]] )
        # M = cv2.getAffineTransform(pts15,pts16)
        # return cv2.warpAffine(img,M,(width,height))
        
        #9 FORWARD FACING W/ DISTORTION 3 (disabled: consistently gets cut off for UK templates)
        # pts17 = np.float32( [[width/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        # pts18 = np.float32( [[width/11,height/11], [width/2.1,height/10], [width*10/11,height/1.95]] )
        # M = cv2.getAffineTransform(pts17,pts18)
        # return cv2.warpAffine(img,M,(width,height))
        
        #[11] 10 FORWARD FACING W/ DISTORTION 4 (disabled: consistently gets cut off for GTSDB Wikipedia templates)
        # def t11():
        #     pts19 = np.float32( [[width*9.5/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        #     pts20 = np.float32( [[width*9.35/10,height/9.99], [width/2.05,height/9.95], [width*9.05/10,height/2.03]] )
        #     M = cv2.getAffineTransform(pts19,pts20)
        #     return cv2.warpAffine(img,M,(width,height))
        
        #[14] 11 FORWARD FACING W/ DISTORTION 5 (disabled: consistently gets cut off for GTSDB Wikipedia templates)
        # def t14():
        #     pts21 = np.float32( [[width*9.5/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        #     pts22 = np.float32( [[width*9.65/10,height/9.95], [width/1.95,height/9.95], [width*9.1/10,height/2.02]] )
        #     M = cv2.getAffineTransform(pts21,pts22)
        #     return cv2.warpAffine(img,M,(width,height))
        
        #12 FORWARD FACING W/ DISTORTION 6 (disabled: consistently gets cut off for UK templates)
        # pts23 = np.float32( [[width*9.25/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
        # pts24 = np.float32( [[width*9.55/10,height/9.85], [width/1.9,height/10], [width*9.3/10,height/2.04]] )
        # M = cv2.getAffineTransform(pts23,pts24)
        # return cv2.warpAffine(img,M,(width,height))
        
        #[1] 13 SHRINK 1
        def t1():
            pts25 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts26 = np.float32( [[width*8/10,height/10], [width*1.34/3,height/10.5], [width*8.24/10,height/2.5]] )
            M = cv2.getAffineTransform(pts25,pts26)
            return cv2.warpAffine(img,M,(width,height))
        
        #[5] 14 SHRINK 2
        def t5():
            pts27 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts28 = np.float32( [[width*8.5/10,height*3.1/10], [width/2,height*3/10], [width*8.44/10,height*1.55/2.5]] )
            M = cv2.getAffineTransform(pts27,pts28)
            return cv2.warpAffine(img,M,(width,height))
        
        #[9] 15 FORWARD FACING W/ DISTORTION 7
        def t9():
            pts29 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts30 = np.float32( [[width*8.85/10,height/9.3], [width/1.9,height/10.5], [width*8.8/10,height/2.11]] )
            M = cv2.getAffineTransform(pts29,pts30)
            return cv2.warpAffine(img,M,(width,height))
        
        #[6] 16 FORWARD FACING W/ DISTORTION 8
        def t6():
            pts31 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts32 = np.float32( [[width*8.75/10,height/9.1], [width/1.95,height/8], [width*8.5/10,height/2.05]] )
            M = cv2.getAffineTransform(pts31,pts32)
            return cv2.warpAffine(img,M,(width,height))
        
        #[4] 17 FORWARD FACING W/ DISTORTION 9
        def t4():
            pts33 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts34 = np.float32( [[width*8.75/10,height/9.1], [width/1.95,height/9], [width*8.5/10,height/2.2]] )
            M = cv2.getAffineTransform(pts33,pts34)
            return cv2.warpAffine(img,M,(width,height))
        
        #[13] 18 FORWARD FACING W/ DISTORTION 10
        def t13():
            pts35 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts36 = np.float32( [[width*8.75/10,height/8], [width/1.95,height/8], [width*8.75/10,height/2]] )
            M = cv2.getAffineTransform(pts35,pts36)
            return cv2.warpAffine(img,M,(width,height))
        
        #[15] 19 FORWARD FACING W/ DISTORTION 11
        def t15():
            pts37 = np.float32( [[width*9/10,height/10], [width/2,height/10], [width*9/10,height/2]] )
            pts38 = np.float32( [[width*8.8/10,height/7], [width/1.95,height/7], [width*8.8/10,height/2]] )
            M = cv2.getAffineTransform(pts37,pts38)
            return cv2.warpAffine(img,M,(width,height))

        # Apply the number of transformations desired
        transforms = [t0,t1,t2,t3,t4,t5,t6,t8,t9,t12,t13,t15]
        if self.num_transformed == len(transforms):
            raise NotImplementedError(f"Only {len(transforms)} fixed affine transformations are currently implemented, but more were requested")

        transformed = transforms[self.num_transformed]()
        self.num_transformed += 1
        return transformed, None



################################
###  MANIPULATION FUNCTIONS  ###
################################
def find_image_exposures(paths, descriptor="sign"):
    """Determines the level of exposure for each image in the provided paths.
    Originally authored by Alexandros Stergiou.
    """
    ii = 0
    exposures = []
    for image_path in paths:
        print(f"Calculating {descriptor} exposures: {float(ii) / float(len(paths)):06.2%}", end='\r')
        exposures.append(find_image_exposure(image_path))
        ii += 1

    print(f"Calculating {descriptor} exposures: 100.0%\r")
    return exposures

def find_image_exposure(image_path):
    img_grey = Image.open(image_path).convert('LA')  # Greyscale with alpha
    img_rgba = Image.open(image_path)
    return image_exposure(img_grey, img_rgba)

def image_exposure(img_grey, img_rgba):
    stat1 = ImageStat.Stat(img_grey)
    # Average pixel brighness
    avg_grey = stat1.mean[0]
    # RMS pixel brighness
    rms_grey = stat1.rms[0]
    
    stat2 = ImageStat.Stat(img_rgba)
    # Perceived brightness using HSPcolour model, adjusting for degree of influence of each channel:

    rgba = stat2.mean  # Average pixels preceived brightness
    r, g, b = rgba[0], rgba[1], rgba[2]  # Ignore rgba[3] if it exists (alpha channel)
    avg_perceived = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

    rgba = stat2.rms  # RMS pixels perceived brightness
    r, g, b = rgba[0], rgba[1], rgba[2]
    rms_perceived = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

    return {
        'avg_grey': avg_grey,
        'rms_grey': rms_grey,
        'avg_perceived': avg_perceived,
        'rms_perceived': rms_perceived
    }


class AbstractManipulation(ABC):
    """Image brightness manipulation template pattern abstract class."""
    def manipulate(self, transformed_data, background_paths, out_dir):
        self.out_dir = out_dir
        sign_paths = [transformed.fg_path for transformed in transformed_data]

        pr = 0  # Progress
        pr_total = len(sign_paths) * len(background_paths)
        self.man_images = []
        for ii in range(len(sign_paths)):
            self.original_synth = transformed_data[ii]
            sign_path = sign_paths[ii]

            fg = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)

            for jj in range(0, len(background_paths)):
                print(f"Manipulating brightness of signs: {float(pr) / float(pr_total):06.2%}", end='\r')
                bg_path = background_paths[jj]
                self.bg_path   = bg_path
                self.sign_path = sign_path

                self.manipulation(fg)
                pr += 1

        print("Manipulating brightness of signs: 100.0%\r\n")
        return self.man_images

    def save_synth(self, man_img, man_type):
        _, sub, el = dir_split(self.bg_path)
        title, _ = el.rsplit('.', 1)

        splits = dir_split(self.sign_path)
        if len(splits) == 5:
            _, _, sign_dir, dmg_dir, element = splits
            head, tail = element.rsplit('.', 1)
        elif len(splits) == 4:
            _, _, sign_dir, dmg_dir = splits
            head, tail = dmg_dir.rsplit('.', 1)
        else:
            raise ValueError(f"Not enough values to unpack (expected 5 or 4, got {len(splits)})")

        save_dir = os.path.join(self.out_dir, sub, "BG_" + title, "SIGN_" + sign_dir, dmg_dir)
        os.makedirs(save_dir, exist_ok=True)  # Create relevant directories dynamically
        save_path = os.path.join(save_dir, head + "_" + man_type + "." + tail)
        if 'numpy' in type(man_img).__module__:
            cv2.imwrite(save_path, man_img)
        elif 'PIL' in type(man_img).__module__:
            man_img.save(save_path)
        man_image = self.original_synth.clone()
        man_image.set_fg_path(save_path)
        man_image.set_manipulation(man_type)
        man_image.bg_path = self.bg_path
        return man_image
    
    @abstractmethod
    def manipulation(self, fg):
        pass


class GammaMan(AbstractManipulation):
    def manipulation(self, fg):
        gammas = [0.2, 0.4, 0.67, 1.0, 1.5, 3.0, 5.0]  # Using 3.0 and not 2.5 because the latter was not noticeable
        for g in gammas:
            # Adapted from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            g_lookup = np.empty((1,256), np.uint8)
            for i in range(256):  # Create look-up table for gamma correction
                g_lookup[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)

            self.man_images.append(self.save_synth(
                cv2.LUT(fg, g_lookup),
                f"gamma_{g}"
            ))


class HistogramMan(AbstractManipulation):
    def manipulation(self, fg):
        from skimage.exposure import match_histograms
        bg = cv2.imread(self.bg_path, cv2.IMREAD_UNCHANGED)

        self.man_images.append(self.save_synth(
            match_histograms(fg, bg, multichannel=False),
            f"hist_matched"
        ))


class ExposureMan(AbstractManipulation):
    """Refactored original exposure manipulation implementation by Stergiou."""
    def manipulation(self, fg):
        bg_exposure = find_image_exposure(self.bg_path)

        if self.original_synth.bg_path is not None and self.original_synth.bg_path != self.bg_path:
            return

        ###   ORIGINAL EXPOSURE IMPLEMENTATION   ###
        brightness_avrg           = 1.0  # TODO: ???
        brightness_rms            = 1.0
        brightness_avrg_perceived = 1.0
        brightness_rms_perceived  = 1.0
        brightness_avrg2          = 1.0
        brightness_rms2           = 1.0

        # abs(desired_brightness - actual_brightness) / abs(brightness_float_value) = ratio
        avrg_ratio           = 11.0159464507  # TODO: ??? How were these calculated?
        rms_ratio            = 8.30320014372
        percieved_avrg_ratio = 3.85546373056
        percieved_rms_ratio  = 35.6344530649
        avrg2_ratio          = 1.20354549572
        rms2_ratio           = 40.1209106864

        img_grey = Image.open(self.sign_path).convert('LA')
        img_rgba = Image.open(self.sign_path).convert('RGBA')

        stat = ImageStat.Stat(img_grey)
        avrg = stat.mean[0]
        rms  = stat.rms[0]

        
        ### IMAGE MANIPULATION MAIN CODE STARTS ###
        # MINIMISE MARGIN BASED ON AVERAGE FOR TWO CHANNEL BRIGHTNESS VARIATION
        margin = abs(avrg - float(bg_exposure[1]))
        brightness_avrg = margin / avrg_ratio
        
        enhancer = ImageEnhance.Brightness(img_rgba)
        avrg_bright_grey = enhancer.enhance(brightness_avrg)
        stat = ImageStat.Stat(avrg_bright_grey)
        avrg = stat.mean[0]  # TODO: How is it minimizing if there is no iteration?

        
        # MINIMISE MARGIN BASED ON ROOT MEAN SQUARE FOR TWO CHANNEL BRIGHTNESS VARIATION
        margin = abs(rms - float(bg_exposure[2]))
        brightness_rms = margin / rms_ratio 
        
        enhancer = ImageEnhance.Brightness(img_rgba)
        rms_bright_grey = enhancer.enhance(brightness_rms)
        stat = ImageStat.Stat(rms_bright_grey)
        rms = stat.rms[0]

        
        # MINIMISE MARGIN BASED ON AVERAGE FOR RGBA ("PERCEIVED BRIGHTNESS")
        # REFERENCE FOR ALGORITHM USED: http://alienryderflex.com/hsp.html
        stat2 = ImageStat.Stat(img_rgba)
        r, g, b, a = stat2.mean
        avrg_perceived = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

        margin = abs(avrg_perceived - float(bg_exposure[3]))
        brightness_avrg_perceived = margin / percieved_avrg_ratio
        
        enhancer = ImageEnhance.Brightness(img_rgba)
        avrg_bright_perceived = enhancer.enhance(brightness_avrg_perceived)
        stat2 = ImageStat.Stat(avrg_bright_perceived)
        r, g ,b, _ = stat2.mean
        avrg_perceived = math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))


        # MINIMISE MARGIN BASED ON RMS FOR RGBA ("PERCEIVED BRIGHTNESS")
        # REFERENCE FOR ALGORITHM USED: http://alienryderflex.com/hsp.html
        r, g, b, a = stat2.rms
        rms_perceived = math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))

        margin = abs(rms_perceived - float(bg_exposure[4]))
        brightness_rms_perceived = margin / percieved_rms_ratio 

        enhancer = ImageEnhance.Brightness(img_rgba)
        rms_bright_perceived = enhancer.enhance(brightness_rms_perceived)
        stat2 = ImageStat.Stat(rms_bright_perceived)
        r, g, b, _ = stat2.rms
        rms_perceived = math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))

        stat3 = ImageStat.Stat(img_rgba)
        avrg2 = stat3.mean[0]
        rms2 = stat3.rms[0]


        #FUSION OF THE TWO AVERAGING METHODS
        margin = abs(avrg2-float(bg_exposure[1]))
        brightness_avrg2 = margin / avrg2_ratio

        enhancer = ImageEnhance.Brightness(img_rgba)
        avrg_bright2 = enhancer.enhance(brightness_avrg2)
        stat3 = ImageStat.Stat(avrg_bright2)
        avrg2 = stat3.mean[0]


        #FUSION OF THE TWO RMS METHODS
        margin = abs(rms2-float(bg_exposure[2]))
        brightness_rms2 = margin / rms2_ratio

        enhancer = ImageEnhance.Brightness(img_rgba)
        rms_bright2 = enhancer.enhance(brightness_rms2)
        stat3 = ImageStat.Stat(rms_bright2)
        rms2 = stat3.rms[0]


        self.man_images.append(self.save_synth(avrg_bright_grey,      "average_grey"))
        self.man_images.append(self.save_synth(rms_bright_grey,       "rms_grey"))
        self.man_images.append(self.save_synth(avrg_bright_perceived, "average_perceived"))
        self.man_images.append(self.save_synth(rms_bright_perceived,  "rms_perceived"))
        self.man_images.append(self.save_synth(avrg_bright2,          "average2"))
        self.man_images.append(self.save_synth(rms_bright2,           "rms2"))


class GammaExposureFastMan(AbstractManipulation):
    """A faster alternative to GammaExposureAccurateMan() that uses a single calculated gamma value rather than
    iterating through a predefined set of gamma values. The calculation provides a rough estimation of the ideal gamma.

    Due to the gamma being a continuous predicted value, as opposed to a selection from a set, the final manipulation
    results are far more visually consistent across transformations as compared to GammaExposureAccurateMan().

    TEST (GTSDB Wikipedia templates, no_damage only, 6 transforms, 600 GTSDB train backgrounds):
      Sped up find_gamma() speed (pre-sort): 3-4 ms
      ~10x faster than GammaExposureAccurateMan() if using just 1 brightness metric ('average_perceived', 'rms_grey', etc.).
      ~2-3x faster than GammaExposureAccurateMan() if using 4 brightness metrics.

      ~2.5 - 3x wider average brightness margin between background and manipulated foreground (i.e. less accurate)
      ~32.5 vs. ~11.5 in domain [0,255]
    """

    def manipulation(self, fg):
        bg_exposure = find_image_exposure(self.bg_path)
        fg_exposure = find_image_exposure(self.sign_path)

        if self.original_synth.bg_path is not None and self.original_synth.bg_path != self.bg_path:
            return

        ## For debug visualisations
        # avrg = ImageStat.Stat(Image.open(self.sign_path).convert('RGBA')).mean[0]

        # margin = abs(avrg - float(bg_exposure['avg']))
        # print(f"original: {margin}")
        ##
        
        import time  ##

        # Pre-define gamma lookup tables to save processing time when > 1 brightness metrics are used
        start = time.time()
        gammas = [0.1, 0.2, 0.4, 0.67, 1.0, 1.5, 3.0, 5.0, 10.0]  # Using 3.0 and not 2.5 because the latter was not noticeable
        g_lookups = []
        for g in gammas:
            # Adapted from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            g_lookup = np.empty((1,256), np.uint8)
            for i in range(256):  # Create look-up table for gamma correction
                g_lookup[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)
            g_lookups.append(g_lookup)
        pretime = time.time() - start

        def find_gamma(bg_brightness: float, fg_brightness: float):
            """Iterate through pre-selected gamma values to minimise marginal brightness difference to background."""
            man_imgs = []
            
            # Calculated estimated gamma required to match brightness with background
            # Inspired by: Babakhani, P., & Zarei, P. (2015). Automatic gamma correction based on average of brightness.
            gamma = math.log10(bg_brightness / 255) / math.log10(fg_brightness / 255)

            # Adapted from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            g_lookup = np.empty((1,256), np.uint8)
            for i in range(256):  # Create look-up table for gamma correction
                g_lookup[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

            man_img = cv2.LUT(fg, g_lookup)
            man_img_convert = cv2.cvtColor(man_img, cv2.COLOR_BGRA2RGBA)  # Convert from OpenCV BGR to PIL RGB
            man_img_pil     = Image.fromarray(man_img_convert)

            ## For calculating accuracy (average brightness margin)
            # global cum_calculated_margin, cum_iterated_margin, count
            # count += 1
            # calc_found = False
            # it_found = False
            # for ii in range(len(man_imgs)):  # Go in ascending order of margin
            #     if (not man_imgs[ii]['gamma'] in gammas):  # Calculated
            #         cum_calculated_margin += man_imgs[ii]['margin']
            #         calc_found = True
            #     else: # Iterated
            #         if it_found == False:
            #             cum_iterated_margin += man_imgs[ii]['margin']
            #             it_found = True
            #     if calc_found and it_found:
            #         break
            # print(f"Cumulative mean calculated margin: {cum_calculated_margin / count:.2f}")
            # print(f"Cumulative mean iterated margin: {cum_iterated_margin / count:.2f}\n")
            ##
            
            return man_img_pil

        # For 'perceived' RGB brightness comparisons, see: Stergiou et al. section 3.2 https://tinyurl.com/ydxzv9nx)
        avrg_bright_perceived = find_gamma(float(bg_exposure['avg_perceived']), float(fg_exposure['avg_perceived']))
        # rms_bright_perceived  = find_gamma(float(bg_exposure['rms_perceived']), float(fg_exposure['rms_perceived']))
        # avrg_bright_grey      = find_gamma(float(bg_exposure['avg_grey']), float(fg_exposure['avg_grey']))
        # rms_bright_grey       = find_gamma(float(bg_exposure['rms_grey']), float(fg_exposure['rms_grey']))

        self.man_images.append(self.save_synth(avrg_bright_perceived, "average_perceived"))
        # self.man_images.append(self.save_synth(rms_bright_perceived,  "rms_perceived"))
        # self.man_images.append(self.save_synth(avrg_bright_grey,      "average_grey"))
        # self.man_images.append(self.save_synth(rms_bright_grey,       "rms_grey"))


class GammaExposureAccurateMan(AbstractManipulation):
    """See GammaExposureFastMan() for speed/accuracy tradeoff description."""

    def manipulation(self, fg):
        bg_exposure = find_image_exposure(self.bg_path)

        if self.original_synth.bg_path is not None and self.original_synth.bg_path != self.bg_path:
            return

        ## For debug visualisations
        # avrg = ImageStat.Stat(Image.open(self.sign_path).convert('RGBA')).mean[0]

        # margin = abs(avrg - float(bg_exposure['avg']))
        # print(f"original: {margin}")
        ##
        
        # Pre-define gamma lookup tables to save processing time when > 1 brightness metrics are used
        gammas = [0.1, 0.2, 0.4, 0.67, 1.0, 1.5, 3.0, 5.0, 10.0]  # Using 3.0 and not 2.5 because the latter was not noticeable
        g_lookups = []
        for g in gammas:
            # Adapted from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            g_lookup = np.empty((1,256), np.uint8)
            for i in range(256):  # Create look-up table for gamma correction
                g_lookup[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)
            g_lookups.append(g_lookup)

        def find_gamma(bg_exposure, brightness_metric: str):
            """Iterate through pre-selected gamma values to minimise marginal brightness difference to background."""
            man_imgs = []
            for g_lookup, gamma in zip(g_lookups, gammas):
                man_img = cv2.LUT(fg, g_lookup)
                man_img_convert = cv2.cvtColor(man_img, cv2.COLOR_BGRA2RGBA)  # Convert from OpenCV BGR to PIL RGB
                man_img_pil     = Image.fromarray(man_img_convert)
                
                # BUG: mean[0] is only the mean for R channel, see above
                # man_brightness = ImageStat.Stat(man_img_pil).mean[0]
                man_brightness = image_exposure(man_img_pil.convert('LA'), man_img_pil)[brightness_metric]
                new_margin     = abs(man_brightness - bg_exposure[brightness_metric])

                man_imgs.append({
                    'gamma' : gamma,
                    'img'   : man_img_pil,
                    'margin': new_margin
                })

            sorter = lambda x : x['margin']  # Sort gamma corrections by ascending margin
            man_imgs.sort(reverse=False, key=sorter)

            ## For debug visualisations
            # for thing in man_imgs:
            #     print(f"{thing['gamma']}: {thing['margin']}")
            # print(f"best gamma: {man_imgs[0]['gamma']}\n")
            # print()
            ##
            
            return man_imgs[0]['img']

        # For 'perceived' RGB brightness comparisons, see: Stergiou et al. section 3.2 https://tinyurl.com/ydxzv9nx)
        avrg_bright_perceived = find_gamma(bg_exposure, 'avg_perceived')
        # rms_bright_perceived  = find_gamma(bg_exposure, 'rms_perceived')
        # avrg_bright_grey      = find_gamma(bg_exposure, 'avg_grey')
        # rms_bright_grey       = find_gamma(bg_exposure, 'rms_grey')

        self.man_images.append(self.save_synth(avrg_bright_perceived, "average_perceived"))
        # self.man_images.append(self.save_synth(rms_bright_perceived,  "rms_perceived"))
        # self.man_images.append(self.save_synth(avrg_bright_grey,      "average_grey"))
        # self.man_images.append(self.save_synth(rms_bright_grey,       "rms_grey"))


def avrg_pixel_rgb(image, chanels):
    stat = ImageStat.Stat(image)
    if (chanels == 4):
        r,g,b,a = stat.rms
    else:
        r,g,b = stat.rms
    
    return [r,g,b]

def find_bw_images(directory):
    images = []
    for sign in load_paths(directory):
        for damage in load_paths(sign):
            img = Image.open(damage).convert('RGBA')
            rgb = avrg_pixel_rgb(img, 4)
            rg = abs(rgb[0] - rgb[1])
            rb = abs(rgb[0] - rgb[2])
            gb = abs(rgb[1] - rgb[2])
            
            temp = dir_split(damage)
            list = temp[-1].split('.')
            if len(list) == 2:
                head = list[0]
            else:
                head = list[0] + '.' + list[1]
                    
            if (rg <= 1 and rb <= 1 and gb <= 1):
                images.append(head)
    return images

# TODO(Allen): Vectorize to improve efficiency and readability
def find_useful_signs(manipulated_images, damaged_dir):
    """Removes bad signs, such as those which are all white or all black."""
    bw_templates = find_bw_images(damaged_dir)

    pr = 0  # Progress
    pr_total = len(manipulated_images) * 2 + len(bw_templates) * len(manipulated_images)

    temp = []
    for man in manipulated_images:
        print(f"Removing useless signs: {float(pr) / float(pr_total):06.2%}", end='\r')
        temp.append(man.fg_path)
        pr += 1

    exposures = find_image_exposures(temp, "manipulated sign")

    # Compile list of black and white signs to be deleted under differnet metrics below
    is_bw = []
    for bw_path in bw_templates:  # O(mn): m is small, so effectively O(n)
        for exposure in exposures:
            if bw_path in exposure[0]:  # Check for a bw template in each path
                is_bw.append(exposure[0])
            pr += 1

    for manipulated in reversed(manipulated_images):
        print(f"Removing useless signs: {float(pr) / float(pr_total):06.2%}", end='\r')

        image_path = manipulated.fg_path
        
        if not os.path.exists(image_path): continue

        # Find brightness
        img = Image.open(image_path).convert('RGBA')
        rgb = avrg_pixel_rgb(img, 4)
        rg = abs(rgb[0] - rgb[1])
        rb = abs(rgb[0] - rgb[2])
        gb = abs(rgb[1] - rgb[2])

        #BUG: Shit detection for white and grey sign 49, leaving plain whites while deleting good ones    

        if rg <= 16 and rb <= 16 and gb <= 16:
            if not manipulated.fg_path in is_bw:
                os.remove(image_path)
                #del manipulated  # TODO: Deletes temporarily disabled in place of temp outer solution
            # Threshold values for black and white images
            elif rgb[0] < 70 and rgb[1] < 70 and rgb[2] < 70:
                os.remove(image_path)
                #del manipulated
            elif rgb[0] > 155 and rgb[1] > 155 and rgb[2] > 155:
                os.remove(image_path)
                #del manipulated
        elif not manipulated.fg_path in is_bw:
            # Delete light blue images
            if rgb[2] > rgb[0] and rgb[2] >= rgb[1]:
                if gb <= 10:
                    os.remove(image_path)
                    #del manipulated
        pr += 1
    print(f"Removing useless signs: 100.0%\r\n")



################################
###  UNUSED NOISE FUNCTIONS  ###
################################
def insert_poisson_noise (image):
    vals = len(np.unique(image))
    vals = 2.05 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def insert_Gaussian_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def insert_speckle_noise(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy

def random_noise_method(image):
    """
    i = random.randint(1, 3)
    if (i == 1):
        return insert_poisson_noise(image)
    elif (i==2):
        return insert_Gaussian_noise(image)
    else:
        return insert_speckle_noise(image)
    """
    image.setflags(write=1)
    # Add noise in every pixel with random probability 0.4
    for im in image:
        px = 0
        for pixel in im:
            apply_noise = random.randint(0,100)
            if apply_noise > 40:
                # RGB values
                r = pixel[0]
                g = pixel[1]
                b = pixel[2]
                a = pixel[3]
                # Find current relative lumination for brighness
                # Based on: https://en.wikipedia.org/wiki/Relative_luminance
                relative_lumination = 0.2126*r + 0.7152*g + 0.0722*b
                # Find differences between RGB values     
                rg = False
                r_to_g = float(r) / float(g)
                if (r_to_g >= 1):
                    rg = True
                
                rb = False
                r_to_b = float(r) / float(b)
                if (r_to_b >= 1):
                    rb = True
                
                gb = False
                g_to_b = float(g) / float(b)
                if (g_to_b >= 1):
                    gb = True
                
                equal = False
                if (r == g == b):
                    equal = True

                # In order to determine the margin in which the new brighness
                # should be within, the upper and lower limits need to be found
                # The Relative luminance in colorimetric spaces has normalised
                # values between 0 and 255
                upper_limit = 255
                lower_limit = 0
                if (relative_lumination + 40 < 255):
                    upper_limit = relative_lumination + 40
                if (relative_lumination - 40 > 0):
                    lower_limit = relative_lumination - 40

                # Compute new brightness value
                new_lumination = random.randint(int(lower_limit), int(upper_limit))

                # Find the three possible solutions that satisfy
                # -> The new lumination chosen based on the Relative luminance equation
                # -> The precentages computed between every rgb value

                solutions = []

                for r in range(1,255):
                    for g in range(1,255):
                        for b in range(1,255):
                            rg = False
                            r_to_g = float(r) / float(g)
                            if (r_to_g >= 1):
                                rg = True
                            
                            rb = False
                            r_to_b = float(r) / float(b)
                            if (r_to_b >= 1):
                                rb = True
                            
                            gb = False
                            g_to_b = float(g) / float(b)
                            if (g_to_b >= 1):
                                gb = True
                            
                            e = False
                            if(r == g == b):
                                e = True
                            
                            if (0.2126*r + 0.7152*g + 0.0722*b == 100) and rg==rg and rb==rb and gb==GB and e==equal:
                                solutions.append([r,g,b])

                # Find the solution that precentage wise is closer to the original
                # difference between the values
                percentages = []

                for solution in solutions:
                    r = solution[0]
                    g = solution[1]
                    b = solution[2]
                    percentages.append((float(r) / float(g)) + (float(r) / float(b)) + (float(g) / float(b)))

                ii = 0
                pos = 0
                best = percentages[0]
                for p in percentages[1:]:
                    if p < best:
                        pos = ii
                    ii += 1

                # Assign new pixel values
                im[px] = [solutions[pos][0], solutions[pos][1], solutions[pos][2], A]
            px += 1
            
    return image