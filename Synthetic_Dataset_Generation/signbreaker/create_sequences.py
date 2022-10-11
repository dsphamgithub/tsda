"""Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.

An interactive window opens where the user selects two anchor points for each background. They represent the position
and size of the foreground image at the start and end of the sequence, with intermediary frames add between according to
the user specified sequence length.
"""

# FIXME: fg_dir must be preprocessed, i.e. sourced from 2_Processed_Images
# TODO: Calculate pole length based on size of anchor point and static variable that represents the standard pole length
#       for the region in question. In GUI draw a line from centre of AP box downwards to show where it will be actually
#       drawn. Alternatively, have option to switch to mode where you select where to place bottom of sign pole and go
#       backwards to draw sign AP box based on value in size slider

OUT_DIR         = "SGTS_Sequences"
LABELS_FILE     = "labels.txt"
MIN_ANCHOR_SIZE = 10
MAX_ANCHOR_SIZE = 200
import argparse
import cv2
from datetime import datetime
import numpy as np
import ntpath
import os
from utils import load_paths, load_files, resize, overlay


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=dir_path, help="path to background image directory")
    parser.add_argument("fg_dir", type=dir_path, help="path to foreground/sign image directory")
    parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
    return parser.parse_args()

def dir_path(path):
    """Validate an argument type as being a directory path."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"""{path} is not a valid path""")


def draw_square(event, x, y, flags, param):
    """OpenCV mouse callback function for drawing square markers."""
    img                = param[0]
    current_anchor_set = param[1]
    window_name        = param[2]

    # Draw square to indicate selection to user
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_anchor_set) <= 1:
            size = cv2.getTrackbarPos("Size", window_name)
            cv2.drawMarker(img, (x,y), (0,255,0), cv2.MARKER_SQUARE, size, 2)
            current_anchor_set.append((x,y,size))  # Append anchor coordinates to anchors list
        else:
            print("Error: you have already selected both anchors. Press [r] to reset the anchors for this image.")

def nothing(x):
    pass

def select_anchor_points(bg_paths, num_bg):
    """Interactive GUI that lets the user select two anchor points for each background by marking squares with variable
    user controlled sizes.
    
    Arguments:
    bg_paths -- list of paths to each background image
    num_bg   -- integer number of background images that need to have anchor points selected
    """
    anchors = []
    count = 1
    for bg in bg_paths:
        print(f"Selecting anchor points for background image {count}/{num_bg}...")
        window_name = "Select anchor points for " + bg

        img = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
        current_anchor_set = []  # Set of 2 tuples, indicating far anchor and near anchor respectively
        selecting = True
        size = (2 * MIN_ANCHOR_SIZE + MAX_ANCHOR_SIZE) // 3  # Using weighted average to set small default size

        # Set up interactive window for user anchor point selection
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_square, param=[img, current_anchor_set, window_name])
        cv2.createTrackbar("Size", window_name, MIN_ANCHOR_SIZE, MAX_ANCHOR_SIZE, nothing)

        while(selecting):
            cv2.imshow(window_name, img)
            k = cv2.waitKey(1) & 0xFF

            # Confirm selection and move onto next background image
            if k == 32 or k == 13:  # Press SPACEBAR key or ENTER key
                if len(current_anchor_set) == 2:
                    anchors.append(current_anchor_set)
                    count += 1
                    selecting = False
                else:
                    print("Error: you must first select 2 anchor points.")

            # Reset the image as a means of undoing changes
            elif k == ord('r') or k == ord('R'):  # Press 'r' key or 'R' key
                current_anchor_set = []
                img = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
                cv2.setMouseCallback(window_name, draw_square, param=[img, current_anchor_set, window_name])

            # Quit early to 
            elif k == 27:  # ESC key
                raise InterruptedError("quitting early")
        cv2.destroyAllWindows()
    return anchors


def main():
    args = parse_arguments()
    labels_path = os.path.join(OUT_DIR, LABELS_FILE)  #TODO: labels_path unused
    sequence_len = args.num_frames

    # Loading argument-specified directories
    bg_paths = load_paths(args.bg_dir)
    fg_paths = load_paths(args.fg_dir)
    num_bg = len(bg_paths)
    num_fg = len(fg_paths)
    print(f"Found {num_bg} background images.")
    print(f"Found {num_fg} foreground images.\n")
    if num_bg == 0 or num_fg == 0:
        print("Error: each input directory must have at least 1 image")
        return

    # Directory structure setup
    if (not os.path.exists(OUT_DIR)):
        out_dir = OUT_DIR
    else:
        timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        out_dir = OUT_DIR + timestamp
    os.mkdir(out_dir)

    # Get user to select anchor points in each background image
    try:
        anchors = select_anchor_points(bg_paths, num_bg)
    except InterruptedError as e:
        print("Error:", str(e))
        return
    print("Anchors (x, y, size):\n" + str(anchors))

    # Generate sequences by overlaying foregrounds over backgrounds according to anchor point data
    seq_ratio = 1 / (sequence_len - 1)
    count = 0
    for bg in bg_paths:
        bg_img = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
        _, bg_filename = ntpath.split(bg)
        bg_name, _ = bg_filename.split('.')  # Remove file extension
        
        for fg in fg_paths:
            fg_img = cv2.imread(fg, cv2.IMREAD_UNCHANGED)
            _, fg_filename = ntpath.split(fg)
            fg_name, _ = fg_filename.split('.')  # Remove file extension
            
            start_size = anchors[count][0][2]
            size_diff  = anchors[count][1][2] - start_size  

            start_x = anchors[count][0][0]
            start_y = anchors[count][0][1]
            x_diff = anchors[count][1][0] - start_x  
            y_diff = anchors[count][1][1] - start_y

            for frame in range(sequence_len):
                diff_ratio = frame * seq_ratio
                size = int(start_size + (diff_ratio * size_diff))
                x = int(start_x + (diff_ratio * x_diff))
                y = int(start_y + (diff_ratio * y_diff))

                fg_img_new = cv2.resize(fg_img, (size,size))
                img_new = overlay(fg_img_new, bg_img, x, y)
                
                img_new_path = os.path.join(out_dir, bg_name + "_" + fg_name + "_" + str(frame) + ".jpg")
                cv2.imwrite(img_new_path, img_new)
        count += 1


def show_anchors(bg_path, fg_path, num_frames):
    """Test function to overlay the interpolated foregrounds on the same image in the UI to view sequence behavior for
    that single image."""
    bg_img = cv2.imread(bg_path)
    fg_img = cv2.imread(fg_path)
    anchors = select_anchor_points([bg_path], 1)
        
    start_size = anchors[0][0][2]
    size_diff = anchors[0][1][2] - start_size
    
    start_x = anchors[0][0][0]
    start_y = anchors[0][0][1]
    
    x_diff = anchors[0][1][0] - start_x  
    y_diff = anchors[0][1][1] - start_y
    
    seq_ratio = 1 / (num_frames - 1)
    for frame in range(num_frames - 1, -1, -1):
        diff_ratio = frame * seq_ratio
        size = int(start_size + (diff_ratio * size_diff))
        x = int(start_x + (diff_ratio * x_diff))
        y = int(start_y + (diff_ratio * y_diff))

        fg_img_new = cv2.resize(fg_img, (size,size))
        bg_img = overlay(fg_img_new, bg_img, x, y)
    cv2.imwrite('overlayed_sequence_manual.jpg', bg_img)
    cv2.imshow('Manually selected sequence', bg_img)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()

    # # Test code
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # bg_path = os.path.join(current_dir, 'Backgrounds/GTSDB/00049.png')
    # sign_path = os.path.join(current_dir, 'Sign_Templates/1_Input/0.jpg')
    # show_anchors(bg_path, sign_path, 8)