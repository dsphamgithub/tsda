class ConfigError(Exception):
    pass

def main():
    import os
    import sys
    import ntpath
    import shutil
    import argparse
    from datetime import datetime
    import random
    import json
    from collections import defaultdict
    from pathlib import Path
    import glob

    from PIL import ImageFile
    import numpy as np
    import cv2

    from synth_image import SynthImage
    from damage import damage_image
    from utils import load_paths, load_files, scale_image, delete_background, to_png
    import manipulate
    from manipulate import RotationTransform, FixedAffineTransform
    from manipulate import ExposureMan, GammaMan, GammaExposureAccurateMan, HistogramMan, GammaExposureFastMan
    import generate
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser(description='Create a synthetically generated traffic sign dataset.')
    # SGTSD stands for "Synthetically Generated Traffic Sign Dataset"
    parser.add_argument('--output_dir', type=str, help='Complete path to directory where dataset will be created',
                        default=os.path.join(current_dir, 'SGTS_Dataset'))
    
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.chdir(current_dir)

    # Open and validate config file
    import yaml
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # TODO: Refactor into config validation function/module
    # Input validation of config file
    valid_final = ['process', 'damage', 'transform', 'manipulate', 'dataset']
    tform_methods = {
        '3d_rotation': RotationTransform(),
        'fixed_affine': FixedAffineTransform()
    }
    man_methods = {
        'exposure': ExposureMan(),
        'gamma': GammaMan(),
        'gamma_exposure': GammaExposureAccurateMan(),
        'gamma_exposure_fast': GammaExposureFastMan(),
        'histogram': HistogramMan()
    }
    valid_dmg = ['no_damage', 'quadrant', 'big_hole', 'graffiti']
    valid_ann = ['retinanet', 'coco']
    valid_dmg_methods = ['ssim', 'pixel_wise']
    if config['sign_width'] <= 0:
        raise ConfigError("Config error: 'sign_width' must be > 0.\n")
    if not config['final_op'] in valid_final:
        raise ConfigError(f"Config error: '{config['final_op']}' is an invalid final_op value.\n")
    if config['num_transform'] < 0 or config['num_transform'] > 11:
        raise ConfigError("Config error: must have 0 <= 'num_transform' <= 15.\n")
    if not config['tform_method'] in tform_methods:
        raise ConfigError(f"Config error: '{config['tform_method']}' is an invalid transformation type.\n")
    if not config['man_method'] in man_methods:
        raise ConfigError(f"Config error: '{config['man_method']}' is an invalid manipulation type.\n")
    for dmg in config['num_damages']:
        if not dmg in valid_dmg and dmg != 'online':
            raise ConfigError(f"Config error: '{dmg}' is an invalid damage type.\n")
    if not config['annotations']['type'] in valid_ann:
        raise ConfigError(f"Config error: '{config['annotations']['type']}' is an invalid annotation type.\n")
    if not config['damage_measure_method'] in valid_dmg_methods:
        raise ConfigError(f"Config error: '{config['damage_measure_method']}' is an invalid damage measure.\n")

    r_params = config['transforms']
    if (r_params['tilt_SD'] < 0 or r_params['tilt_SD'] > 90 or
            r_params['tilt_range'] < 0 or r_params['tilt_range'] > 90):
        raise ConfigError("Config error: must have 0 <= 'tilt_SD' <= 90 and 0 <= 'tilt_range' <= 90.\n")
    if (r_params['Z_SD'] < 0 or r_params['Z_SD'] > 180 or
            r_params['Z_range'] < 0 or r_params['Z_range'] > 180):
        raise ConfigError("Config error: must have 0 <= 'Z_SD' <= 180 and 0 <= 'Z_range' <= 180.\n")
    if r_params['online'] is True and config['tform_method'] != '3d_rotation':
        raise ConfigError(f"Config error: online transformations only work "
                          f"with the 3d_rotation transformation method.\n")
    if r_params['prob'] < 0 or r_params['prob'] > 1:
        raise ConfigError("Config error: must have 0 <= 'prob' <= 1.\n")

    g_params = config['graffiti']
    for g_param in g_params:
        if (g_params[g_param] <= 0.0 or g_params[g_param] > 1.0) and g_param != 'solid':
            raise ConfigError(f"Config error: must have 0.0 < 'graffiti:{g_param}' <= 1.0.\n")
    if g_params['initial'] > g_params['final']:
        raise ConfigError("Config error: 'graffiti:initial' must be <= 'graffiti:final'.\n")
    
    if not config['reuse_data']['damage'] and config['reuse_data']['manipulate']:
        raise ConfigError("Config error: 'reuse_data:damage' must be true if 'reuse_data:manipulate' is true.\n")

    print("Generating dataset using the 'config.yaml' configuration.\n")

    # Directory names excluded from config.yaml to make use of .gitignore simpler
    base_dir        = "Sign_Templates"
    input_dir       = os.path.join(base_dir, "1_Input")
    processed_dir   = os.path.join(base_dir, "2_Processed")
    damaged_dir     = os.path.join(base_dir, "3_Damaged")
    transformed_dir = os.path.join(base_dir, "4_Transformed")
    manipulated_dir = os.path.join(base_dir, "5_Manipulated")
    bg_dir    = "Backgrounds"
    final_dir = args.output_dir

    if config['reuse_data']['damage'] and not os.path.exists(os.path.join(damaged_dir, 'damaged_data.npy')):
        raise FileNotFoundError('Config error: damaged_data.npy does not exist. Cannot re-use data\n')
    if config['reuse_data']['manipulate'] and not os.path.exists(os.path.join(manipulated_dir, 'manipulated_data.npy')):
        raise FileNotFoundError('Config error: manipulated_data.npy does not exist. Cannot re-use data\n')

    # Create the output directories if they don't exist already
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    input_error = (f"Error: {input_dir} must be populated with template signs to proceed. A link to example "
                    "data can be found in the README.\n")
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        raise ValueError(input_error)
    if len(load_paths(input_dir)) == 0:
        raise ValueError(input_error)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.mkdir(processed_dir)

    # Seed the random number generator
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    


    ##################################
    ###  BACKGROUND PREPROCESSING  ###
    ##################################
    background_paths = glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.png", recursive=True) + \
        glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.jpg", recursive=True)
    background_images = []



    #############################
    ###  IMAGE PREPROCESSING  ###
    #############################
    # Rescale images and make white backgrounds transparent
    paths = load_files(input_dir)
    for path in paths:
        _, filename = ntpath.split(path)
        name, extension = filename.rsplit('.', 1)
        
        img = scale_image(path, config['sign_width'])  # Rescale the image
        save_path = os.path.join(processed_dir, name) + ".png"
        img.save(save_path)

        if not img.mode[-1] == 'A':
            delete_background(save_path, save_path)  # Overwrite the newly rescaled image

    if config['final_op'] == 'process':
        return



    #########################
    ###  APPLYING DAMAGE  ###
    #########################
    reusable = config['reuse_data']['damage']
    data_file_path = os.path.join(damaged_dir, "damaged_data.npy")
    if not reusable:
        if os.path.exists(damaged_dir):  # Remove any old output
            shutil.rmtree(damaged_dir)
        os.mkdir(damaged_dir)
        damaged_data = []

        ii = 0
        processed = load_files(processed_dir)
        for image_path in processed:
            if config['num_damages']['online'] is False:
                print(f"Damaging signs: {float(ii) / float(len(processed)):06.2%}", end='\r')
            synth_img = SynthImage(image_path, int(ntpath.split(image_path)[1].rsplit('.', 1)[0]))
            damaged_data.append(damage_image(synth_img, damaged_dir, config, background_images))
            ii += 1
        if config['num_damages']['online'] is False:
            print(f"Damaging signs: 100.0%\r\n")
        else:
            print(f"Damaging signs on-the-fly in file generation step.\n")
        damaged_data = [cell for row in damaged_data for cell in row]  # Flatten the list
        if config['num_damages']['online'] is False:  # Saved data is useless in the online case
            np.save(data_file_path, damaged_data, allow_pickle=True)
    elif os.path.exists(data_file_path):
        damaged_data = np.load(data_file_path, allow_pickle=True)
        print("Reusing pre-existing damaged signs.\n")
    else:
        raise FileNotFoundError(f"Error: Damaged data file does not exist - cannot reuse.\n")

    if config['final_op'] == 'damage':
        return



    ##################################
    ###  APPLYING TRANSFORMATIONS  ###
    ##################################
    if os.path.exists(transformed_dir):
        shutil.rmtree(transformed_dir)

    t_method = config['tform_method']
    if config['transforms']['online'] is False:
        os.mkdir(transformed_dir)

        print("Transforming signs...", end='\r')
        transformed_data = []
        for damaged in damaged_data:
            save_dir = os.path.join(transformed_dir, str(damaged.class_num))
            transformed_data.append(tform_methods[t_method].transform(damaged, save_dir, config['num_transform']))
            del damaged  # Clear memory that's no longer be needed as we go
        del damaged_data
        transformed_data = [cell for row in transformed_data for cell in row]  # Flatten the list
        print('\n')
    else:
        transformed_data = damaged_data
        print("Transforming signs on-the-fly in file generation step.\n")

    if config['final_op'] == 'transform':
        return



    ####################################
    ###  MANIPULATING EXPOSURE/FADE  ###
    ####################################
    reusable = config['reuse_data']['manipulate']
    data_file_path = os.path.join(manipulated_dir, "manipulated_data.npy")
    if not reusable:
        if os.path.exists(manipulated_dir):
            shutil.rmtree(manipulated_dir)
        os.mkdir(manipulated_dir)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        for bg_folders in load_paths(bg_dir):
            to_png(bg_folders)
        print('\n', end='')
            
        background_paths = glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.png", recursive=True)
        
        m_method = config['man_method']
        manipulated_data = man_methods[m_method].manipulate(transformed_data, background_paths, manipulated_dir)
        if m_method == 'exposure':
            manipulate.find_useful_signs(manipulated_data, damaged_dir)

        # Delete SynthImage objects for any signs that were removed
        manipulated_data[:] = [x for x in manipulated_data if os.path.exists(x.fg_path)]
        np.save(data_file_path, manipulated_data, allow_pickle=True)
    else:
        manipulated_data = np.load(data_file_path, allow_pickle=True)
        print("Reusing pre-existing manipulated signs.\n")
    
    # Prune dataset by randomly sampling from manipulated images
    if config['prune_dataset']['prune']:
        max_images = config['prune_dataset']['max_images']
        images_dict = defaultdict(list)
        for img in manipulated_data:
            images_dict[os.path.dirname(img.fg_path)].append(img)
        # Sample manipulated-transformed images for each background/class/damage
        images_dict = {k:random.sample(v, max_images) for k,v in images_dict.items() if len(v) >= max_images}
        manipulated_data = [img for images_list in images_dict.values() for img in images_list]
        assert len(manipulated_data) != 0, "Set of manipulated images is empty after pruning"

    if config['final_op'] == 'manipulate':
        return



    ###############################
    ###  GENERATING FINAL DATA  ###
    ###############################
    # Create timestamped directory instead of overwriting
    dir_head, dir_tail = ntpath.split(final_dir)
    final_dir = os.path.join(dir_head, f"{dir_tail}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    images_dir = os.path.join(final_dir, "Images")
    labels_format = config['annotations']['type']
    damage_labelling = config['annotations']['damage_labelling'] == True
    
    # Initialise annotation files according to config parameters
    if labels_format == 'retinanet':
        labels_path = os.path.join(final_dir, "labels.txt")
    elif labels_format == 'coco':
        labels_path = os.path.join(final_dir, "_annotations.coco.json")
        classes = [int(Path(p).stem) for p in glob.glob(f'{processed_dir}{os.path.sep}*.png')]
        labels_dict = {'categories': [], 'images': [], 'annotations': []}
        labels_dict["categories"] += [{"id": 0, "name": "signs", "supercategory": "none"}]
        labels_dict["categories"] += [{'id:': c, 'name': str(c), 'supercategory': 'signs'} for c in sorted(classes)]
    about_path = os.path.join(final_dir, "generated_images_about.txt")

    # Clean and recreate the parent images directory
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(images_dir)
    
    total_gen = len(manipulated_data)
    print(f"Files to be generated: {total_gen}")

    ii = 0
    labels_file = open(labels_path, "w")
    
    for synth_image in manipulated_data:
        print(f"Generating files: {float(ii) / float(total_gen):06.2%}", end='\r')
        
        c_num = synth_image.class_num
        d_type = synth_image.damage_type
        class_dir = os.path.join(images_dir, f"{c_num}", f"{c_num}_{d_type}")
        # Create the directory for each class+damage combination if it doesn't already exist
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        fg_path = os.path.join(class_dir, f"{c_num}_{d_type}_{ii}")
        final_fg_path = fg_path + ".jpg"

        d_online = config['num_damages']['online']
        t_online = config['transforms']['online']
        if d_online is True:
            synth_image = damage_image(synth_image, damaged_dir, config, background_images, single_image=True)
        if t_online is True and random.random() <= config['transforms']['prob']:
            synth_image = tform_methods[t_method].transform(synth_image, None, 1)[0]

        image = generate.new_data(synth_image, (d_online or t_online))
        if labels_format == 'retinanet':
            synth_image.write_label_retinanet(labels_file, damage_labelling)
        elif labels_format == 'coco':
            synth_image.write_label_coco(labels_dict, ii, 
                                         os.path.relpath(final_fg_path, final_dir), image.shape, damage_labelling)
        cv2.imwrite(final_fg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        ii += 1
    print(f"Generating files: 100.0%\r\n")
    
    if labels_format == "coco":
        labels_dict['images'] = sorted(labels_dict['images'], key=lambda x: x['id'])
        labels_dict['annotations'] = sorted(labels_dict['annotations'], key=lambda x: x['id'])
        json.dump(labels_dict, labels_file, indent=4)
    labels_file.close()

    string = "-------------------------------------\nBREAKDOWN OF FILES GENERATED BY " \
             "CLASS\n-------------------------------------\n"
    for class_dir in load_paths(images_dir):
        c_total = 0
        for damage_dir in load_paths(class_dir):
            c_total += len(load_paths(damage_dir))
        _, c_num = ntpath.split(class_dir)
        c_str = f"Generated {c_total} examples for sign class {c_num}\n"
        string += c_str
    string += f"\nTOTAL: {total_gen}\n"
    string += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    string += "-------------------------------------"

    with open(about_path, "w") as text_file:
        text_file.write(string)
    
    # Save config file with dataset
    with open(os.path.join(final_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(e)
