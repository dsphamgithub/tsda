from datetime import datetime
from pathlib import Path
import random

class SynthImage:
    def __init__(self, fg_path, class_num, fg_image=None, damage_type=None, damage_tag=None, damage_ratio=0.0,
            sector_damage=[], transform_type=None, man_type=None, bg_path=None, bounding_axes=None, fg_coords=None,
            fg_size=None):
        self.__check_class(class_num)
        self.__check_damage(damage_ratio)

        self.fg_path = fg_path
        self.fg_image = fg_image
        self.class_num = class_num

        self.damage_type = damage_type
        self.damage_tag = damage_tag
        self.damage_ratio = damage_ratio
        self.sector_damage = sector_damage

        self.transform_type = transform_type
        self.man_type = man_type
        self.bg_path = bg_path

        self.bounding_axes = bounding_axes
        self.fg_coords = fg_coords
        self.fg_size = fg_size

    def __repr__(self):
        return f"fg_path={self.fg_path}"

    def set_fg_path(self, fg_path):
        self.fg_path = fg_path

    def set_fg_image(self, fg_image):
        self.fg_image = fg_image
    
    def set_damage(self, damage_type, damage_tag, damage_ratio, sector_damage):
        self.__check_damage(damage_ratio)
        self.__check_sector_damage(sector_damage)
        self.damage_type = damage_type
        self.damage_tag = damage_tag
        self.damage_ratio = damage_ratio
        self.sector_damage = sector_damage

    def set_transformation(self, transform_type):
        self.__check_transformation(transform_type)
        self.transform_type = transform_type

    def set_manipulation(self, man_type):
        self.__check_manipulation(man_type)
        self.man_type = man_type

    def clone(self):
        return SynthImage(
            fg_path=self.fg_path,
            class_num=self.class_num,
            fg_image=self.fg_image,
            damage_type=self.damage_type,
            damage_tag=self.damage_tag,
            damage_ratio=self.damage_ratio,
            sector_damage=self.sector_damage,
            transform_type=self.transform_type,
            man_type=self.man_type,
            bg_path=self.bg_path,
            bounding_axes=self.bounding_axes,
            fg_coords=self.fg_coords,
            fg_size=self.fg_size
        )
        
    def write_label_retinanet(self, labels_file, damage_labelling=True):
        axes = self.bounding_axes
        bounds = f"{axes[0]} {axes[1]} {axes[2]} {axes[3]}"
        if damage_labelling:
            labels_file.write(f"{self.fg_path} {bounds} class={self.class_num} "
                              f"{self.damage_type}={self.damage_tag} damage={self.damage_ratio} "
                              f"sector_damage={self.sector_damage} "
                              f"transform_type={self.transform_type} man_type={self.man_type} "
                              f"bg={self.bg_path}\n")
        else:
            labels_file.write(f"{self.fg_path} {bounds} class={self.class_num} "
                              f"transform_type={self.transform_type} man_type={self.man_type} "
                              f"bg={self.bg_path}\n")
            
    def write_label_coco(self, labels_dict, id, img_path, img_dims, damage_labelling=True):
        axes = self.bounding_axes
        labels_dict['images'].append({
            "id": id,
            "background_name": Path(self.bg_path).stem,
            "license": 1,
            "file_name": img_path,
            "height": img_dims[0],
            "width": img_dims[1],
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        labels_dict['annotations'].append({
            "id": id,
            "image_id": id,
            "category_id": self.class_num,
            "bbox": [axes[0], axes[2], axes[1] - axes[0], axes[3] - axes[2]],
            "area": (axes[1] - axes[0]) * (axes[3] - axes[2]),
            "segmentation": [],
            "iscrowd": 0,
        })
        if damage_labelling:
            labels_dict['annotations'][-1]['damage'] = self.damage_ratio
            labels_dict['annotations'][-1]['damage_type'] = self.damage_type
            labels_dict['annotations'][-1]['sector_damage'] = self.sector_damage

    def __check_class(self, class_num):
        if class_num < 0:
            raise TypeError(f"class_num={class_num} is invalid: must be >= 0")

    def __check_damage(self, damage_ratio):
        if damage_ratio < 0.0 or damage_ratio > 1.0:
            raise TypeError(f"damage_ratio={damage_ratio} is invalid: must have 0.0 <= damage_ratio <= 1.0")

    def __check_sector_damage(self, sector_damage_ratios):
        for damage_ratio in sector_damage_ratios:
            if damage_ratio < 0.0 or damage_ratio > 1.0:
                raise TypeError(f"{sector_damage_ratios} is invalid: must have 0.0 <= damage_ratio <= 1.0")

    def __check_transformation(self, transform_type):
        if not isinstance(transform_type, str) and transform_type < 0:
            raise TypeError(f"transform_type={transform_type} is invalid: must be >= 0")

    def __check_manipulation(self, man_type):
        if man_type is None:
            raise TypeError(f"man_type={man_type} is invalid: must be valid string")
        
    @staticmethod
    def gen_sign_coords(bg_dims, fg_dims):
        """Randomly generate sign coordinates and sign size."""
        bg_height, bg_width = bg_dims
        _, fg_width = fg_dims
        
        current_ratio = fg_width / bg_width  
        target_ratio = random.uniform(0.033, 0.099)
        scale_factor = target_ratio / current_ratio
        new_size = int(fg_width * scale_factor)
        
        # Randomise sign placement within middle third of background
        fg_x = random.randint(0, bg_width - new_size)
        third = bg_height // 3
        if(new_size>third):
            fg_y = random.randint(0, bg_height-new_size)
        else:
            fg_y = random.randint(third, bg_height - third)
                
        return fg_x, fg_y, new_size
