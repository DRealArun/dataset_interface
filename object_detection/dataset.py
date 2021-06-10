from typing import Sequence, Dict

import os
import yaml
import glob
import cv2
import random
import numpy as np
# import xmltodict

from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader

class DatasetCustom(object):
    def __init__(self, root, transforms, split_name):
        self.root = root
        self.transforms = transforms
        self.split_name = split_name
        self.data_list = []
        if os.path.isdir(self.root):
            self.data_list = self.__get_image_data(os.path.join(root, 'images', self.split_name),
                                                   os.path.join(root, self.split_name + '.yaml'))

    def __get_image_data(self, image_dir: str,
                         annotations_file_path: str) -> Sequence[Dict[str, str]]:
        '''Returns a list of dictionaries containing image paths and image
        annotations for all images in the given directory. Each dictionary
        in the resulting list is of the following format:
        {
            'img': path to an image (starting from image_dir),
            'annotations': a dictionary containing class labels and bounding
                           box annotations for all objects in the image
        }

        Keyword arguments:
        image_dir: str -- name of a directory with RGB images
        annotations_file_path: str -- path to an image annotation file

        '''
        image_list = []
        annotations = {}
        with open(annotations_file_path, 'r') as annotations_file:
            annotations = yaml.load(annotations_file)

        for x in os.listdir(image_dir):
            name, _ = x.split('.')
            image_data = {}
            img_name = name + '.jpg'

            image_data['img'] = os.path.join(image_dir, img_name)
            image_data['annotations'] = annotations[img_name]
            image_list.append(image_data)
        return image_list

    def __getitem__(self, idx: int):
        img_path = self.data_list[idx]['img']
        annotations = self.data_list[idx]['annotations']

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for annotation in annotations:
            xmin = annotation['xmin']
            xmax = annotation['xmax']
            ymin = annotation['ymin']
            ymax = annotation['ymax']
            boxes.append([xmin, ymin, xmax, ymax])

            label = annotation['class_id']
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data_list)


class DatasetVOC(object):
    def __init__(self, root, transforms, split_name, class_metadata):
        self.root = root
        self.transforms = transforms
        self.split_name = split_name

        self.data_list = []
        if os.path.isdir(self.root):
            self.data_list = self.__get_image_data(os.path.join(root, 'images', self.split_name),
                                                   os.path.join(root, 'annotations', self.split_name))

        self.class_metadata = class_metadata

    def __get_image_data(self, image_dir: str,
                         annotations_dir: str) -> Sequence[Dict[str, str]]:
        '''Returns a list of dictionaries containing image paths and image
        annotations for all images in the given directory. Each dictionary
        in the resulting list is of the following format:
        {
            'img': path to an image (starting from image_dir),
            'annotations': a dictionary containing class labels and bounding
                           box annotations for all objects in the image
        }

        Keyword arguments:
        image_dir: str -- name of a directory with RGB images
        annotations_file_path: str -- path to an image annotation file

        '''
        image_list = []

        for x in os.listdir(image_dir):
            name, _ = x.split('.')
            image_data = {}
            img_name = name + '.jpg'
            annotation_name = name + '.xml'

            image_data['img'] = os.path.join(image_dir, img_name)

            annotation = None
            # with open(os.path.join(annotations_dir, annotation_name), 'r') as annotations_file:
            #     image_data['annotations'] = xmltodict.parse(annotations_file.read())

            image_list.append(image_data)
        return image_list

    def __getitem__(self, idx: int):
        img_path = self.data_list[idx]['img']
        annotations = self.data_list[idx]['annotations']

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        if type(annotations['annotation']['object']) == list:
            for annotation in annotations['annotation']['object']:
                xmin = int(annotation['bndbox']['xmin'])
                xmax = int(annotation['bndbox']['xmax'])
                ymin = int(annotation['bndbox']['ymin'])
                ymax = int(annotation['bndbox']['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])

                label = self.class_metadata[annotation['name']]
                labels.append(label)
        else:
            annotation = annotations['annotation']['object']
            xmin = int(annotation['bndbox']['xmin'])
            xmax = int(annotation['bndbox']['xmax'])
            ymin = int(annotation['bndbox']['ymin'])
            ymax = int(annotation['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])

            label = self.class_metadata[annotation['name']]
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data_list)


class OnlineImageComposer(Dataset):
    def __init__(self, root, label_config, background_dict,
                 range_dict, transforms, invert_mask=False, syn_prob=0.65):
        self._root = root
        self._label_config = label_config
        self._range_dict = range_dict
        self._background_dict = background_dict
        self._invert_mask = invert_mask
        self.transforms = transforms
        self._generate_operator_map()
        self._syn_prob = syn_prob
        self.parse_dataset()
        self._length = len(self.img_label_map.keys())

    def parse_dataset(self):
        label_2_folder_map = {}
        with open(self._label_config, 'r') as stream:
            try:
                label_2_folder_map = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.img_label_map = {}
        self.label_folder_map = {}
        index = 1 # Since oth index is for __background class
        for folder in label_2_folder_map.values():
            folder_path = os.path.join(self._root, folder)
            if os.path.exists(folder_path):
                self.label_folder_map[index] = folder
                for f in glob.iglob(os.path.join(folder_path, '*.jpg')):
                    self.img_label_map[f] = index
                index += 1

    def __len__(self):
        return self._length

    def _get_mask_path(self, img_path):
        temp = img_path.split('/')[-1]
        path_name = img_path.split(temp)[0]
        anno_path = os.path.join(path_name, 'masks', temp.split('.jpg')[0] + "_mask.pbm")
        return anno_path

    def _generate_operator_map(self):
        self._operator_map = {
            'brightness': self.brightness_operator,
            'sharpness': self.sharpness_operator,
            'shear_x': self.shear_x_operator,
            'shear_y': self.shear_y_operator,
            'contrast': self.contrast_operator,
            'rotation': self.rotation_operator,
            'resize': self.resize_operation,
        }

    def brightness_operator(self, img, mask, factor):
        return ImageEnhance.Brightness(img).enhance(factor), mask

    def sharpness_operator(self, img, mask, factor):
        return ImageEnhance.Sharpness(img).enhance(factor), mask

    def contrast_operator(self, img, mask, factor):
        return ImageEnhance.Contrast(img).enhance(factor), mask

    def shear_x_operator(self, img, mask, x_level):
        return img.transform(img.size, Image.AFFINE,
                             (1, x_level, 0, 0, 1, 0)), mask

    def shear_y_operator(self, img, mask, y_level):
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, 0, y_level, 1, 0)), mask

    def rotation_operator(self, img, mask, deg):
        rot_img = img.rotate(deg)
        rot_mask = mask.rotate(deg)
        return rot_img, rot_mask

    def resize_operation(self, img, mask, factor):
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        if factor == 1:
            return img, mask
        else:
            old_width = img.width
            old_height = img.height
            new_width = int(old_width * factor)
            new_height = int(old_height * factor)
            if factor < 1:
                sampling = Image.HAMMING
            else:
                sampling = Image.BILINEAR
            resized_img = img.resize((new_width, new_height), sampling)
            resized_mask = mask.resize((new_width, new_height), sampling)
            if factor < 1:
                # Since the original image background is mostly white
                # Hence we make an educated guess to use white as the
                # extrapolated color when padding the downscaled image
                op_img = Image.new('RGB', (old_width, old_height), (255, 255, 255))
                op_mask = Image.new('L', (old_width, old_height), 0)
                cx = old_width // 2
                cy = old_height // 2
                x_min = cx - new_width // 2
                y_min = cy - new_height // 2
                position = (x_min, y_min)
                op_img.paste(resized_img, position)
                op_mask.paste(resized_mask, position)
            else:
                cx = new_width // 2
                cy = new_height // 2
                box = (cx - old_width // 2, cy - old_height // 2,
                       cx + old_width // 2, cy + old_height // 2)
                op_img = resized_img.crop(box)
                op_mask = resized_mask.crop(box)
            return op_img, op_mask

    def _draw_one_discrete_sample(self, min_val, max_val, increment):
        # Includes the min_val and max_val
        discrete_space = np.arange(min_val, max_val + increment, increment)
        # For floating point arguments, the length of the result is ceil((stop - start)/step).
        # Because of floating point overflow, this rule may result in the last element of out
        # being greater than stop. Hence add an upper bound.
        discrete_space = np.minimum(max_val, discrete_space)
        return np.random.choice(discrete_space)

    def _make_image_transparent(self, img, mask):
        img = img.convert("RGBA")
        img_array = np.array(img)
        img_array[:, :, 3] = np.array(mask)
        img = Image.fromarray(img_array)
        return img

    def _get_composed_image(self, fg_img, pil_fg_mask, bg_img):
        # TODO Add another argument called min_percent_visibile to account for partial
        # object view
        fg_mask = np.array(pil_fg_mask)
        y_vals, x_vals = np.where(fg_mask != 0)
        w, h = fg_img.size
        x_min = max(min(x_vals), 0)
        y_min = max(min(y_vals), 0)
        x_max = min(max(x_vals), w - 1)
        y_max = min(max(y_vals), h - 1)
        max_pos_x_delta = (w - 1) - x_max
        max_pos_y_delta = (h - 1) - y_max
        min_neg_x_delta = -x_min
        min_neg_y_delta = -y_min
        x_delta = self._draw_one_discrete_sample(min_neg_x_delta, max_pos_x_delta, 1)
        y_delta = self._draw_one_discrete_sample(min_neg_y_delta, max_pos_y_delta, 1)
        box = (x_min, y_min, x_max, y_max)

        cropped_fg_img = fg_img.crop(box)
        cropped_fg_mask = pil_fg_mask.crop(box)
        transparent_fg = self._make_image_transparent(cropped_fg_img, cropped_fg_mask)

        composed_img = bg_img.copy()
        position = (x_min + x_delta, y_min + y_delta)
        composed_img.paste(transparent_fg, position, transparent_fg)

        composed_mask = Image.new('L', fg_img.size, 0)
        composed_mask.paste(cropped_fg_mask, position, cropped_fg_mask)
        return composed_img, composed_mask

    def _sample_solid_background(self, dim):
        # dim in (width, height)
        RGB_values = self._background_dict['solid_colors']
        RGB_tuple = random.choice(RGB_values)
        return Image.new('RGB', dim, RGB_tuple)

    def _sample_image_background(self, dim):
        # dim in (width, height)
        img_paths = self._background_dict['img_background_paths']
        img_path = random.choice(img_paths)
        return self._read_image(img_path).resize(dim)

    def _sample_background(self, fg_img, fg_mask):
        if 'solid_colors' in self._background_dict.keys() and \
                'img_background_paths' in self._background_dict.keys():
            if random.random() < 0.5:
                bg_img = self._sample_solid_background(fg_img.size)
            else:
                bg_img = self._sample_image_background(fg_img.size)
        elif 'img_background_paths' in self._background_dict.keys():
            bg_img = self._sample_image_background(fg_img.size)
        else:
            bg_img = self._sample_solid_background(fg_img.size)
        return bg_img

    def _apply_transform(self, op_id, img, mask, range_tuple):
        min_val, max_val, inc = range_tuple
        mag = self._draw_one_discrete_sample(min_val, max_val, inc)
        #         print("Operation:", op_id, "Magnitude:", mag)
        op_img, op_mask = self._operator_map[op_id](img, mask, mag)
        return op_img, op_mask

    def _synthesize_image(self, fg_img, fg_mask):
        op_img = fg_img
        op_mask = fg_mask
        # Apply image transformation operations
        for key in self._range_dict['img_tf_ranges'].keys():
            range_tuple = self._range_dict['img_tf_ranges'][key]
            op_img, op_mask = self._apply_transform(key, op_img, op_mask, range_tuple)

        # Apply image composition
        if 'solid_colors' in self._background_dict.keys() or \
                'img_background_paths' in self._background_dict.keys():
            bg_img = self._sample_background(op_img, op_mask)
            op_img, op_mask = self._get_composed_image(op_img, op_mask, bg_img)

        # Apply image appearance modifications
        for key in self._range_dict['img_app_ranges'].keys():
            range_tuple = self._range_dict['img_app_ranges'][key]
            op_img, op_mask = self._apply_transform(key, op_img, op_mask, range_tuple)
        return op_img, op_mask

    def _read_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        pil_mask = Image.fromarray(mask)
        if self._invert_mask:
            pil_mask = ImageOps.invert(pil_mask)
        return pil_mask

    def _get_bbox(self, pil_fg_mask):
        fg_mask = np.array(pil_fg_mask)
        y_vals, x_vals = np.where(fg_mask != 0)
        w, h = pil_fg_mask.size
        x_min = max(min(x_vals), 0)
        y_min = max(min(y_vals), 0)
        x_max = min(max(x_vals), w - 1)
        y_max = min(max(y_vals), h - 1)
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = list(self.img_label_map.keys())[idx]
        image = self._read_image(img_path)
        mask = self._read_mask(self._get_mask_path(img_path))
        if random.random() <= self._syn_prob:
            image, mask = self._synthesize_image(image, mask)
        boxes = [self._get_bbox(mask)]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([self.img_label_map[img_path]])
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
