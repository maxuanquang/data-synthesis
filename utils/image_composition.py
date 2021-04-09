#!/usr/bin/env python3

import json
import warnings
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
from pascal_voc_writer import Writer as xml_writer


class ImageComposition():
    """ 
    Composes images together in random ways, applying transformations to the foreground to create a synthetic
    combined image.
    """

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8 # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 2
        self.foreground_size = 0.03 # diện tích foreground so với background

    def _validate_and_process_args(self, args):
        # Validates input arguments and sets up class variables
        # Args:
        #     args: the ArgumentParser command line arguments

        self.silent = args.silent

        # Validate the count - so luong anh generate
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate the width and height - kich thuoc anh generate
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        # Validate and process the output type - mac dinh output la jpg
        if args.output_type is None:
            self.output_type = '.jpg' # default
        else:
            if args.output_type[0] != '.': # xu ly neu khong co "."
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}' # xu ly file output

        # Validate and process output and input directories
        self._validate_and_process_output_directory()
        self._validate_and_process_input_directory()

    def _validate_and_process_output_directory(self):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images' # return 'output_dir/images'
        self.xml_output_dir = self.output_dir / 'xml' # return 'output_dir/xml'

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)
        self.xml_output_dir.mkdir(exist_ok=True)

        if not self.silent:
            # Check for existing contents in the images directory
            for _ in self.images_output_dir.iterdir():
                # We found something, check if the user wants to overwrite files or quit
                should_continue = input('images output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

            for _ in self.xml_output_dir.iterdir():
                # We found something, check if the user wants to overwrite files or quit
                should_continue = input('xml output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_input_directory(self):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir does not exist: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if x.name == '_foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        # Expected directory structure:
        # + foregrounds_dir
        #     + category_dir
        #         + foreground_image.png

        self.foregrounds_dict = dict()

        for category_dir in self.foregrounds_dir.iterdir():
            if not category_dir.is_dir():
                warnings.warn(f'file found in foregrounds directory (expected super-category directories), ignoring: {category_dir}')
                continue

            # This is a category directory
            for image_file in category_dir.iterdir():
                if not image_file.is_file():
                    warnings.warn(f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                    continue
                if image_file.suffix != '.png':
                    warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                    continue

                # Valid foreground image, add to foregrounds_dict
                category = category_dir.name

                if category not in self.foregrounds_dict.keys():
                    self.foregrounds_dict[category] = []

                self.foregrounds_dict[category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue
            
            # Kiểm tra kích thước background
            background = Image.open(image_file)
            bg_width, bg_height = background.size
            max_crop_x_pos = bg_width - self.width
            max_crop_y_pos = bg_height - self.height
            assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(image_file)}'
            assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(image_file)}'

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generating {self.count} images with...')

        # Create all images/masks (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(1, self.max_foregrounds)
            foregrounds = []
            for _ in range(num_foregrounds):
                # Randomly choose a foreground
                category = random.choice(list(self.foregrounds_dict.keys()))
                foreground_path = random.choice(self.foregrounds_dict[category])

                # 1 foregrounds list có thể có nhiều category
                foregrounds.append({
                    'category':category,
                    'foreground_path':foreground_path,
                })

            # Compose foregrounds and background
            composite, bbox_info = self._compose_images(foregrounds, background_path) # return: bbox_info + composite

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}' # e.g. 00000023.jpg

            # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}' # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename # e.g. my_output_dir/images/00000023.jpg
            composite = composite.convert('RGB') # remove alpha
            composite.save(composite_path)

            # Save xml file to the annotations sub-directory
            xml_filename = f'{save_filename}.xml'
            xml_path = self.output_dir / 'xml' / xml_filename
            writer = xml_writer(composite_path,self.width,self.height)
            for (category,startX,startY,endX,endY) in bbox_info:
                writer.addObject(category,startX,startY,endX,endY)
            writer.save(xml_path)

    def _compose_images(self, foregrounds, background_path):
        # Composes a foreground image and a background image
        # Validation should already be done by now.
        # Args:
        #     foregrounds: a list of dicts with format:
        #       [{
        #           'category':category,
        #           'foreground_path':foreground_path,
        #       },...]
        #     background_path: the path to a valid background image
        # Returns:
        #     composite: the composed image
        #     bbox: the list of bbox tuples (startX, startY, endX, endY)

        # Initialize bounding box information list: bbox_info = [(category,startX,startY,endX,endY)]
        bbox_info = []

        # Open background and convert to RGBA
        background = Image.open(background_path)
        # background = self._transform_background(background_path)
        background = background.convert('RGBA')

        # Crop background to desired size (self.width x self.height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))

        for fg in foregrounds:
            # fg = {
            #   'category':category,
            #   'foreground_path':foreground_path
            # }
            fg_path = fg['foreground_path']

            # Perform transformations
            fg_image = self._transform_foreground(fg_path)

            # Choose a random x,y position for the foreground
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]

            while max_x_position <= 0 or max_y_position <= 0:
                fg_image = fg_image.resize((fg_image.size[0] // 2, fg_image.size[1] // 2), resample=Image.BICUBIC)
                max_x_position = composite.size[0] - fg_image.size[0]
                max_y_position = composite.size[1] - fg_image.size[1]

            while fg_image.size[0] * fg_image.size[1] >= self.foreground_size * composite.size[0] * composite.size[1]:
                fg_image = fg_image.resize((fg_image.size[0] // 2, fg_image.size[1] // 2), resample=Image.BICUBIC)
                max_x_position = composite.size[0] - fg_image.size[0]
                max_y_position = composite.size[1] - fg_image.size[1]

            # tạo ra bbox info và vị trí dán
            category = fg["category"]
            startX = int(random.randint(0, max_x_position))
            startY = int(random.randint(0, max_y_position))
            endX = int(startX + fg_image.size[0])
            endY = int(startY + fg_image.size[1])
            bbox_info.append((category,startX,startY,endX,endY))
            paste_position = (startX, startY)

            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color = (0, 0, 0, 0))
            new_fg_image.paste(fg_image, paste_position)

            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = fg_image.getchannel(3)
            new_alpha_mask = Image.new('L', composite.size, color = 0)
            new_alpha_mask.paste(alpha_mask, paste_position)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

        return composite, bbox_info

    def _transform_foreground(self,fg_path): # Translations, Rotations, Changes in scale, Shearing, Horizontal/Vertical flip
        # Open foreground and get the alpha channel
        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        assert np.any(fg_alpha == 0), f'foreground needs to have some transparency: {str(fg_path)}' # kenh alpha = 0 la transparent hoan toan

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # # Scale the foreground
        # scale = random.random() * .5 + .5 # Pick something between .5 and 1
        # new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        # fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .4 + .7 # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Horizontal or Vertical or not
        flip_type = random.random()
        if flip_type <= 0.5:
            fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)

        # Smoothing and Bluring
        blur_type = random.random()
        if blur_type <= 0.3:
            fg_image = fg_image.filter(ImageFilter.GaussianBlur(radius=2))
        if blur_type <= 0.6:
            fg_image = fg_image.filter(ImageFilter.BoxBlur(radius=2))
        if blur_type <= 1.0:
            fg_image = fg_image.filter(ImageFilter.MedianFilter(size=3))

        # Add any other transformations here...

        return fg_image

    def _transform_background(self,fg_path):
        # Open background
        fg_image = Image.open(fg_path)

        # ** Apply Transformations **
        # Adjust foreground brightness
        brightness_factor = random.random() * .2 + .7 # Pick something between 0.7 and 0.9
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Smoothing and Bluring
        blur_type = random.random()
        if blur_type <= 0.3:
            fg_image = fg_image.filter(ImageFilter.GaussianBlur(radius=2))
        if blur_type <= 0.6:
            fg_image = fg_image.filter(ImageFilter.BoxBlur(radius=2))
        if blur_type <= 1.0:
            fg_image = fg_image.filter(ImageFilter.MedianFilter(size=3))

        # Add any other transformations here...

        return fg_image

    # Start here
    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()
        # self._create_info()
        print('Image composition completed.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="The input directory. \
                        This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' directory which \
                        contains supercategory directories (e.g. 'animal', 'vehicle'), each of which contain category \
                        directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on a \
                        transparent background (e.g. a grizzly bear on a transparent background).")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="The directory where images, masks, \
                        and json files will be placed")
    parser.add_argument("--count", type=int, dest="count", required=True, help="number of composed images to create")
    parser.add_argument("--width", type=int, dest="width", required=True, help="output image pixel width")
    parser.add_argument("--height", type=int, dest="height", required=True, help="output image pixel height")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")
    parser.add_argument("--silent", action='store_true', help="silent mode; doesn't prompt the user for input, \
                        automatically overwrites files")

    args = parser.parse_args()

    image_comp = ImageComposition()
    image_comp.main(args)

    # python D:\cocosynth\python\image_composition.py --input_dir "D:\cocosynth\datasets\ring_dataset\input" --output_dir "D:\cocosynth\datasets\ring_dataset\output" --count 100 --width 1280 --height 720 