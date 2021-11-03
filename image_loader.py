from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import data_transforms
import torch
import os

import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from skimage import io, transform
import random

import cv2
import math
import matplotlib.pyplot as plt

def get_image_dataset(root_dir='./', split='train', resize=(256, 256), batch_size=16, shuffle=True, num_workers=0):
    
    curr_transforms = {
    'train': data_transforms.get_train_transforms(resize),
    'val': data_transforms.get_test_transforms(resize)}
    dataset = AbstractArtDataset(root_dir, curr_transforms[split])

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_random_line(sketch_img, max_line_len=250):
    def is_valid_coord(x_coord, y_coord, img_shape):
        return x_coord >= 0 and x_coord < img_shape[0] and y_coord >= 0 and y_coord < img_shape[1]

    def get_neighbor_pixels(x_coord, y_coord, img_shape):
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        pixel_coord = [(x_coord + i[0], y_coord + i[1]) for i in directions if is_valid_coord(x_coord + i[0], y_coord + i[1], img_shape)]
        return pixel_coord

    # Find random non-white point on the image
    result = np.where(sketch_img != 1.0)
    list_of_indices = list(zip(result[0], result[1]))
    if len(list_of_indices) == 0:
      return sketch_img

    starting_pixel = random.choice(list_of_indices)

    list_of_chosen_indices = set()
    frontier = get_neighbor_pixels(starting_pixel[0], starting_pixel[1], sketch_img.shape[:-1])
    
    while len(frontier) > 0 and len(list_of_chosen_indices) < max_line_len:
        curr_coord = frontier.pop(0)

        if sketch_img[curr_coord[0], curr_coord[1], 0] != 1.0:
            list_of_chosen_indices.add(curr_coord)

        neighbors = get_neighbor_pixels(curr_coord[0], curr_coord[1], sketch_img.shape[:-1])
        for neighbor in neighbors:
            if neighbor not in list_of_chosen_indices and neighbor not in frontier:
                if sketch_img[neighbor[0], neighbor[1], 0] != 1.0:
                  frontier.append(neighbor)

    return_img = np.zeros(sketch_img.shape, dtype=np.float)
    return_img[:, :, :] = 1.0

    for coord in list_of_chosen_indices:
        return_img[coord[0], coord[1], :] = sketch_img[coord[0], coord[1], :]
    
    return return_img

def get_random_crop_image(img):
    def generate_remove_tile_list(options=[0, 1, 2, 3]):
       if len(options) == 1:
         return [[]]

       return_options = []
       for option in options:
         options_list = generate_remove_tile_list([i for i in options if i != option])
         for i in options_list:
            return_options.append([option] + i)

       return return_options

    all_options = list(generate_remove_tile_list([0, 1, 2, 3]))
    
    option = random.choice(all_options)
    option = random.choice([option, random.choices(option, k=2), random.choices(option, k=1)])

    img_shape = img.shape

    if img_shape[1] > 255:
      img = img[:, :255, :]
    img_shape = img.shape

    M, N = (math.ceil(img.shape[0]/2), math.ceil(img.shape[1]/2))

    cropped_image = np.zeros_like(img)
    k = 0
    for x in range(0,img_shape[0],M):
        for y in range(0,img_shape[1],N):
            if k in option:
              cropped_image[x:x+M,y:y+N] = img[x:x+M,y:y+N]
            else:
              cropped_image[x:x+M,y:y+N] = 255

            k = k + 1
            
    return cropped_image


class AbstractArtDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.image_names = [i for i in os.listdir(root_dir) if '.png' in i]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx)
        img_name = os.path.join(self.root_dir, self.image_names[idx])

        im = cv2.imread(img_name, cv2.IMREAD_UNCHANGED) 
        
        BGR = im[...,0:3].astype(np.float)/255
        A   = im[...,3].astype(np.float)/255
        bg  = np.zeros_like(BGR).astype(np.float)+1   # white background
        fg  = A[...,np.newaxis]*BGR                   # new alpha-scaled foreground
        bg = (1-A[...,np.newaxis])*bg                 # new alpha-scaled background
        res = cv2.add(fg, bg)                         # sum of the parts
        res = (res*255).astype(np.uint8)              # scaled back up
        img = res
        
        img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1).astype(float) / 255.0
        cropped_image = np.ones_like(img)

        for i in range(random.randint(1, 10)):
            new_img = get_random_line(img)
            new_img[new_img < 1.0] = 0.0
            new_img[new_img >= 1.0] = 1.0

            cropped_image[cropped_image < 1.0] = 0.0
            cropped_image[cropped_image >= 1.0] = 1.0
            cropped_image += new_img

            cropped_image[cropped_image <= 1.0] = 0.0
            cropped_image[cropped_image > 1.0] = 1.0
        
        target_image = get_random_line(img)
        target_image[target_image < 1.0] = 0.0
        target_image[target_image >= 1.0] = 1.0
        
        sample = {'image': target_image, 'random_crop_image': cropped_image}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":



    def show_landmarks_batch(sample_batched):
      """Show image with landmarks for a batch of samples."""
      images_batch, cropped_batch = \
              sample_batched['image'], sample_batched['random_crop_image']
      batch_size = len(cropped_batch)
      im_size = cropped_batch.size(2)
      grid_border_size = 2

      
      grid = utils.make_grid(cropped_batch)
      plt.imshow(grid.numpy().transpose((1, 2, 0)))

    dataloader = get_image_dataset('./data/processed_images/')
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['random_crop_image'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()