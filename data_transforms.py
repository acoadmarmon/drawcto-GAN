from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
from skimage import io, transform
import torch

def get_test_transforms(input_size):

  return transforms.Compose([
      Rescale(input_size),
      ToTensor()
  ])

def get_train_transforms(input_size):
  return transforms.Compose([
      Rescale(input_size),
      #transforms.RandomHorizontalFlip()
      ToTensor()
  ])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, cropped_image = sample['image'], sample['random_crop_image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        cropped_image = cropped_image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'random_crop_image': torch.from_numpy(cropped_image)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, cropped_image = sample['image'], sample['random_crop_image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        random_crop_img = transform.resize(cropped_image, (new_h, new_w))


        img[img < img.mean()] = 0.0
        img[img >= img.mean()] = 1.0

        random_crop_img[random_crop_img < random_crop_img.mean()] = 0.0
        random_crop_img[random_crop_img >= random_crop_img.mean()] = 1.0

        return {'image': img, 'random_crop_image': random_crop_img}