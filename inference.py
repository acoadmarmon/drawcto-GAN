from model import unet
import torch
import cv2
import numpy as np
import data_transforms as dt

netG = unet.ResNetUNet(6)

netG.load_state_dict(torch.load('generator_model.torch', map_location=torch.device('cpu')))

def inference(image_path):
    im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
    img = None
    if im.shape[-1] == 4:
        BGR = im[...,0:3].astype(np.float)/255
        A   = im[...,3].astype(np.float)/255 
        bg  = np.zeros_like(BGR).astype(np.float)+1   # white background
        fg  = A[...,np.newaxis]*BGR                   # new alpha-scaled foreground
        bg = (1-A[...,np.newaxis])*bg                 # new alpha-scaled background
        res = cv2.add(fg, bg)                         # sum of the parts
        res = (res*255).astype(np.uint8)              # scaled back up
        img = res
    else:
        img = im
    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1).astype(float) / 255.0
    img = dt.Rescale((256,256))({'image': img, 'random_crop_image': img})['image']
    img = dt.ToTensor()({'image': img, 'random_crop_image': img})['image']

    fake = netG(img.unsqueeze(0).float()).squeeze(0).detach().numpy().transpose(1, 2, 0)
    
    fake = cv2.GaussianBlur(fake,(3,3),cv2.BORDER_DEFAULT)
    fake[fake < 1.0] = 0.0
    fake[fake >= 1.0] = 1.0
    fake = np.expand_dims(fake, axis=2)
    old_img = img.detach().numpy().transpose(1, 2, 0)
    img = fake + img.detach().numpy().transpose(1, 2, 0)
    img[img <= 1.0] = 0.0
    img[img > 1.0] = 1.0

    
    img = cv2.cvtColor(cv2.resize(img, (600, 600), interpolation = cv2.INTER_AREA).astype('float32'),cv2.COLOR_GRAY2RGB)*256.0
    cv2.imwrite('new_canvas.png', img)
    return img

if __name__ == "__main__":
    inference('./new_canvas.png')