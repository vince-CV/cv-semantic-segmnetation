from torchvision import models
import torch
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

import urllib

import numpy as np
import time
import glob
import os

unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

def get_images_and_mask_images(root='C:/Users/xwen2/Downloads/mri-images'):
    mask_images = glob.glob(os.path.join(root, '*_mask.tif'))
    
    images = []
    
    for m_img in mask_images:
        ind = m_img.find('_mask.tif')
        image_path = m_img[:ind] + '.tif'
        images.append((image_path, m_img))
    return images

images = get_images_and_mask_images()

preprocess = T.Compose([ T.ToTensor(),])


def brain_mri_abnormally_detection(model, img_path, mask_path, cuda_device=None):
    
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure
    img = Image.open(img_path).convert("RGB")
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.gca().set_title('Input Image')
    plt.axis('off')
    
    mask_img = Image.open(mask_path).convert("RGB")
    plt.subplot(1, 3, 2)
    plt.imshow(mask_img)
    plt.gca().set_title('Mask (G-truth)')
    plt.axis('off')
    
    input_tensor = preprocess(img)

    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to(cuda_device)
        model = model.to(cuda_device)

    with torch.no_grad():
        output = unet(input_batch)


    pred = torch.round(output[0]).squeeze().detach().cpu().numpy()
    
    r = np.zeros_like(pred).astype(np.uint8)
    g = np.zeros_like(pred).astype(np.uint8)
    b = np.zeros_like(pred).astype(np.uint8)

    idx = pred == 1
    r[idx] = 100
    pred_img = np.stack([r, g, b], axis=2)
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_img)
    plt.gca().set_title('Pred Segments')
    plt.axis('off')
    plt.show()
    
    return

device = 'cuda:0'

for i in range(4):

    brain_mri_abnormally_detection(unet, images[i][0], images[i][1], device)

