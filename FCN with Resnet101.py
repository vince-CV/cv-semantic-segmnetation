from torchvision import models
import torch
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

import urllib

import numpy as np
import time
import cv2

def download_image(url, filename):
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
        
    return


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    
    return rgb


cars = ('https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg', 'cars.jpg')


fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True).eval()

download_image(cars[0], cars[1])
img = Image.open(cars[1]).convert('RGB')

trf = T.Compose([T.Resize(640), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

inp = trf(img).unsqueeze(0)

out = fcn_resnet101(inp)['out']                                    # infer model
print ("model output size: ", out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()     # apply argmax along the dim=1 to get predicted labels for each pixel
print ("Class type: ", np.unique(om))


rgb = decode_segmap(om)

cv2.imshow("segment results", rgb)
cv2.waitKey()









