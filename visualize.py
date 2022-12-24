from __future__ import absolute_import, division, print_function
from fileinput import filename
#matplotlib inline

import os
import glob
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist
import time
from tqdm import tqdm
model_name = "mono_1024x320"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();

img_clear_list =[]
image_path = "/data1/cth/"
with open('/data1/cth/nuscenes/val_img.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        #data_path = image_path + line
        data_path = line
        img_clear_list.append(data_path)

save_path_list = ['nuscenes/depth/CAM_FRONT/','nuscenes/depth/CAM_FRONT_RIGHT/','nuscenes/depth/CAM_FRONT_LEFT/',
                    'nuscenes/depth/CAM_BACK/','nuscenes/depth/CAM_BACK_LEFT/','nuscenes/depth/CAM_BACK_RIGHT/']
#img_clear_list = sorted(glob.glob(image_path+"*"))

for img in tqdm(img_clear_list):
    file_name = str(img).split('/')[-1]

    if '__CAM_FRONT__' in file_name:
        save_path = image_path + save_path_list[0]
    elif '__CAM_FRONT_RIGHT__' in file_name:
        save_path = image_path + save_path_list[1]
    elif '__CAM_FRONT_LEFT__' in file_name:
        save_path = image_path + save_path_list[2]
    elif '__CAM_BACK__' in file_name:
        save_path = image_path + save_path_list[3]
    elif '__CAM_BACK_LEFT__' in file_name:
        save_path = image_path + save_path_list[4]
    elif '__CAM_BACK_RIGHT__' in file_name:
        save_path = image_path + save_path_list[5]

    input_image = pil.open(img).convert('RGB')
    original_width, original_height = input_image.size
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    depth_img = pil.fromarray(disp_resized_np)
    #save_name = str(img).split('/')[-1].replace(".jpg",".tif")
    save_name = file_name.replace(".jpg",".tif")
    save_ = save_path + save_name
    depth_img.save(save_)
'''
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.imshow(input_image)
plt.title("Input", fontsize=22)
plt.axis('off')

plt.subplot(212)
plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
plt.title("Disparity prediction", fontsize=22)
plt.axis('off');
'''