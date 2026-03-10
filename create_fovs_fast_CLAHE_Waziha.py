## Project: AMD IRIS data
## Re-written: Dr. Waziha Kabir
## July 28, 2022

import os, sys, argparse
import os.path as osp
import numpy as np
from utils.get_mask import crop_to_fov
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm
from skimage.measure import regionprops
import cv2

parser = argparse.ArgumentParser()
#parser.add_argument('--im_path', type=str, default='/home/shared/IRIS_AMD_model_training/IRIS_AMD_test_images_MC_V1/', help='path to training data')
#parser.add_argument('--im_path_out_wk', type=str, default='data/cropped_images_waziha/', help='path data')
parser.add_argument('--im_path_out', type=str, default='data/cropped_images/', help='path data without enhancement')
parser.add_argument('--im_path_out_enh', type=str, default='data/cropped_images_7567_waziha/cropped_images_wenh/', help='path data with enhancement')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')



if __name__ == '__main__':

    args = parser.parse_args()
    #im_path_out_wk = args.im_path_out_wk
    im_path_out = args.im_path_out
    im_path_out_enh = args.im_path_out_enh
    #im_path = args.im_path
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    #os.makedirs(im_path_out_wk, exist_ok=True)
    os.makedirs(im_path_out, exist_ok=True)
    os.makedirs(im_path_out_enh, exist_ok=True)
    im_list = sorted(os.listdir(im_path_out))
    rsz = Resize(tg_size)
    print('total amount of images = {}'.format(len(im_list)))
    print('to be stored at {}'.format(im_path_out))
    for i in tqdm(range(len(im_list))):
        im_name = im_list[i]
        img_crop = cv2.imread(osp.join(im_path_out,im_name))
        #print('image shape is:', img_clahe.shape)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_1 = clahe.apply(img_crop[:,:,0]) * 1
        img_2 = clahe.apply(img_crop[:,:,1]) * 1
        img_3 = clahe.apply(img_crop[:,:,2]) * 1
        img_4 = cv2.merge([img_1,img_2,img_3])
        #print('merged clahe image shape is: ', img_4.shape)
        cv2.imwrite(osp.join(im_path_out_enh,im_name),img_4)
