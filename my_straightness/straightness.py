import os
import cv2
import time
import glob
import numpy as np
from skimage.exposure import match_histograms

# checks if the image is blurry
def check_blur(image, threshold=10):
    blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(blur, cv2.CV_64F).var()
    if fm < threshold:
        return 0
    else:
        return 1

# checks for the brightness level of the image
def brightness_level(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    return np.mean(v)

# create gamma correction table using the lookup table
def adjust_gamma(image, gamma = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# apply required gamma value depending on image illumination
def gamma_correction(image, gamma):
    if brightness_level(image) < 8.0: # enhance gamma level
        gamma = 1.3
        adjusted = adjust_gamma(image, gamma=gamma)
    elif brightness_level(image) > 13.0: # decrease gamma level
        gamma = 0.7
        adjusted = adjust_gamma(image, gamma=gamma)
    return adjusted

# matches histogram with the reference image provided
def match_histogram(image, ref):
    matched = match_histograms(image, ref, channel_axis=-1)
    return matched
    
# stitching all images all at once using openCV stitcher
def stitching(images, mode = 0):
    stitchy=cv2.Stitcher.create()

    if mode == 0: # stitches all images at once
        print(f'Stitching all images together.')
        (dummy,output)=stitchy.stitch(images)
        return output
    else: # stitches images in order to avoid unnecessary errors
        print(f'Stitching images one by one.')
        output = images[0]
        for i in range(len(images)-1):
            imgs_ = [output,images[i+1]]
            (dummy,output)=stitchy.stitch(imgs_)
        return output

def read_images(ROI_start_y=900,ROI_region=1100,path=None):
    imgs = []
    if path == None:
        pwd = os.path.dirname(os.path.abspath(__file__))
        images = glob.glob(pwd+'\\images\\*.bmp')
        images.sort()
        start = time.time()
        for i in images:
            img = cv2.imread(i)
            img = img[ROI_start_y:(ROI_start_y+ROI_region), 0:img.shape[1]]
            # img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            imgs.append(img)
        end = time.time()
        print(f'time taken to read images: {end-start}')
        # print(f'Read {len(images)} images successfully.')
        return imgs
    else:
        images = glob.glob(path)
        images.sort()
        for i in images:
            img = cv2.imread(i)
            img = img[ROI_start_y:(ROI_start_y+ROI_region), 0:img.shape[1]]
            # img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            imgs.append(img)
        # print(f'Read {len(images)} images successfully.')
        return imgs
    
def read_images_unchanged(ROI_start_y=900,ROI_region=1100,path=None):
    imgs = []
    if path == None:
        pwd = os.path.dirname(os.path.abspath(__file__))
        images = glob.glob(pwd+'\\images\\*.bmp')
        images.sort()
        start = time.time()
        for i in images:
            img = cv2.imread(i,cv2.IMREAD_UNCHANGED)
            IMAGE = img[ROI_start_y:(ROI_start_y+ROI_region), 0:img.shape[1]]  # y:y+h, x:x+w
            IMAGE_int = np.array(IMAGE).astype(int)
            IMAGE_3D_MATRIX = cv2.merge((IMAGE_int, IMAGE_int, IMAGE_int)) if len(IMAGE_int.shape) < 3 else IMAGE_int
            imgs.append(IMAGE_3D_MATRIX)
        end = time.time()
        print(f'time taken to read unchanged images: {end-start}')
        # print(f'Read {len(imgs)} unchanged_images successfully.')
        return imgs
    else:
        images = glob.glob(path)
        images.sort()
        for i in images:
            img = cv2.imread(i,cv2.IMREAD_UNCHANGED)
            IMAGE = img[ROI_start_y:(ROI_start_y+ROI_region), 0:img.shape[1]]  # y:y+h, x:x+w
            IMAGE_int = np.array(IMAGE).astype(int)
            IMAGE_3D_MATRIX = cv2.merge((IMAGE_int, IMAGE_int, IMAGE_int)) if len(IMAGE_int.shape) < 3 else IMAGE_int
            imgs.append(IMAGE_3D_MATRIX)
        # print(f'Read {len(imgs)} unchanged_images successfully.')
        return imgs