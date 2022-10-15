# Aman Kumar
# 2020CSB1153
# Task 1


# Importing Necessary Libraries
from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data 
from skimage.metrics import structural_similarity , peak_signal_noise_ratio
from skimage.color import *
from skimage import feature
from math import log10, sqrt
from skimage.io import imread


# Finding Gaussian Kernel
def gaussian_kernel(size, sigma):
    rng = size//2
    arr1 = np.linspace(-(rng),rng,size)
    for i in range(size):
        val = np.sqrt(2*np.pi)*sigma
        x = -np.power((arr1[i] - 0)/sigma, 2)/2
        val1 = np.e**(x)
        arr1[i] = (val)*val1
        arr1[i] = 1/arr1[i]
    final_output = np.outer(arr1.T, arr1.T)
 
    final_output = final_output * (1.0 / final_output.max())
 
    return final_output

# Convolving Image with Gaussian Kernel to Calculate Gaussian Blur
def convolution(img , kernel):
    img_r , img_c = img.shape
    krnl_r , krnl_c = kernel.shape

    op = np.zeros((img_r,img_c))

    ht = int((krnl_r-1)/2)
    wd = int((krnl_c-1)/2)

    pad_img = np.zeros((img_r + 2*ht , img_c + 2*wd))

    for i in range(img_r):
        for j in range(img_c):
            pad_img[i+ht][j+wd] = img[i][j]

    for i in range(img_r):
        for j in range(img_c):
            prod = kernel
            arr = pad_img[i:i+krnl_r,j:j+krnl_c]
            prod = prod*arr
            for i1 in range(krnl_r):
                for j1 in range(krnl_c):
                    op[i][j] += prod[i1][j1]

    return op

# Calculating Sobel Gradient and Direction
def calc_gradient(image):
    opr_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    img_x = convolution(image,opr_x)

    opr_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_y = convolution(image,opr_y)
    img_x1 = img_x**2
    img_y1 = img_y**2
    v = img_x1 + img_y1
    magnitude = np.sqrt(v)
    mx = magnitude.max()
    magnitude = magnitude*255.0
    magnitude/=mx


    dirn = np.arctan2(img_y,img_x)
    dirn = np.rad2deg(dirn)
    dirn+=180

    return magnitude,dirn

# Doing Non-Maximal Suppression on Input Image
def non_maximal_suppr(magnitude,direction):
    img_r , img_c = magnitude.shape
    fin = np.zeros((img_r,img_c))

    for i in range(1,img_r-1):
        for j in range(1,img_c-1):
            dirn = direction[i][j]

            if(0<= dirn < 180/8) or ( 15*180 / 8 <= dirn <= 180):
                bef = magnitude[i][j-1]
                aft = magnitude[i][j+1]
            elif (3*180/8 <= dirn < 5*180/8) or (11 * 180 / 8 <= dirn < 13 * 180 / 8):
                bef = magnitude[i-1][j]
                aft = magnitude[i+1][j]
            elif (180/8 <= dirn < 3*180/8) or (9*180/8 <= dirn < 11*180/8):
                bef = magnitude[i-1][j+1]
                aft = magnitude[i+1][j-1]
            else:
                bef = magnitude[i-1][j-1]
                aft = magnitude[i+1][j+1]
            
            if(magnitude[i][j]>=bef and magnitude[i][j]>=aft):
                fin[i][j] = magnitude[i][j]

    return fin

# Thresholding on input image with given low and high Threshold
def thresholding(img,low,high,weak):
    low = low*255
    high = high*255
    res = np.zeros(img.shape)
    img_r , img_c = img.shape

    for i in range(img_r):
        for j in range(img_c):
            if img[i][j] >= high:
                res[i][j] = 255
            if img[i][j]<=high and img[i][j]>=low:
                res[i][j] = weak
            
    return res

# Hysteresis on Input Image in all 4 directions
def hysteresis(img):
    img_r , img_c = img.shape

    arr = img.copy()

    res = img.copy()

    fin = np.zeros(img.shape)
    for k in range (0,4):
        if k<2:
            for i in range(1,img_r-1):
                if(k==0):
                    for j in range(1,img_c-1):
                        if arr[i][j] == 50:
                            if arr[i-1][j] == 255 or arr[i][j-1] == 255 or arr[i-1][j-1] == 255 or arr[i][j+1] == 255 or arr[i+1][j] == 255 or arr[i+1][j+1] == 255 or arr[i+1][j-1] == 255 or arr[i-1][j+1] == 255:
                                res[i][j]+= 255
                            else:
                                res[i][j] = 0   
                else:
                    for j in range(img_c-1,0,-1):
                        if arr[i][j] == 50:
                            if arr[i-1][j] == 255 or arr[i][j-1] == 255 or arr[i-1][j-1] == 255 or arr[i][j+1] == 255 or arr[i+1][j] == 255 or arr[i+1][j+1] == 255 or img[i+1][j-1] == 255 or arr[i-1][j+1] == 255:
                                res[i][j] += 255
                            else:
                                res[i][j] = 0
        else:
            for i in range(img_r-1,0,-1):
                if(k==2):
                    for j in range(1,img_c-1):
                        if arr[i][j] == 50:
                            if arr[i-1][j] == 255 or arr[i][j-1] == 255 or arr[i-1][j-1] == 255 or arr[i][j+1] == 255 or arr[i+1][j] == 255 or arr[i+1][j+1] == 255 or arr[i+1][j-1] == 255 or arr[i-1][j+1] == 255:
                                res[i][j]+= 255
                            else:
                                res[i][j] = 0   
                else:
                    for j in range(img_c-1,0,-1):
                        if arr[i][j] == 50:
                            if arr[i-1][j] == 255 or arr[i][j-1] == 255 or arr[i-1][j-1] == 255 or arr[i][j+1] == 255 or arr[i+1][j] == 255 or arr[i+1][j+1] == 255 or img[i+1][j-1] == 255 or arr[i-1][j+1] == 255:
                                res[i][j] += 255
                            else:
                                res[i][j] = 0
    
    for i in range (img_r):
        for j in range (img_c):
            if res[i][j] > 255:
                res[i][j] = 255

    return res

# Calculating PSNR Value
def psnr(img,img1):
    mn = np.mean((img-img1)**2)

    if(mn == 0):
        return 100
    
    val = 20 * log10(255/sqrt(mn))

    return val
    
def myCannyEdgeDetector(inputImg,low,high):
    img = inputImg
    if len(inputImg.shape) == 3:
        inputImg = rgb2gray(inputImg)
    kernel = gaussian_kernel(3,0.5)
    Gauss_Img = convolution(inputImg,kernel)
    Gauss_Img,dirn = calc_gradient(Gauss_Img)
    Gauss_Img = non_maximal_suppr(Gauss_Img,dirn)
    Gauss_Img = thresholding(Gauss_Img,low,high,50)
    Gauss_Img = hysteresis(Gauss_Img)
    fig, axes = plt.subplots(1,3,figsize=(16, 8))
    img1 = feature.canny(inputImg)
    psnr_val = psnr(img1,Gauss_Img)
    mini = img1.min()
    rng = (img1.max())- mini
    o1 = (img1 - mini)/rng
    c1 = (Gauss_Img - mini)/rng
    pp = peak_signal_noise_ratio(o1,c1)
    print(pp)
    ssim_val = structural_similarity(img1.astype('bool'),Gauss_Img.astype('bool'))
    axes[0].imshow(img1,cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Edge Detection Using Inbuilt Canny')
    axes[1].imshow(Gauss_Img,cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Edge Detection Using myCannyEdgeDetector')
    axes[2].imshow(img,cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Original Image')
    str1 = 'PSNR Value is: ' + str(psnr_val)
    str1 = str1 + '\nSSIM Value is: ' + str(ssim_val)
    plt.figtext(0.15,0.03,str1)
    plt.show()
    print('PSNR Value is ',psnr_val)
    print('SSIM Value is ',ssim_val)

# Main Function
if __name__ == '__main__':
    inputImg = imread('image/img5.jpg')
    low = 0.02 
    high = 0.06
    myCannyEdgeDetector(inputImg,low,high)