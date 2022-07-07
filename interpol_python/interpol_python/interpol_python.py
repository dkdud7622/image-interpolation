from configparser import Interpolation
import cv2
import numpy as np
import math
from math import log10, sqrt
from timeit import default_timer as timer


def PSNR(original, compare) :
    mse = np.mean((original - compare)**2)
    if (mse == 0) :
        return 100
    maxValue = 255.0
    psnr = 20*log10(maxValue/sqrt(mse))
    return psnr

def nearest (image,scale_factor):
    start = timer()
    rows,cols,C = image.shape
    scaled_height = math.ceil(rows * scale_factor[0])
    scaled_width = math.ceil(cols * scale_factor[1])
    scaled_image = np.zeros((scaled_height, scaled_width,C), np.uint8)

    x_ratio = cols/scaled_width
    y_ratio = rows/scaled_height
    print("Start nearest interpolation.")
    for c in range(C):
        for i in range(scaled_height):
            for j in range(scaled_width):
                x = int(x_ratio * j) #==trunc()
                y = int(y_ratio * i)
                scaled_image[i][j][c] = image[y][x][c]
    print("End nearest")
    end = timer()
    print("Time taken : ",round(end-start,3),'s')
    print()
    return scaled_image.astype('uint8')



def bilinear(image,scale_factor):

    start = timer()
    rows,cols,C = image.shape
    scaled_height = math.ceil(rows * scale_factor[0])
    scaled_width = math.ceil(cols * scale_factor[1])

    scaled_image = np.zeros((scaled_height, scaled_width,C), np.uint8)

    x_ratio = cols / scaled_width;
    y_ratio = rows / scaled_height;
    #print(x_ratio,y_ratio)
    print("Start bilinear interpolation.")
    for c in range(C):
        for i in range(scaled_height): #row(y)
            for j in range(scaled_width): #col(x)

                x = int(x_ratio * j) #==trunc()
                y = int(y_ratio * i)
                dx = (x_ratio * j) - x
                dy = (y_ratio * i) - y
                if x+1 > cols-1 : x -= 1
                if y+1 > rows-1 : y -= 1
                c1 = image[y][x][c] * (1-dy) + image[y+1][x][c] * dy
                c2 = image[y][x+1][c] * (1-dy) + image[y+1][x+1][c] * dy
                #if i == 0 and j == 0 :
                #    print(x,y)
                #    print(x_diff)
                #    print(y_diff)
                #    print(c1)
                #    print(c2)
                scaled_image[i][j][c] = int(c1 * (1-dx) + c2 * dx)

    print("End bilinear")
    end=timer()
    print("Time taken : ",round(end-start,3),'s')
    print()
    return scaled_image.astype('uint8')




def bicubicConvolution(x,a):
    if (abs(x) >=0) & (abs(x) <=1): 
        return (a+2)*(abs(x)**3)-(a+3)*(abs(x)**2)+1
    elif (abs(x) > 1) & (abs(x) < 2): 
        return a*(abs(x)**3)-(5*a)*(abs(x)**2)+(8*a)*abs(x)-4*a
    return 0

def bicubic(image,scale_factor) :
    start = timer()
    rows,cols,C= image.shape
    scaled_height = math.ceil(rows * scale_factor[0])
    scaled_width = math.ceil(cols * scale_factor[1])
    scaled_image = np.zeros((scaled_height, scaled_width,C), np.uint8)

    x_ratio = cols / scaled_width;
    y_ratio = rows / scaled_height;
    a = -3/4
    print("Start bicubic interpolation.")
    for c in range(C):
        for i in range(scaled_height): #row(y)
            for j in range(scaled_width): #col(x)
                x = j * x_ratio + 2
                y = i * y_ratio + 2

                x1 = 1+x-int(x)
                x2 = x-int(x)
                x3 = int(x) +1 -x
                x4 = int(x) +2 -x
                #if i == 0 and j == 0 :
                #    print(x1, x2, x3, x4)
                y1 = 1+y-int(y)
                y2 = y-int(y)
                y3 = int(y) +1 -y
                y4 = int(y) +2 -y
            
                x_1 = int(max((min(x-x1,cols-1)),0))
                x_2 = int(max((min(x-x2,cols-1)),0))
                x_3 = int(max((min(x+x3,cols-1)),0))
                x_4 = int(max((min(x+x4, cols-1)),0))

                y_1 = int(max((min(y-y1,rows-1)),0))
                y_2 = int(max((min(y-y2,rows-1)),0))
                y_3 = int(max((min(y+y3,rows-1)),0))
                y_4 = int(max((min(y+y4, rows-1)),0))

                mat1 = np.matrix([[bicubicConvolution(x1,a),bicubicConvolution(x2,a),bicubicConvolution(x3,a),bicubicConvolution(x4,a)]])
                mat2 = np.matrix([[image[y_1,x_1,c],image[y_2,x_1,c],image[y_3,x_1,c],image[y_4,x_1,c]],
                                   [image[y_1,x_2,c],image[y_2,x_2,c],image[y_3,x_2,c],image[y_4,x_2,c]],
                                   [image[y_1,x_3,c],image[y_2,x_3,c],image[y_3,x_3,c],image[y_4,x_3,c]],
                                   [image[y_1,x_4,c],image[y_2,x_4,c],image[y_3,x_4,c],image[y_4,x_4,c]]])
                mat3 = np.matrix([[bicubicConvolution(y1,a)],[bicubicConvolution(y2,a)],[bicubicConvolution(y3,a)],[bicubicConvolution(y4,a)]])
                scaled_image[i,j,c] = max(min(255,np.dot(np.dot(mat1,mat2),mat3)),0)
    print("End bicubic")
    end = timer()
    print("Time taken : ",round(end-start,3),'s')
    print()
    return scaled_image.astype('uint8')




def save_image(img,name,scale_factor):
    path = name+str(int(scale_factor[0]))+"x"+str(int(scale_factor[1]))+".png"
    cv2.imwrite(path,img)

if __name__ == '__main__':

    img = cv2.imread("Lenna.png")  # gray scale image
    simg = cv2.resize(img,dsize=(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)

    x = float(input("Enter a factor to scale x: "))
    y = float(input("Enter a factor to scale y: "))
    print()

    rate = (x, y)
    uimg=cv2.resize(simg,dsize=(0,0),fx=rate[0],fy=rate[1],interpolation=cv2.INTER_LINEAR)
    print("scaling size is ", uimg.shape)
    print()

    img_nearest_result = nearest(simg, rate)
    img_bilinear_result = bilinear(simg, rate)
    img_bicubic_result = bicubic(simg , rate)

    print("nearest PSNR : " + str(round(PSNR(uimg,img_nearest_result),3)),'dB')
    print("bilinear PSNR : "+str(round(PSNR(uimg,img_bilinear_result),3)),'dB')
    print("bicubic PSNR : "+str(round(PSNR(uimg,img_bicubic_result),3)),'dB')
    print()
    cv2.imshow('origin',img);
    cv2.imshow('small_origin',simg);
    cv2.imshow('nearest_function',img_nearest_result);
    cv2.imshow('bilinear_function',img_bilinear_result);
    cv2.imshow('bicubic_function',img_bicubic_result);
    
    save_image(img_nearest_result,'nearest',rate)
    save_image(img_bilinear_result,'bilinear',rate)
    save_image(img_bicubic_result,'nearest',rate)

    #useLibrary=cv2.resize(img,dsize=(512,512),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('library',useLibrary)
    print('Please press any key to exit....')
    cv2.waitKey()

