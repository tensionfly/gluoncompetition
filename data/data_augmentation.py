import cv2
import numpy as np
import random

import mxnet as mx
from mxnet.gluon.data.vision import transforms


def ColorNormalize(img:np.array,mean:np.array,std:np.array):
    """
    img : h,w,c
    mean: 3,
    std: 3,
    """
    mean=mean.reshape((1,1,3))
    std=std.reshape((1,1,3))
    img_normalize=(img-mean)*(1.0/std)

    return img_normalize

def ImageRotate(image:np.array, angle, center=None, scale=1.0,keep_context=False):
    """
    image: np.array h,w,c
    angle: 0-360
    center:(x,y)
    scale: default 1.0
    keep_context:try to keep context by using scaling

    return rotated image: np.array h,w,c
    """

    (h, w) = image.shape[:2] 
    if center is None: 
        center = (w // 2, h // 2) 
    
    if keep_context:
        cx=center[0]
        cy=center[1]
        rad=math.radians(angle)
        rad_cos=math.cos(rad)
        rad_sin=math.sin(rad)

        corner_points=np.array([[-cx,w-cx,w-cx,-cx],
                                [-cy,-cy,h-cy,h-cy]])
        rotate_matrix=np.array([[rad_cos,rad_sin],[-rad_sin,rad_cos]])

        corner_points_rotated=np.dot(rotate_matrix,corner_points)
        xy_max=corner_points_rotated.max(axis=1)
        xy_min=corner_points_rotated.min(axis=1)

        ratio=((xy_max-xy_min)*np.array([1.0/w,1.0/h])).max()
        scale=1.0/ratio

    M = cv2.getRotationMatrix2D(center, angle, scale) 
    rotated = cv2.warpAffine(image, M, (w, h),borderValue=(0,0,0)) 

    return rotated

def ImageRandomFlipRotate(data:np.array): 
    """
    data: np.array h,w,c

    return data: np.array h,w,c
    """
  
    if np.random.uniform() > 0.5:
        data = np.flip(data, axis=1)
        # print('h')

    if np.random.uniform() > 0.5:
        data = np.flip(data, axis=0)
        # print('v')
    
    if np.random.uniform() > 0.5:
        data=np.rot90(data,1,(0,1))
        # print('r')

    # if np.random.uniform() > 0.5:
    #     size_guass=random.randint(3,10)
    #     x=random.uniform(0,3)
    #     if size_guass%2==0:
    #         size_guass+=1
    #     data=cv2.GaussianBlur(data,(size_guass,size_guass),x)
        # print("blu")

    return data

def GluonTransformation(data:mx.nd.array):
    """
    data: mx.nd.array h,w,c

    retrun data: mx.nd.array
    """

    data=mx.nd.array(data)
    transform= transforms.Compose([
                                transforms.RandomResizedCrop(200,(0.8,1.0)),
                                transforms.CenterCrop((300,300)),

                                transforms.RandomFlipLeftRight(),
                                transforms.RandomFlipTopBottom(),

                                transforms.RandomLighting(0.3),
                                transforms.RandomColorJitter(brightness=0.1, contrast=0.1, 
                                                            saturation=0.1, hue=0.2),

                                transforms.Resize(384),
                                transforms.ToTensor(),# h,w,c -> c, h, w
                                transforms.Normalize(0, 1)
                                ])
    data=transform(data)
    return data

# if __name__=='__main__':

#     img=cv2.imread('1.jpg')
#     img_out=ImageRotate(img,30)

#     # img_out=transformation(img)
#     cv2.imshow('ori',img)
#     cv2.imshow('rotate',img_out)
#     cv2.waitKey(0)
#     # cv2.imshow('img',mx.nd.clip(img_out,0,255).asnumpy().astype(np.uint8))
#     # cv2.imshow('img',img_out.asnumpy().astype(np.uint8))

#     # cv2.waitKey(0)
#     print('done!')