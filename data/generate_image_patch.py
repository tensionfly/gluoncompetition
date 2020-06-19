import os
import math
from osgeo import gdal,gdal_array
from gdalconst import GA_ReadOnly
from tqdm import tqdm

def generate_patch(input_dir:str,output_dir:str,patch_width:int,patch_height:int,x_off:int,y_off:int):
    """
    input_dir: input image directory
    output_dir: output image directory
    patch_width: the width of the cliping image block (pixle)
    patch_height: the height of the cliping image block (pixle)
    x_off: the overlap on image x axis (pixle)
    y_off: the overlap on image y axis (pixle)
    """

    img_list=os.listdir(input_dir)

    for img_name in tqdm(img_list,ascii=True):
        img_path=input_dir+img_name
        name_not_tif=img_name.split('.')[0]

        img_array = gdal_array.DatasetReadAsArray(
            gdal.Open(img_path, GA_ReadOnly))
        
        dimension_img_array=len(img_array.shape)

        if dimension_img_array==3:
            (_,height, width) = img_array.shape
        elif dimension_img_array==2:
            (height, width) = img_array.shape
        else:
            print('The dimension of image is not right')
            break

        x_num=math.ceil((width-patch_width)/(patch_width-x_off))+1
        y_num=math.ceil((height-patch_height)/(patch_height-y_off))+1

        k=0
        for i in range(y_num):
            if i==y_num-1:
                s_y=height-patch_height
            else:
                s_y=i*(patch_height-y_off)

            for j in range(x_num):
                if j==x_num-1:
                    s_x=width-patch_width
                else:
                    s_x=j*(patch_width-x_off)
                
                if dimension_img_array==3:
                    img_clip=img_array[:,s_y:s_y+patch_height,s_x:s_x+patch_width]
                elif dimension_img_array==2:
                    img_clip=img_array[s_y:s_y+patch_height,s_x:s_x+patch_width]

                filename=output_dir+name_not_tif+'_'+str(k)+'.tif'
                
                driver = gdal.GetDriverByName("GTiff")
                driver.CreateCopy(filename, gdal_array.OpenArray(img_clip, None))#, options=["COMPRESS=LZW", "PREDICTOR=2"])
                
                k+=1
