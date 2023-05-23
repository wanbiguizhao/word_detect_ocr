import os
import time
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2 as cv 
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
def splitImage(image_bin):
    """
    将图片切割成16*48的小片段，步长为4
    """
    # 第一步假设高度已经填充为48吧，宽度是4的整数倍
    step=4
    h,w=image_bin.shape
    beg_index=0
    end_index=15
    location_list=[]
    while end_index<w:
        location_list.append((beg_index,end_index))
        beg_index+=step 
        end_index+=step
    return location_list
def imageStrip(image_bin):
    # 切除行中两边白色的地方
    h, w = image_bin.shape
    # 垂直投影，获得每个字的边界
    vprojection = np.zeros(image_bin.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        #w_w[i]=sum(image_bin[:,i])
        for j in range(h):
            if image_bin[j, i ] == 0:
                w_w[i] += 1
    beg_index=0
    while beg_index<len(w_w):
        if w_w[beg_index]<=5:
            beg_index+=1
        else:
            break
    end_index=w -1
    while end_index>beg_index:
        if w_w[end_index]<=5:
            end_index-=1
        else:
            break
    return max(beg_index-16,0),min(end_index+16,w-1)
def pickle_data_proc_image(image_path):
    #for image_path in  image_path_list :
    #print(image_path)
    image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    # 切掉白色的两边
    blur = cv.GaussianBlur(image,(5,5),0)
    ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    h,w=th_image.shape
    if h>=48:
        return [] 
    beg_index,end_index=imageStrip(th_image)
    w=end_index-beg_index+1
    h_padding=48-h
    w_padding=(4-w%4)%4
    top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
    left,right=w_padding//2,w_padding-(w_padding//2)
    new_image = cv.copyMakeBorder(image[:,beg_index:end_index+1], top, bottom, left, right, cv.BORDER_CONSTANT, value=(255,))
    return[  new_image[: ,beg:end+1 ] for beg,end in splitImage(new_image) ]
def pickle_data(data_dir,num_cpus=1):
    data_list = []


    if num_cpus==1:
        for image_path in tqdm( glob(os.path.join(data_dir,"*","*.png"),recursive=True)):
            data_list.extend(pickle_data_proc_image(image_path))
    else:
        with Pool(nodes=num_cpus) as pool:
            for image_data_list in  list(tqdm(pool.imap(pickle_data_proc_image,glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:5000]) )):
                data_list.extend(image_data_list)
    with open("tmp/constract_image_pice_5000.pkl",'wb') as imagePiceData:       
        pickle.dump(data_list,imagePiceData)


def show_word_pice(offest=1000):
    with open("tmp/constract_image_pice.pkl",'rb') as imagePiceData:
        data=pickle.load(imagePiceData)
        print(type(data))
        from matplotlib import pyplot as plt
        print(len(data))
        for x in range(32):
            plt.subplot(1,32,x+1)
            plt.imshow(data[x+offest])
        plt.show()
        time.sleep(10)
def pickle_data_wip_proc_image(image_path):
    image_info={}
    image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    # 切掉白色的两边
    blur = cv.GaussianBlur(image,(5,5),0)
    ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    h,w=th_image.shape
    if h>=48:
        return None
    beg_index,end_index=imageStrip(th_image)
    w=end_index-beg_index+1
    h_padding=48-h
    w_padding=(4-w%4)%4
    top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
    left,right=w_padding//2,w_padding-(w_padding//2)
    new_image = cv.copyMakeBorder(image[:,beg_index:end_index+1], top, bottom, left, right,cv.BORDER_CONSTANT, value=(255,))
    image_info["image_path"]=image_path
    image_info["image"]=new_image
    image_info["crop_info"]=[beg_index,end_index]
    image_info["seg_index"]=[]
    # 记录一下图片，然后记录一下，图片切割后的图片例子，
    # new_image_list=[] # 存储一下切割后的
    # image_mapping={}, key 表示的切割的第i个元素， value： { image_index:{这个元素在图片中的索引位置},beg_index:int,end_index:int} 的数据
    for beg,end in splitImage(new_image):
        image_info["seg_index"].append(
            [beg,end]
        )
    return image_info
def pickle_data_wip(data_dir,num_cpus=1):
    # pickle 一种新的数据结构
    data_list = []

    image_info={
            "image_path":[],
            "image":[],
            "crop_info":[]
        }
    if num_cpus==1:
        for image_path in tqdm( glob(os.path.join(data_dir,"*","*.png"),recursive=True)):

            seg_info=pickle_data_wip_proc_image(image_path)
            if not seg_info:
                continue
            image_info["image"].append(seg_info["image"])
            image_info["image_path"].append(seg_info["image_path"])
            image_info["crop_info"].append(seg_info["crop_info"])
            for beg,end in image_info["seg_index"]:
                data_list.append(
                    {
                        "image_index":len(image_info["image"])-1,
                        "seg_beg_index":beg,
                        "seg_end_index":end
                    }
                )

    else:
        with Pool(nodes=num_cpus) as pool:
            for seg_info in  list(tqdm(pool.imap(pickle_data_wip_proc_image,glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:1000]) )):
                if not seg_info:
                    continue
                image_info["image"].append(seg_info["image"])
                image_info["image_path"].append(seg_info["image_path"])
                image_info["crop_info"].append(seg_info["crop_info"])
                for beg,end in seg_info["seg_index"]:
                    data_list.append(
                        {
                            "image_index":len(image_info["image"])-1,
                            "seg_beg_index":beg,
                            "seg_end_index":end
                        }
                    )
                
    with open("tmp/constract_wip_all.pkl",'wb') as imagePiceData:       
        pickle.dump({"data_list":data_list,"image_info":image_info},imagePiceData)
if __name__=="__main__":
    # from matplotlib import pyplot as plt
    # ds=WordImagePiceDataset(data_dir="tmp/project_ocrSentences")
    # print(len(ds))
    # plt.figure(32)
    # for x in range(32):
    #     plt.subplot(1,32,x+1)
    #     plt.imshow(ds[x][0])
    # plt.show()
    # time.sleep(10) 
    #pickle_data("tmp/project_ocrSentences",8)
    pickle_data_wip("tmp/project_ocrSentences",12)
    #show_word_pice(offest=10000)
