import os
import time
import numpy as np
from paddle.io import Dataset
from glob import glob
from tqdm import tqdm
import cv2 as cv 
from pathos.multiprocessing import ProcessingPool as Pool


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

class WordImagePiceDataset(Dataset):
    """
    数据集，可以自行将 图片切割成 16*48的数据集
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_dir, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WordImagePiceDataset, self).__init__()
        self.data_list = []
        #for image_path in tqdm(glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:20]):
        # 需要重新设计一下数据结构。
        for image_path in tqdm(glob(os.path.join(data_dir+"*.png"),recursive=True)[:20]):
            image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
            # 切掉白色的两边
            blur = cv.GaussianBlur(image,(5,5),0)
            ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            h,w=th_image.shape
            if h>=48:
                continue 
            beg_index,end_index=imageStrip(th_image)
            w=end_index-beg_index+1
            h_padding=48-h
            w_padding=(4-w%4)%4
            top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
            left,right=w_padding//2,w_padding-(w_padding//2)
            new_image = cv.copyMakeBorder(image[:,beg_index:end_index+1], top, bottom, left, right,cv.BORDER_CONSTANT, value=(255,))
            # 记录一下图片，然后记录一下，图片切割后的图片例子，
            # new_image_list=[] # 存储一下切割后的
            # image_mapping={}, key 表示的切割的第i个元素， value： { image_index:{这个元素在图片中的索引位置},beg_index:int,end_index:int} 的数据
            self.data_list.extend([  new_image[: ,beg:end+1 ] for beg,end in splitImage(new_image) ])
            #print(image_dir_path,image_pure_name,extension)
            #填充到48的整数倍 
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image = self.data_list[index]
        image=image.astype('float32')
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return image, 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

class WIPDataset(Dataset):
    """
    数据集，可以自行将 图片切割成 16*48的数据集
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_dir, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WIPDataset, self).__init__()
        self.data_list = []
        self.image_info={
            "image_path":[],
            "image":[],
            "origin_image":[],
            "crop_info":[]
        }
        self.half_width_padding=8# 为了保证图片可以被窗口扫描到，在图片左右两侧进行填充，超参数。
        #for image_path in tqdm(glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:20]):
        # 需要重新设计一下数据结构。
        for image_path in sorted(tqdm(glob(os.path.join(data_dir+"*.png"),recursive=False))):

            image=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
            
            h,w=image.shape
            if h!=80:
                resize_hh=64
                resize_image=cv.resize(image,(w,resize_hh)) #.resize(64,w)# 强制变化为高度是64的情况。
            else:
                resize_image=image
            # 切掉白色的两边
            blur = cv.GaussianBlur(resize_image,(5,5),0)

            ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            # 这块要做一次变化，等比例变成64的高度。
            
            #beg_index,end_index=imageStrip(th_image)
            #w=end_index-beg_index+1
            #h_padding=48-h
            #w_padding=(4-w%4)%4
            #top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
            #left,right=w_padding//2,w_padding-(w_padding//2)
            
            new_image = cv.copyMakeBorder(th_image, 0, 0, self.half_width_padding, self.half_width_padding,cv.BORDER_CONSTANT, value=(255,))# 给图片前后都加了8白色，方便可以窗口扫描到。
            self.image_info["image_path"].append(image_path)
            self.image_info["origin_image"].append(image)
            self.image_info["image"].append(new_image)
            #self.image_info["crop_info"].append([beg_index,end_index])
            # 记录一下图片，然后记录一下，图片切割后的图片例子，
            # new_image_list=[] # 存储一下切割后的
            # image_mapping={}, key 表示的切割的第i个元素， value： { image_index:{这个元素在图片中的索引位置},beg_index:int,end_index:int} 的数据
            for beg,end in splitImage(new_image):
                self.data_list.append(
                    {
                        "image_index":len(self.image_info["image"])-1,# 记录dataset的index使用的图片是哪个
                        "seg_beg_index":beg,
                        "seg_end_index":end
                    }
                )

            #self.data_list.extend([  new_image[: ,beg:end+1 ] for beg,end in splitImage(new_image) ])
            #print(image_dir_path,image_pure_name,extension)
            #填充到48的整数倍 
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        seg_image_info = self.data_list[index]
        seg_beg_index=seg_image_info["seg_beg_index"]
        seg_end_index=seg_image_info["seg_end_index"]
        image_index=seg_image_info["image_index"]
        image=self.image_info["image"][image_index]
        seg_image=image[:,seg_beg_index:seg_end_index+1]
        float_seg_image=seg_image.astype('float32')
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        if self.transform is not None:
            image_0,image_1 = self.transform(float_seg_image)
            return [image_0,image_1,seg_image], 0
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return [seg_image,seg_image,seg_image], 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

class WIPObjDataset(Dataset):
    """
    数据集，可以自行将 图片切割成 16*48的数据集
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_path, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WIPObjDataset, self).__init__()
        self.data_list = []
        self.image_info={
            "image_path":[],
            "image":[],
            "crop_info":[]
        }
        with open(data_path,'rb') as objfile:
            data=pickle.load(objfile)
            self.data_list=data["data_list"]
            self.image_info=data["image_info"]

        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        seg_image_info = self.data_list[index]
        seg_beg_index=seg_image_info["seg_beg_index"]
        seg_end_index=seg_image_info["seg_end_index"]
        image_index=seg_image_info["image_index"]
        image=self.image_info["image"][image_index]
        seg_image=image[:,seg_beg_index:seg_end_index+1]
        float_seg_image=seg_image.astype('float32')
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        if self.transform is not None:
            image_0,image_1 = self.transform(float_seg_image)
            return [image_0,image_1,seg_image], 0
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return [seg_image,seg_image,seg_image], 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

class WIPByteDataset(Dataset):
    """
    从二进制图片中生成数据集
    """
    def __init__(self, image_byte, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WIPByteDataset, self).__init__()
        self.data_list = []
        self.image_info={
            "image_path":[],
            "image":[],
            "crop_info":[]
        }
        
        #for image_path in tqdm(glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:20]):
        # 需要重新设计一下数据结构。
        #nparr = np.from(img_str, np.uint8)
        
        
        image = cv.imdecode(np.frombuffer(image_byte,np.uint8), cv.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR in OpenCV 3.1
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        h,w=th_image.shape
        if h>48: # 之后再处理吧
            # 直接硬做裁剪处理
            reduce_height=h-48
            reduce_up_height=reduce_height//2
            reduce_down_height=reduce_height-reduce_height//2
            th_image=th_image[reduce_up_height:-reduce_down_height,:]
            h,w=th_image.shape
        h_padding=48-h
        w_padding=(4-w%4)%4
        top, bottom = h_padding//2, h_padding-(h_padding//2)# 上下部分填充
        #left,right=w_padding//2,w_padding-(w_padding//2) # 左右部分填充
        new_image = cv.copyMakeBorder(image, top, bottom, 0, 0,cv.BORDER_CONSTANT, value=(255,))
        self.new_image=new_image
        self.origin_image=image
        
        for beg,end in splitImage(new_image):
            self.data_list.append(
                {
                    "seg_beg_index":beg,
                    "seg_end_index":end
                }
            )
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        seg_image_info = self.data_list[index]
        seg_beg_index=seg_image_info["seg_beg_index"]
        seg_end_index=seg_image_info["seg_end_index"]

        seg_image=self.new_image[:,seg_beg_index:seg_end_index+1]
        float_seg_image=seg_image.astype('float32')# 变成tensor了。
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        if self.transform is not None:
            image_0,image_1 = self.transform(float_seg_image)
            return [image_0,image_1,seg_image], 0
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return [seg_image,seg_image,seg_image], 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)




class WordImagePiceDatasetOBJ(Dataset):
    """
    直接使用pkl的数据
    """
    def __init__(self, data_path, label_path=None, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(WordImagePiceDatasetOBJ, self).__init__()
        self.data_list = []
        self.image_info={
            "image_path":[],
            "image":[],
            "crop_info":[]
        }
        with open(data_path,'rb') as objfile:
            self.data_list=pickle.load(objfile)
            # self.data_list=data["data_list"]
            # self.image_info=data["image_info"]
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image = self.data_list[index]
        image=image.astype('float32')
        # 读取灰度图
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # image = image.astype('float32')
        # # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        # #label = int(label)
        # 返回图像和对应标签
        return image, 0

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

import pickle
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

        # data_list=proc_image(glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:])
    else:
        #path_list=list()
        with Pool(nodes=num_cpus) as pool:
            for image_data_list in pool.map(pickle_data_proc_image,glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:100]):
                data_list.extend(image_data_list)
        # data_list=list(
        #             tqdm(
        #                     pool.imap(proc_image,glob(os.path.join(data_dir,"*","*.png"),recursive=True)[:100])
        #                 )
        #             )
        
    with open("tmp/constract_image_pice.pkl",'wb') as imagePiceData:
        #np.save(imagePiceData,data_list)        
        pickle.dump(data_list,imagePiceData)


def show_word_pice():
    with open("tmp/constract_image_pice.pkl",'rb') as imagePiceData:
        data=pickle.load(imagePiceData)
        print(type(data))
        from matplotlib import pyplot as plt
        print(len(data))
        for x in range(32):
            plt.subplot(1,32,x+1)
            plt.imshow(data[x+200])
        plt.show()
        time.sleep(10)
def show_word_pice_dataset():
    #wip=WordImagePiceDatasetOBJ("tmp/constract_image_pice.pkl")
    wip=WIPObjDataset("tmp/constract_wip_all.pkl")
    from matplotlib import pyplot as plt
    for x in range(32):
        plt.subplot(1,32,x+1)
        plt.imshow(wip[x+200][0][0])
    plt.show()
def show_image_byte_dataset():
    #wip=WordImagePiceDatasetOBJ("tmp/constract_image_pice.pkl")
    wip=WIPByteDataset(open("tmp/project_ocrSentences/1954-01/1954-01_03_007.png","rb").read())
    from matplotlib import pyplot as plt
    for x in range(32):
        plt.subplot(1,32,x+1)
        plt.imshow(wip[x+200][0][0])
    plt.show()
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
    #pickle_data("tmp/project_ocrSentences",2)
    # show_word_pice()
    show_image_byte_dataset()
    print("10")