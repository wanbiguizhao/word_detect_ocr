
# %%
from collections import defaultdict
from functools import reduce
from sklearn.model_selection import train_test_split
import os
import re 
PROJECT_DIR= os.path.dirname(
                        os.path.dirname(os.path.realpath( __file__))
                            )
DATASET_DIR=os.path.join(PROJECT_DIR,"tmp/dataset02")

import glob
import paddle.vision.transforms as transforms
import cv2 as cv 
def explain_labels(clean_row,labels_path):
    """
    对
    35+*
    35++
    35-40*
    35+3
    这一类的数据进行解析
    O(n)时间复杂度

    """
    import_flag=False 
    # 写一个状态机吧
    num_list=[]
    find_num_flag=False
    current_num=-1
    # 要求使用两个堆栈
    # 一个是数字堆栈
    # 一个是符号堆栈
    num_stack=[]
    flag_stack=[]
    for x in clean_row:
        if x.isdigit():
            if not find_num_flag:
                find_num_flag=True
                current_num=int(x) 
            else:
                current_num=current_num*10+int(x)
        else:
            if find_num_flag:
                num_stack.append(current_num)
                find_num_flag=False
            if   x not in "*+-":
                print(labels_path,'->',clean_row,"Data Format Error ")
                raise Exception()
            flag_stack.append(x)
    if find_num_flag:
        num_stack.append(current_num)
    assert len(num_stack)!=0
    # 开始对数据进行处理
    while len(flag_stack)>0:
        if flag_stack[-1]=="*":
            import_flag=True
            flag_stack.pop(-1)
        elif flag_stack[-1]=="+":
            plus_count=0
            while flag_stack and flag_stack[-1]=="+":
                flag_stack.pop(-1)
                plus_count+=1
            # 此时flag_stack必须清空了
            assert len(flag_stack)==0 
            if len(num_stack)==1:
                # 35++++ 这种情况
                num_list= list(range(num_stack[0],num_stack[0]+plus_count+1))
            elif len(num_stack)==2:
                # 35+3 的情况
                # 不能出现 35++3的情况
                assert plus_count==1
                num_list=list(range(num_stack[0],num_stack[1]+num_stack[0]+1 ))
            else:
                # 数字太多了
                print(labels_path,'->',clean_row,"Data Format Error ")
                raise Exception()
        elif flag_stack[-1]=="-":
            assert len(num_stack)==2 and len(flag_stack)==1
            flag_stack.pop(-1)
            num_list=list(range(num_stack[0],num_stack[1]+1))
        else:
            print(labels_path,'->',clean_row,"Data Format Error ")
            raise Exception()
    # 没有出现过特殊符号的情况
    if len(flag_stack)==0 and len(num_list)==0:
        num_list=num_stack
    #print(clean_row,"\t",list(zip(num_list,[import_flag]*len(num_list))))
    return num_list,import_flag
def do_image_name_index_check(dataset_image_list):
    """
    对数据进行检查，确保图片的名称和索引可以一一对应上，
    例如：word_seg_00001_type_05.png  
        00001 
        代表dataset_image_list[1]=="word_seg_00001_type_05.png"
    """
    for index, image_info in enumerate(dataset_image_list):
        matchObj = re.match(
        r'.*word.*_(?P<id_ds>\d+)_.*_(?P<id_cluster>\d+)', image_info)
        assert matchObj
        id_ds=matchObj.groupdict()["id_ds"]
        assert index==int(id_ds)
def load_image_labels_info(dataset_dir):
    """
    因为采用小批量标注的原因，每次标注的结果，放在一个文件夹下面。
    返回，每个标注文件的位置，
        图片类型：WORD_TYPE 图片属于汉字的一部分 SPACE_TYPE 图片可以用来切割两个汉字
        重要性：标注该图片属于两个汉字挨着特别紧密的情况
    .
    ├── wis_01
        |---labels.txt
    ├── wis_02
    """
    import_data_index=[]# 
    WORD_TYPE=0# 表示图像是汉字的一部分
    SPACE_TYPE=1# 表示图像是两个汉字中间间隔
    SPACE_TYPE_LIST=[]
    DATASET=[]
    for dir_path in os.listdir(dataset_dir):
        ds_image_dir=os.path.join(dataset_dir,dir_path)
        #print(ds_image_dir)
        dataset_image_list=sorted(glob.glob(os.path.join(ds_image_dir,"word*.png")))
        do_image_name_index_check(dataset_image_list)
        one_dir_data=[{ "Image_Path":image_path,
                        #"Image":cv.imread(image_path,cv.IMREAD_GRAYSCALE) ,
                        "Image_Type":WORD_TYPE,
                        "Import_Flag": False } for image_path in dataset_image_list]
        labels_path=os.path.join(dataset_dir,dir_path,'labels.txt')
        if not os.path.exists(labels_path) or not os.path.isfile(labels_path):
            # 文件不存在
            assert False 
        with open(labels_path,'r') as lab_file:
            # 找到目录下对应的
            for rowdata in lab_file.readlines():
                #print(rowdata)
                clean_row=rowdata.strip("\n").strip("\t")
                if not clean_row:
                    continue 
                num_list,import_flag=explain_labels(clean_row,labels_path)
                for num_index in num_list:
                    one_dir_data[num_index]["Image_Type"]=SPACE_TYPE
                    one_dir_data[num_index]["Import_Flag"]=import_flag
        DATASET.extend(one_dir_data)
    return DATASET
            # 返回的应该是[dataset_dir下的路径，标签]

def pipline_data_mlp(dataset_dir,expansion=2,test_size=0.2):
    """
    专门为神经网络的图片使用，
    expansion，表示对于标记为import_flag的图片多复制几次。
    expansion=1，2，3，4，5，6 表示复制几次
    """
    def merge_data(x,y):
        # 数据结构变化，原来的数据结构是：[{},{},{}]现在变化为{key:[],key:[],}
        if x is None:
            x=defaultdict(list)
        t=1 if y["Import_Flag"] else expansion
        while t<=expansion:
            # 执行多次复制功能
            for key,val in y.items():
                x[key].append(val)
            t+=1
        return x 
    
    labels_image_info=load_image_labels_info(dataset_dir) 
    train_data,test_data=train_test_split(labels_image_info,test_size=test_size)
    train_labels=reduce(merge_data,[None]+train_data)# 执行了多次复制功能
    test_labels=reduce(merge_data,[None]+test_data)# 执行了多次复制功能
    # 加载图片
    #print(type(new_labels_image_info), sum(new_labels_image_info["Image_Type"]), [ [key,len(val)] for key,val in new_labels_image_info.items()])
    #train_data ,test_data =train_test_split(labels_image_info,test_size=0.2)
    return train_labels,test_labels

def pipline_data_gru(dataset_dir,size=80,test_size=0.2):
    """
    专门为gru的图片使用数据集
    train_test_split必须重新调整，
    
    """
    def merge_data(x,y):
        # 数据结构变化，原来的数据结构是：[{},{},{}]现在变化为{key:[],key:[],}
        if x is None:
            x=defaultdict(list)
        for key,val in y.items():
                x[key].append(val)
        return x 
    labels_image_info=load_image_labels_info(dataset_dir)#一个长度为80个标签，然后长度为40往前分标签，然后再做区分。
    gru_labels_image_label=[]
    # 这里安装硬代码进行安装
    beg_index=0
    end_index=80
    step=40
    while end_index<len(labels_image_info):
        gru_labels_image_label.append(
            labels_image_info[beg_index:end_index]
        )
        beg_index+=step
        end_index+=step
    train_data,test_data=train_test_split(gru_labels_image_label,test_size=test_size)# 数据先分配了一下
    train_labels=defaultdict(list)
    for td in train_data:
        # td 的数结构是[{key:v,key,:v}]做一次数据转换
        for k , v in reduce(merge_data,[None]+td).items():
            train_labels[k].append(v)
        
    test_labels=defaultdict(list)
    for td in test_data:
        # td 的数结构是[{key:v,key,:v}]做一次数据转换
        for k , v in reduce(merge_data,[None]+td).items():
            test_labels[k].append(v)


    # 加载图片
    #print(type(new_labels_image_info), sum(new_labels_image_info["Image_Type"]), [ [key,len(val)] for key,val in new_labels_image_info.items()])
    #train_data ,test_data =train_test_split(labels_image_info,test_size=0.2)
    return train_labels,test_labels
def pipline_infer_data_mlp(dataset_dir):
    pass

#%%
#load_image_labels_info(DATASET_DIR)    

class DataCooker:
    """
    对数据进行加工,以后有时间再重构代码
    """
    def __init__(self,data_dir,expansion=2,test_size=0.2) -> None:

        pass

if __name__ == "__main__":
    #pipline_data_mlp(DATASET_DIR,expansion=3)
    pipline_data_gru(DATASET_DIR)
# %%
