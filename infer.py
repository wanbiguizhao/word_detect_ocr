
import argparse
import shutil
import paddle 
from models.resnetmodels import HackResNet
from data.dataset import WIPByteDataset
from network import WordImageSliceMLPCLS
from train import get_dataloader
from paddle.metric import accuracy
from data.dataset import WIPDataset
from tools.loader import image_transform
from tools.render import render_html
from paddle.io import DataLoader 
from PIL import Image
import glob, os
parser = argparse.ArgumentParser(description="推测一个神经网络")
parser.add_argument("--data",type=str,default="mocov1/dataset", metavar="DIR", help="path to dataset,指向按行切割的图片的文件夹目录")
parser.add_argument(
    "--expansion",
    default=1,
    type=int,
    metavar="EXPAN",
    help="对于连着的图片的数据，复制倍数",
)
parser.add_argument(
    "--test-size",
    default=0.01,
    type=float,
    metavar="TS",
    help="test size of all data ",
    #dest="lr",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="加载dataset的work",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help=" 这个意思莫非是，多卡GPU的情况，256被多卡平均使用，想多了，用不到多卡",
)
parser.add_argument("--cpu", action="store_true", help="使用cpu训练")# paddle的学习率使用策略和pytorch不一样


def load_model():
    # 这块先进行硬编码把
    encoder_k_model=HackResNet(num_classes=128)
    encoder_q_model=HackResNet(num_classes=128)
    # encoder_k_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_k_model.pdparams"))
    # encoder_q_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_q_model.pdparams"))
    encoder_k_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_020_encoder_k_model.pdparams"))
    encoder_q_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_020_encoder_k_model.pdparams"))
    cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_k_model,encoder_model_q=encoder_q_model,freeze_flag=True)
    cls_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_020_model.pdparams"))
    return cls_model

def test_infer(args):
    cls_model=load_model()
    cls_model.eval()
    train_loader,test_loader=get_dataloader(dataset_dir=args.data,expansion=1,args=args)
    with paddle.no_grad():
        sumacc=0
        for bid, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
            output=cls_model(batch_image[0])
            acc = accuracy(output, batch_image_type.unsqueeze(1))
            sumacc+=float(acc)
            avgacc=sumacc/(bid+1)
            print(avgacc)



def batch_image_infer():
    # 使用模型对对图片进行预测
    # 在图片上画出单个汉字。
    cls_model=load_model()
    #cls_model.eval()
    #cls_model
    BASIC_IMAGE_DIR="tmp/infer"
    for image_dir in os.listdir(BASIC_IMAGE_DIR):
        with paddle.no_grad():
            fast_infer(cls_model,f"{BASIC_IMAGE_DIR}/{image_dir}")




def fast_infer(cls_model,image_data_dir):
    #注意要解析的图片文件名以s开头
    model_ds=WIPDataset(data_dir=image_data_dir+"/s",transform=image_transform())#这个是模型使用，要对数据做一些变化。
    train_loader = DataLoader(
            model_ds,
            batch_size=256,
            shuffle=False,
            num_workers=1,
            #pin_memory=True, paddle 没有过
            #sampler=None,
            drop_last=False,
        )
    import glob, os
    for f in glob.glob(f"{image_data_dir}/word_s*.png"):
        os.remove(f)
    image_index=0
    predict_index_list=[]#记录预测为1的image_index,方便修改进行处理。
    for k, (images, _) in enumerate(train_loader):    
        predict_info=cls_model(images[0])
        predict_labels=paddle.argmax(predict_info,axis=-1)
        seg_img_numpy=images[2].numpy()# 这个图片是原始的图片
        img_len=seg_img_numpy.shape[0]
        i=0
        while i< img_len:
            # 
            # axes=plt.subplot(4,24,j)
            # axes.set_title(str(predict_info[i])+"->"+str(i))
            seg_img=seg_img_numpy[i,:,:]
            h,w=seg_img.shape
            seg_img[:,w//2]=0
            #plt.imshow(seg_img )
            pil_image=Image.fromarray(seg_img)
            predict_label=int(predict_labels[i])
            pil_image.save("{}/word_seg_{:04d}_type_{:02d}.png".format(image_data_dir,image_index,predict_label))
            if predict_label==1:
                predict_index_list.append(image_index)
            image_index+=1
            i+=1
    with open(f"{image_data_dir}/predict_labels.txt","w") as predict_labels_file:
        for x in predict_index_list:
            predict_labels_file.write(f"{str(x)}\n")
    print(image_data_dir,len(predict_index_list))
    render_html(f"{image_data_dir}")

def test_fast_infer():
    # 把要预测的图片放到 tmp/project_ocrSentences_dataset/中 
    #自动给出预测的预测结果，切图
    cls_model=load_model()
    cls_model.eval()
    fast_infer(cls_model,image_data_dir="tmp/project_ocrSentences_dataset/word_image_slice")




def image_byte_infer():
    from paddle.vision import transforms
    from paddle.io import DataLoader 
    from  mocov1.pp_infer import WIPDataset
    from mocov1.moco.loader import TwoCropsTransform
    from PIL import Image
    from mocov1.render import render_html
    normalize = transforms.Normalize(
            mean=[0.485], std=[0.229]
        )
        # 咱们就先弄mocov1的数据增强
    augmentation = [
            #transforms.RandomResizedCrop((16,48), scale=(0.2, 1.0)),
            #transforms.RandomGrayscale(p=0.2), 啥也别说了，paddle没有这个功能
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    wip=WIPByteDataset(open("tmp/project_ocrSentences/1954-01/1954-01_03_007.png","rb").read(),transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = DataLoader(
            wip,
            batch_size=256,
            shuffle=False,
            num_workers=1,
            #pin_memory=True, paddle 没有过
            #sampler=None,
            drop_last=False,
        )
    cls_model=load_model()
    result=[]
    for k, (images, _) in enumerate(train_loader):  
        print(type(images[0]))  
        predict_info=cls_model(images[0])

        predict_labels=paddle.argmax(predict_info,axis=-1)# 预测的每个图片切片的类型。
        #predict_labels 根据这个重新图片的切割的位置。
        for index in range(len(predict_labels)):
            wip_index=index+k*256
            seg_image_info = wip.data_list[wip_index]
            seg_beg_index=seg_image_info["seg_beg_index"]
            seg_end_index=seg_image_info["seg_end_index"]
            if int(predict_labels[index])==1:
                mid_index=(seg_beg_index+seg_end_index)//2
                result.append([wip_index,mid_index])
                wip.origin_image[:,mid_index]=0
                # 需要提供一个坐标，用来标记汉字的位置。
            
      
    pil_image=Image.fromarray(wip.origin_image)
    pil_image.show()
    
def main():
    args = parser.parse_args()
    test_infer(args) 
    #cls_model=load_model()


def infer_single_image(image_byte):
    """
    输入一张图片：
    输出：基于图片的输出分割线。
    """
    image_byte_infer() 

def copy_image():
    # 胶水代码，之后可以删除。
    # 从ocr_data 把png图片存放到指定目录，进行预测工作。
    src_dir="tmp/ocr_data"
    dst_dir="tmp/infer"
    for num_dir in os.listdir(src_dir):
        for image_name in os.listdir(f"{src_dir}/{num_dir}/sentence/"):
            pure_name,ext=os.path.splitext(image_name)
            tar_get_dir=f"{dst_dir}/{num_dir}_{pure_name}"
            os.makedirs(tar_get_dir,exist_ok=True)
            shutil.copy2(f"{src_dir}/{num_dir}/sentence/{image_name}",f"{tar_get_dir}/{image_name}")

if __name__ == '__main__':
    # copy_image()
    # exit()
    with paddle.no_grad():
        #load_dataset_from_image()
        #test_fast_infer()
        batch_image_infer()
        #image_byte_infer()