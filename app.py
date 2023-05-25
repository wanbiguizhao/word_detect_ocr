from flask import Flask, render_template, request,jsonify
import json
import paddle
import json
import numpy as np
try:
    from mocov1.cls.pdpd.models import WordImageSliceMLPCLS
    from mocov1.moco.resnetmodels import HackResNet
    from mocov1.data.dataset import WIPByteDataset
except:
    from cls.pdpd.models import WordImageSliceMLPCLS
    from moco.resnetmodels import HackResNet
    from data.dataset import WIPByteDataset

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
cls_model=load_model()
cls_model.eval()
app = Flask(__name__)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def build_dataloader(image_byte,batch_size=256):
    from paddle.vision import transforms
    from paddle.io import DataLoader 
    #from mocov1 import WIPByteDataset
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

    wip=WIPByteDataset(image_byte,transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = DataLoader(
            wip,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            #pin_memory=True, paddle 没有过
            #sampler=None,
            drop_last=False,
        )
    return wip,train_loader
def det_image(image_bytes,batch_size=256):
    batch_size=256
    result=[]
    wip,train_loader=build_dataloader(image_bytes,batch_size=batch_size)   
    for k, (images, _) in enumerate(train_loader):  
        #print(images[0])  
        print(type(images[0]))
        predict_info=cls_model(images[0])

        predict_labels=paddle.argmax(predict_info,axis=-1)# 预测的每个图片切片的类型。
        #predict_labels 根据这个重新图片的切割的位置。
        for index in range(len(predict_labels)):
            wip_index=index+k*batch_size
            seg_image_info = wip.data_list[wip_index]
            seg_beg_index=seg_image_info["seg_beg_index"]
            seg_end_index=seg_image_info["seg_end_index"]
            #if int(predict_labels[index])==1:
            mid_index=(seg_beg_index+seg_end_index)//2
            result.append([wip_index,int(predict_labels[index]),mid_index])
    han_appear_flag=False
    word_list=[]
    han_beg_index=0
    pre_pixel_index=0
    for ret in result:
        label_index,label_type,pixel_index=ret[0],ret[1],ret[2]
        if label_type==0:# 表示出现了汉字
            if han_appear_flag==True:
                continue
            else:
                han_appear_flag=True 
                #word_list.append([beg_pixel_index,pixel_index])
                han_beg_index=pre_pixel_index
        else:
            if han_appear_flag==True:
                han_appear_flag=False
                word_list.append([han_beg_index,pixel_index])
                wip.origin_image[:,han_beg_index]=0
                wip.origin_image[:,pixel_index]=0
            else:
                pass 
        pre_pixel_index=pixel_index
    if han_appear_flag:
        word_list.append([han_beg_index,pixel_index])
    # from PIL import Image 
    
    # pil_image=Image.fromarray(wip.origin_image)
    # pil_image.show()


    return word_list
@app.route("/",methods = ['GET', 'POST'])
def hello_world():

    batch_size=256            
    if request.method =="POST":
        image_bytes = request.files['file'].read()
        result=det_image(image_bytes,batch_size=batch_size)
                
        return jsonify(json.loads(json.dumps(result,cls=NpEncoder)))
    html="""
    <html>
    <body>
      <form action = "/" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>   
    </body>
    </html>
    """
    return html

# flask  --debug --app mocov1/app run

if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port="8088",
        debug=False
    )