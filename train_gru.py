import random
import warnings
import os
import paddle
from paddle import nn
from paddle.io import DataLoader 
from paddle.vision import transforms
from data.datapreprocess import pipline_data_gru
from data.train_dataset import GRUDataset
from network import WordImageSliceGRUCLS
import paddle.optimizer as optim
from lr import MYLR
from paddle.metric import accuracy
from paddle.metric import Recall
from visualdl import LogWriter

from network import load_model
from options.train_options import trainparser



def train(train_loader:DataLoader, model:nn.Layer,loss_function, optimizer, epoch, args):
    """
    训练模型
    """
    sumloss=0
    sumacc=0
    recall=Recall()
    model.train()
    for bid, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
        output=model(batch_image[0])
        #loss=loss_function(output.reshape(-1,2),paddle.to_tensor(batch_image_type).reshape([-1,1]).unsqueeze(-1))# 2,80 是硬编码
        loss=loss_function(output,paddle.to_tensor(batch_image_type).reshape([train_loader.batch_size,-1]).unsqueeze(-1))# 2,80 是硬编码
        acc = accuracy(output.reshape([-1,2]),paddle.to_tensor(batch_image_type).reshape([-1,1]))
        
        recall.update(output.reshape([-1,2]).argmax(-1).unsqueeze(-1), paddle.to_tensor(batch_image_type).reshape([-1]).unsqueeze(-1))
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        sumloss+=float(loss)
        avgloss=sumloss/(bid+1)
        sumacc+=float(acc)
        avgacc=sumacc/(bid+1)
    with LogWriter(logdir=args.logdir) as writer:
        # use `add_scalar` to record scalar values
        writer.add_scalar(tag="train/loss", step=epoch, value=avgloss)
        writer.add_scalar(tag="train/acc", step=epoch, value=avgacc)
        writer.add_scalar(tag="train/recall", step=epoch, value=recall.accumulate())
    return {"loss":avgloss,"acc":avgacc,"epoch":epoch,"recall":recall.accumulate()}

def eval(test_loader, model:nn.Layer,loss_function, epoch, args):
    """
    评估模型
    """
    sumloss=0
    sumacc=0
    recall=Recall()
    for bid ,(batch_image,batch_image_type,batch_image_import_flag) in enumerate(test_loader):
        output=model(batch_image[0])
        loss=loss_function(output,paddle.to_tensor(batch_image_type).reshape([test_loader.batch_size,-1]).unsqueeze(-1))# 2,80 是硬编码
        acc = accuracy(output.reshape([-1,2]),paddle.to_tensor(batch_image_type).reshape([-1,1]))
        recall.update(output.reshape([-1,2]).argmax(-1).unsqueeze(-1), paddle.to_tensor(batch_image_type).reshape([-1]).unsqueeze(-1))
        sumloss+=float(loss)
        avgloss=sumloss/(bid+1)
        
        sumacc+=float(acc)
        avgacc=sumacc/(bid+1)
    with LogWriter(logdir=args.logdir) as writer:
        # use `add_scalar` to record scalar values
        writer.add_scalar(tag="eval/acc", step=epoch, value=avgacc)
        writer.add_scalar(tag="eval/loss", step=epoch, value=avgloss)
        writer.add_scalar(tag="eval/recall", step=epoch, value=recall.accumulate())
    return {"loss":avgloss,"acc":avgacc,"epoch":epoch,"recall":recall.accumulate()}


def get_dataloader(dataset_dir,expansion,args):
    # 获得数据loader
    train_data,test_data=pipline_data_gru(dataset_dir=dataset_dir,test_size=args.test_size)
    # 按照比例划分train 和 test数据
    pin_transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485], std=[0.229]
            )
        ]
    )
    # 数据预处理
    train_dataset,test_dataset=GRUDataset(train_data,pin_transform),GRUDataset(test_data,pin_transform)
    train_loader=DataLoader(train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=args.workers,
        batch_sampler=None,
        drop_last=False,
    )
    test_loader=DataLoader(test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=args.workers,
        batch_sampler=None,
        drop_last=False,
    )
    return train_loader,test_loader

def checkpoint(model:nn.Layer,optimizer,model_info:dict,checkpoint_dir):
    save_prefix="gru_epoch_{:03d}_".format(model_info["epoch"])
    model_path=os.path.join(checkpoint_dir, save_prefix+"model.pdparams")
    encoder_k_model_path=os.path.join(checkpoint_dir, save_prefix+"encoder_k_model.pdparams")
    encoder_q_model_path=os.path.join(checkpoint_dir, save_prefix+"encoder_q_model.pdparams")
    optimizer_path=os.path.join(checkpoint_dir, save_prefix+"optimizer.pdopt")
    paddle.save(model.state_dict(),model_path)
    paddle.save(model.encoder_model_K.state_dict(),encoder_k_model_path)
    paddle.save(model.encoder_model_Q.state_dict(),encoder_q_model_path)
    paddle.save(optimizer.state_dict(),optimizer_path)
def main():
    parser=trainparser()
    args = parser.parse_args() 
    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(args.seed)
        #cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.cpu :
        paddle.device.set_device('cpu')
        warnings.warn(
            "当前采用cpu训练模型，最好选用GPU进行训练模型，在CPU上训练速度会非常慢 "
        )
    else:
        paddle.device.set_device('gpu')
    
    # 初始化模型
    encoder_q_model,encoder_k_model=load_model(args.moco_model)# 加载对比模型作为backbone
    cls_model=WordImageSliceGRUCLS(encoder_model_k=encoder_q_model,encoder_model_q=encoder_k_model,freeze_flag=False)
    
    # 加载数据
    train_loader,test_loader=get_dataloader(dataset_dir= args.data,expansion=args.expansion,args=args)
    args.cos=True
    lr=MYLR(learning_rate=args.lr,cos=args.cos,verbose=True,schedule=args.schedule,epochs=args.epochs)
    optimizer = optim.SGD(
            learning_rate=lr,
            parameters=cls_model.parameters(),
            weight_decay=args.weight_decay,
     )# pytorch 对应的优化器是SGD
    # 模型训练和评估
    loss_function = nn.CrossEntropyLoss(weight=paddle.to_tensor([0.25,0.75]))
    for epoch in range(args.start_epoch, args.epochs):
        train_info=train(train_loader, cls_model,loss_function, optimizer, epoch, args)
        #print(train_info)
        lr.step(epoch)# 更新一下学习率
        with paddle.no_grad():
            cls_model.eval()
            print("eval",eval(test_loader,cls_model,loss_function,epoch,args))
            cls_model.train()
        if epoch>0 and epoch%args.checkpoint_steps==0:
            checkpoint(cls_model,optimizer,train_info,args.checkpoint)
#    if global_steps % args.checkpoint_steps == 0:
if __name__ == "__main__":
    main()
    
