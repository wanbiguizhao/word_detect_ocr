from paddle import nn 
import paddle
from models.resnetmodels import HackResNet
from models.moco import MoCo
def load_model(model_path):
    # 加载基于对比学习的两组编码器
    model = MoCo(
        HackResNet
    )
    model.set_state_dict(paddle.load(model_path))
    print(model)
    encoder_k_model=model.encoder_k
    encoder_q_model=model.encoder_k
    return encoder_q_model,encoder_k_model 

class WordImageSliceMLPCLS(nn.Layer):
    """
    基于神经网络的0-1分类器
    """
    def __init__(self,encoder_model_k:nn.Layer,encoder_model_q:nn.Layer,dim=128,num_classes=2,freeze_flag=False) -> None:
        super().__init__()
        self.encoder_model_K:nn.Layer=encoder_model_k #基于对比学习的backbone 
        self.encoder_model_Q:nn.Layer=encoder_model_q #基于对比学习的backbone 
        #self.encoder_model=self.encoder_model_Q
        # 冻结backbone的参数
        self.gate=nn.Sequential( 
            nn.Linear(2*dim,dim),
            nn.Linear(dim,2)
            )
        self.linear = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16), 
                nn.ReLU(),
                #nn.Linear(6, 8),
                nn.Linear(16, num_classes),
            )
        self.g0_sig=nn.Sigmoid()
        self.g1_sig=nn.Sigmoid()
        if freeze_flag:
            self.freeze_backone()

    def forward(self, *inputs, **kwargs):
        kx=self.encoder_model_K(inputs) # batch_size,dim
        qx=self.encoder_model_Q(inputs)
        gate_value=self.gate(paddle.concat([kx,qx],axis=1))
        g0=self.g0_sig(gate_value[:,0].unsqueeze(-1))
        g1=self.g1_sig(gate_value[:,1].unsqueeze(-1))
        x=self.linear(g0*kx+g1*qx)
        return x
    def freeze_backone(self):
        # 冻结编码器参数。
        for encoder_model in [self.encoder_model_K,self.encoder_model_Q]:
            for param in encoder_model.parameters():
                param.stop_gradient = True
