import paddle.optimizer as optim
import math
class MYLR(optim.lr.LRScheduler):
    """ 
    自定义的一个学习率曲线
    """

    def __init__(self,learning_rate=0.1,epochs=500, last_epoch=-1, verbose=False,cos=False,schedule=[120, 160]):
        
        self.cos=cos
        self.epochs=epochs# 总共的epoch数量
        self.schedule=schedule
        super(MYLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.cos:
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.epochs))
        else: 
            lr=self.base_lr
            for milestone in self.schedule:
                lr *= 0.1 if self.last_epoch >= milestone else 1.0
            return lr
