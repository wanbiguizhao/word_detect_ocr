# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import paddle
import paddle.nn as nn


class MoCo(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # 这个编码器用的一样
        self.encoder_q:nn.Layer = base_encoder(num_classes=dim)
        self.encoder_k:nn.Layer = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            #param_k.data.copy_(param_q.data)  # initialize pytorch
            param_k.set_value(param_q) #paddle
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))#randn paddle 和pytorch 不一样的地方
        self.queue = nn.functional.normalize(self.queue, axis=0)# 对数据进行一次归一化操作，K个字典

        self.register_buffer("queue_ptr", paddle.zeros([1], dtype='int32'))# 单独生成了一个指针？指向某个位置。指向了某个梯度？,paddle 要求必须是[1]这样的数组

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            #param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)# 动量更新参数
            param_k.set_value(param_k* self.m + param_q * (1.0 - self.m) )
            # paddle 的代码

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)# 收集了所有的K

        batch_size = keys.shape[0] # 知道了batch_size

        ptr = int(self.queue_ptr)# 指针
        assert self.K % batch_size == 0  # for simplicity，K必须是batch_size的整数倍。

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T # 更新了队列，把batch中的keys更新到队列中。
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr # 更新指针

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # 之后的论文不再使用这个步骤了，shuffling BN.

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, axis=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = paddle.einsum("nc,nc->n", q, k).unsqueeze(-1)# 百度把pytorch中要添加[]的地方去掉，不添加中括号的地方加上。
        # negative logits: NxK
        l_neg = paddle.einsum("nc,ck->nk", q, self.queue.clone().detach())

        # logits: Nx(1+K)
        logits = paddle.concat([l_pos, l_neg], axis=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = paddle.zeros((logits.shape[0],), dtype='int32')#.cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels# labels 表示logits中第0列是正确的


# utils
@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        paddle.ones_like(tensor) for _ in range(paddle.distributed.get_world_size())
    ]
    paddle.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = paddle.concat(tensors_gather, axis=0)
    return output
