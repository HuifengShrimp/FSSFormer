import argparse
import json
import spu.utils.distributed as ppd
import time
import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import sklearn.metrics as metrics
torch.autograd.set_detect_anomaly(True)
import torch.nn.init as init


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates).clone()

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        def exp_approx(x):
            iterations = 1  # 迭代次数
            result = 1.0 + x / (1 << iterations)  # 初始化结果
            for _ in range(iterations):
                result = result*result  # 平方运算
            return result

        def reciprocal_approx(x):
            iterations = 1  # 迭代次数
            result = 1.0 + x / (1 << iterations)  # 初始化结果
            for _ in range(iterations):
                result = result * (2 - x * result)  # 牛顿-拉夫逊迭代
            return result

        def softmax_approx(x):
            maximum_value = torch.max(x)  # 最大值
            logits = x - maximum_value  # 计算logits
            numerator = F.relu(logits)  # 计算分子
            inv_denominator = torch.sum(numerator,dim=-1,keepdim=True)+1e-6  # 计算分母的倒数
            return numerator / inv_denominator  # 返回softmax值

        attn = softmax_approx(sim)
        #1:attn = F.relu(sim)
        #0:attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out, attn

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []
        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x
        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_numer, return_attn = False):
        x = self.numerical_embedder(x_numer)
        b, _, _ = x.size()
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x, attns = self.transformer(x, return_attn = True)
        x = x[:, 0]
        logits = self.to_logits(x)
        if not return_attn:
            return logits

        return logits, attns

def weights_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            print(name, param.shape)
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            print(name, param.shape)
            # 使用常数值初始化偏置
            nn.init.constant_(param, 0.0)


def train(model, n_epochs=500, lr=0.01):
    print('Train model with plaintext features\n------\n')
    x, y = breast_cancer()
    x = torch.Tensor(x)
    y = torch.Tensor(y).view(-1, 1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        pred_y = model(x)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print('Train model finished\n------\n')

def breast_cancer(
    train: bool = True,
    *,
    normalize: bool = True,
):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']
    y = y.astype(dtype=np.float64)
    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    if train:
        x_ = x_train
        y_ = y_train
    else:
        x_ = x_test
        y_ = y_test
    return x_.astype(dtype=np.float32), y_.astype(dtype=np.float32)
    
import time

def run_inference_on_cpu(model):
    print('Run on CPU\n------\n')
    x_test, y_test = breast_cancer(False)
    x = torch.Tensor(x_test)
    start_ts = time.time()
    y_pred = model(x).cpu().detach().numpy()
    end_ts = time.time()
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)

from collections import OrderedDict
from jax.tree_util import tree_map


def run_inference_on_spu(model):
    print('Run on SPU\n------\n')

    # load parameters and buffers on P1
    params_buffers = OrderedDict()
    for k, v in model.named_parameters():
        params_buffers[k] = v
       # print(f"model.parameters: {v}\n")    
    for k, v in model.named_buffers():
        params_buffers[k] = v
    params = ppd.device("P1")(
        lambda input: tree_map(lambda x: x.detach().numpy(), input)
    )(params_buffers)

    # load inputs on P2
    x, _ = ppd.device("P2")(breast_cancer)(False)

    start_ts = time.time()
    y_pred_ciphertext, _ = ppd.device('SPU')(model)(params, x, return_attn=True)
    end_ts = time.time()
    y_pred_plaintext = ppd.get(y_pred_ciphertext)
    _, y_test = breast_cancer(False)
    auc = metrics.roc_auc_score(y_test, y_pred_plaintext)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")
    return auc


    
if __name__=='__main__':
    torch.manual_seed(0)
    model = FTTransformer(
        num_continuous=30,  
        dim=64,            
        depth=6,           
        heads=8,           
        dim_head=16,        
        dim_out=1,          
        attn_dropout=0.1,   
        ff_dropout=0.1      
    )
    weights_init(model)
    train(model)
    run_inference_on_cpu(model)
    run_inference_on_spu(model)

