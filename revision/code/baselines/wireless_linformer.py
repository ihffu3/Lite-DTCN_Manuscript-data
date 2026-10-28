"""Wireless TMLP LinFormer/DSLH adaptation of Jin et al., arXiv:2410.21351v1.

No low-rank attention, future-label augmentation, pretraining or import-time IO.
Unspecified paper widths/embedding/initialization follow frozen protocol 1.3.
"""
import torch
from torch import nn

CONFIGS={
    'paper_scale_direct_adapter':dict(d_model=512,layers=6,time_hidden=20,ff_hidden=2048),
    'light_direct_adapter':dict(d_model=128,layers=2,time_hidden=20,ff_hidden=512),
}

class TimeMLP(nn.Module):
    def __init__(self,length,hidden):
        super().__init__();self.fc1=nn.Linear(length,hidden);self.fc2=nn.Linear(hidden,length)
    def forward(self,x):return self.fc2(torch.relu(self.fc1(x.transpose(1,2)))).transpose(1,2)

class EncoderBlock(nn.Module):
    def __init__(self,length,d_model,time_hidden,ff_hidden):
        super().__init__();self.time_mlp=TimeMLP(length,time_hidden)
        self.norm1=nn.LayerNorm(d_model,eps=1e-5);self.norm2=nn.LayerNorm(d_model,eps=1e-5)
        self.ff=nn.Sequential(nn.Linear(d_model,ff_hidden),nn.ReLU(),nn.Linear(ff_hidden,d_model))
    def forward(self,x):
        x=self.norm1(x+self.time_mlp(x));return self.norm2(x+self.ff(x))

class SeparableHead(nn.Module):
    """NL=1 direct target: Linear(L,1,no bias) then Linear(d_model,F,bias)."""
    def __init__(self,length,d_model,features):
        super().__init__();self.time=nn.Linear(length,1,bias=False);self.features=nn.Linear(d_model,features)
    def forward(self,x):return self.features(self.time(x.transpose(1,2)).squeeze(-1))

class WirelessLinFormer(nn.Module):
    def __init__(self,*,history_length=10,n_tx=8,n_freq=32,d_model=512,layers=6,time_hidden=20,ff_hidden=2048):
        super().__init__()
        for value in (history_length,n_tx,n_freq,d_model,layers,time_hidden,ff_hidden):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Dimensions must be positive integers')
        self.history_length=history_length;self.n_tx=n_tx;self.n_freq=n_freq;self.d_model=d_model
        self.n_layers=layers;self.time_hidden=time_hidden;self.ff_hidden=ff_hidden;self.features=n_tx*n_freq*2
        self.embedding=nn.Linear(self.features,d_model)
        self.encoder=nn.ModuleList([EncoderBlock(history_length,d_model,time_hidden,ff_hidden) for _ in range(layers)])
        self.head=SeparableHead(history_length,d_model,self.features)
        self.apply(self._init)
    @staticmethod
    def _init(layer):
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)
        elif isinstance(layer,nn.LayerNorm):nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,history,conditions=None):
        if (not isinstance(history,torch.Tensor) or history.ndim!=5 or history.shape[0]<1
                or tuple(history.shape[1:])!=(self.history_length,self.n_tx,self.n_freq,2)
                or history.dtype!=self.embedding.weight.dtype):raise ValueError('Invalid normalized RI history')
        x=self.embedding(history.reshape(history.shape[0],self.history_length,self.features))
        for block in self.encoder:x=block(x)
        return self.head(x).reshape(history.shape[0],self.n_tx,self.n_freq,2)

def build_wireless_linformer(config='paper_scale_direct_adapter',*,n_tx=8,n_freq=32):
    name=config if isinstance(config,str) else config.get('id')
    if name not in CONFIGS:raise ValueError('Unknown frozen wireless LinFormer configuration')
    if isinstance(config,dict):
        for key,value in config.items():
            if key!='id' and (key not in CONFIGS[name] or CONFIGS[name][key]!=value):raise ValueError('Changed frozen LinFormer field: '+key)
    return WirelessLinFormer(n_tx=n_tx,n_freq=n_freq,**CONFIGS[name])

def weighted_future_mse(prediction,target):
    """Real-coordinate WMSE; NL=1 is exactly the common normalized RI MSE.

    For a multi-target diagnostic its scale is half the paper's complex norm
    convention; all optimization weights n**(-1/2) remain identical.
    """
    if prediction.shape!=target.shape or prediction.ndim<3 or prediction.shape[1]<1:raise ValueError('Expected [B,NL,...] matching predictions/targets')
    weights=torch.arange(1,prediction.shape[1]+1,device=prediction.device,dtype=prediction.dtype).rsqrt()
    losses=(prediction-target).square().flatten(2).mean(2)
    return (losses*weights[None,:]).mean()

def complexity(model):
    l=model.history_length;d=model.d_model;m=model.time_hidden;f=model.ff_hidden;c=model.features;n=model.n_layers
    params=(c*d+d)+n*(2*l*m+m+l+4*d+2*d*f+f+d)+(l+d*c+c)
    macs=l*c*d+n*(2*d*l*m+2*l*d*f)+d*l+d*c
    return {'parameters':params,'macs':macs,'definition':'Linear MACs only; excludes bias, activation, LayerNorm, additions; not latency'}
