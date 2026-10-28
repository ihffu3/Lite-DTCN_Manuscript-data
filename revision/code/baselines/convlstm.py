"""Fixed-history peephole ConvLSTM CSI adaptation of Lv et al. (VTC2024).

Spatial axes are BS-Tx x subcarrier, RI are two input channels. No closed-loop
pilot trigger, future reset, persistent online state or external data is used.
"""
import torch
from torch import nn

CONFIGS={'five_layer_adapter':(64,32,8,4,1),'light_adapter':(16,16)}

class ConvLSTMCell(nn.Module):
    def __init__(self,input_channels,hidden_channels,height,width,kernel=3):
        super().__init__()
        if kernel<1 or kernel%2!=1:raise ValueError('Use an odd positive convolution kernel')
        self.hidden_channels=hidden_channels;self.height=height;self.width=width;self.kernel=kernel
        self.gates=nn.Conv2d(input_channels+hidden_channels,4*hidden_channels,kernel,padding=kernel//2,bias=True)
        # Elementwise learned spatial peepholes; geometry is fixed by the CSI grid.
        self.peephole_i=nn.Parameter(torch.zeros(1,hidden_channels,height,width))
        self.peephole_f=nn.Parameter(torch.zeros(1,hidden_channels,height,width))
        self.peephole_o=nn.Parameter(torch.zeros(1,hidden_channels,height,width))
        nn.init.xavier_uniform_(self.gates.weight);nn.init.zeros_(self.gates.bias)
    def forward(self,x,h,c):
        i,f,g,o=self.gates(torch.cat((x,h),dim=1)).chunk(4,dim=1)
        i=torch.sigmoid(i+self.peephole_i*c);f=torch.sigmoid(f+self.peephole_f*c)
        c=f*c+i*torch.tanh(g);o=torch.sigmoid(o+self.peephole_o*c)
        return o*torch.tanh(c),c

class ConvLSTMPredictor(nn.Module):
    def __init__(self,*,history_length=10,n_tx=8,n_freq=32,hidden_channels=(64,32,8,4,1),kernel=3):
        super().__init__()
        if not hidden_channels or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (history_length,n_tx,n_freq,*hidden_channels)):
            raise ValueError('Dimensions must be positive integers')
        self.history_length=history_length;self.n_tx=n_tx;self.n_freq=n_freq
        self.hidden_channels=tuple(hidden_channels);self.kernel=kernel
        channels=(2,*self.hidden_channels)
        self.cells=nn.ModuleList([ConvLSTMCell(channels[i],channels[i+1],n_tx,n_freq,kernel) for i in range(len(hidden_channels))])
        self.readout=nn.Conv2d(channels[-1],2,1,bias=True)
        nn.init.xavier_uniform_(self.readout.weight);nn.init.zeros_(self.readout.bias)
    def forward(self,history,conditions=None):
        if (not isinstance(history,torch.Tensor) or history.ndim!=5 or history.shape[0]<1
                or tuple(history.shape[1:])!=(self.history_length,self.n_tx,self.n_freq,2)
                or history.dtype!=self.readout.weight.dtype):raise ValueError('Invalid normalized RI history')
        sequence=history.permute(0,1,4,2,3)
        for cell in self.cells:
            h=sequence.new_zeros((history.shape[0],cell.hidden_channels,self.n_tx,self.n_freq));c=torch.zeros_like(h)
            outputs=[]
            for t in range(self.history_length):
                h,c=cell(sequence[:,t],h,c);outputs.append(h)
            sequence=torch.stack(outputs,dim=1)
        return self.readout(sequence[:,-1]).permute(0,2,3,1)

def build_convlstm(config='five_layer_adapter',*,n_tx=8,n_freq=32):
    name=config if isinstance(config,str) else config.get('id')
    if name not in CONFIGS:raise ValueError('Unknown frozen ConvLSTM configuration')
    if isinstance(config,dict):
        for key,value in config.items():
            if key=='id':continue
            expected={'hidden_channels':list(CONFIGS[name]),'kernel':3,'peephole':True}.get(key)
            if expected is None or value!=expected:raise ValueError('Changed frozen ConvLSTM field: '+key)
    return ConvLSTMPredictor(n_tx=n_tx,n_freq=n_freq,hidden_channels=CONFIGS[name])

def complexity(model):
    area=model.n_tx*model.n_freq;channels=(2,*model.hidden_channels)
    params=0;macs=0
    for c,h in zip(channels[:-1],channels[1:]):
        params+=(c+h)*4*h*model.kernel**2+4*h+3*h*area
        macs+=model.history_length*area*(c+h)*4*h*model.kernel**2
    params+=2*channels[-1]+2;macs+=area*2*channels[-1]
    return {'parameters':params,'macs':macs,'definition':'Conv MACs only; excludes peephole/gate elementwise arithmetic, activations and bias; not latency'}
