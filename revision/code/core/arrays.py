"""Axis conversion and realization-local historical windows."""
import numpy as np


def complex_to_ri(value):
    x=np.asarray(value)
    if not np.iscomplexobj(x) or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty complex array')
    return np.stack((x.real,x.imag),axis=-1)


def ri_to_complex(value):
    x=np.asarray(value)
    if (x.ndim<1 or x.shape[-1]!=2 or not x.size or np.iscomplexobj(x)
            or not np.issubdtype(x.dtype,np.number) or not np.isfinite(x).all()):
        raise ValueError('Expected finite real array with final RI axis of length2')
    dtype=np.complex64 if x.dtype==np.float32 else np.complex128
    return (x[...,0]+1j*x[...,1]).astype(dtype,copy=False)


def _positive_integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def window_indices(n_time,history_length,horizon_steps,origin_start=19,stride=1):
    n=_positive_integer(n_time,'n_time')
    length=_positive_integer(history_length,'history_length')
    horizon=_positive_integer(horizon_steps,'horizon_steps')
    stride=_positive_integer(stride,'stride')
    if (isinstance(origin_start,(bool,np.bool_)) or not isinstance(origin_start,(int,np.integer))
            or origin_start<length-1):
        raise ValueError('origin_start must be an integer with sufficient history')
    origins=np.arange(origin_start,n-horizon,stride,dtype=np.int64)
    if not len(origins):
        raise ValueError('No valid forecast origins inside this realization')
    history=origins[:,None]+np.arange(-length+1,1,dtype=np.int64)[None,:]
    return history,origins+horizon


def record_parts(record):
    h=np.asarray(record['H'])
    time=np.asarray(record['time_s'],dtype=np.float64)
    meta=record.get('metadata',record.get('meta'))
    if h.ndim!=4 or not np.iscomplexobj(h) or not h.size or not np.isfinite(h).all():
        raise ValueError('H must be finite complex [T,U,Nt,Nf]')
    if (time.shape!=(h.shape[0],) or len(time)<2 or not np.isfinite(time).all()
            or not np.allclose(np.diff(time),.0005,rtol=0,atol=1e-10)):
        raise ValueError('time_s must match T on the0.5ms observation grid')
    if not isinstance(meta,dict) or any(not isinstance(meta.get(k),str) or not meta[k]
            for k in ('environment_id','realization_id','user_group_id','split')):
        raise ValueError('Missing explicit split/environment/realization/user-group metadata')
    return h,time,meta


def build_windows(record,history_length,horizon_steps,origin_start=19,stride=1):
    h,time,meta=record_parts(record)
    hist,target=window_indices(len(h),history_length,horizon_steps,origin_start,stride)
    # For each origin retain the synchronized user order; never join records.
    users=np.tile(np.arange(h.shape[1]),len(hist))
    histories=np.repeat(hist,h.shape[1],axis=0)
    targets=np.repeat(target,h.shape[1])
    origins=histories[:,-1]
    x=h[histories,users[:,None]]
    y=h[targets,users]
    rows=[dict(environment_id=meta['environment_id'],realization_id=meta['realization_id'],
               user_group_id=meta['user_group_id'],user_id=int(u),t_index=int(t),
               target_index=int(k),horizon_steps=int(horizon_steps),
               horizon_ms=float(horizon_steps*.5)) for u,t,k in zip(users,origins,targets)]
    return {'X':complex_to_ri(x),'Y':complex_to_ri(y),'rows':rows}
