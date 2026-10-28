"""H is a row channel. Only prediction enters precoding, truth enters evaluation."""
import numpy as np


def _complex(x,min_ndim=1):
    a=np.asarray(x,dtype=np.complex128)
    if a.ndim<min_ndim or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected finite nonempty complex channel')
    return a


def _scalar(x,name,allow_zero=False):
    x=float(x)
    if not np.isfinite(x) or x<0 or (not allow_zero and x==0):
        raise ValueError(name+' has invalid range')
    return x


def _energy(x,axis):
    try:
        with np.errstate(over='raise',invalid='raise'):
            value=np.sum(np.abs(x)**2,axis=axis,dtype=np.float64)
    except FloatingPointError as exc:
        raise ValueError('Channel energy overflow') from exc
    if not np.isfinite(value).all():raise ValueError('Nonfinite channel energy')
    return value


def make_precoder(prediction,scheme='ZF',noise_variance=.1,power=1.,rcond=1e-10):
    h=_complex(prediction,2)
    k,nt=h.shape[-2:]
    if k>nt:raise ValueError('This protocol requires users<=Tx antennas')
    p=_scalar(power,'power');noise=_scalar(noise_variance,'noise_variance',True)
    rcond=_scalar(rcond,'rcond')
    hermitian=h.conj().swapaxes(-1,-2)
    gram=h@hermitian
    if scheme.upper()=='ZF':
        v=hermitian@np.linalg.pinv(gram,rcond=rcond,hermitian=True)
        alpha=0.
    elif scheme.upper()=='RZF':
        if noise==0:raise ValueError('RZF requires positive noise; use ZF at zero loading')
        alpha=k*noise/p
        v=hermitian@np.linalg.solve(gram+alpha*np.eye(k),np.broadcast_to(np.eye(k),gram.shape))
    else:raise ValueError('scheme must be ZF or RZF')
    norm=np.sqrt(_energy(v,(-2,-1)))
    fallback=norm==0
    w=np.divide(np.sqrt(p)*v,norm[...,None,None],out=np.zeros_like(v),where=norm[...,None,None]>0)
    fixed=np.eye(nt,k,dtype=np.complex128)*np.sqrt(p/k)
    w=np.where(fallback[...,None,None],fixed,w)
    singular=np.linalg.svd(h,compute_uv=False)
    rank=np.sum(singular>np.sqrt(rcond)*singular[...,:1],axis=-1)
    return {'W':w,'fallback':fallback,'rank':rank,'rank_deficient':rank<k,'alpha':alpha}


def spectral_efficiency(true_future,w,noise_variance):
    h=_complex(true_future,2);w=_complex(w,2)
    if w.shape[:-2]!=h.shape[:-2] or w.shape[-2:]!=(h.shape[-1],h.shape[-2]):
        raise ValueError('H[...,K,Nt] and W[...,Nt,K] shape mismatch')
    noise=_scalar(noise_variance,'noise_variance',True)
    effective=h@w
    powers=_energy(effective,axis=())
    k=h.shape[-2]
    desired=np.diagonal(powers,axis1=-2,axis2=-1)
    interference=np.sum(np.where(np.eye(k,dtype=bool),0.,powers),axis=-1)
    denominator=interference+noise
    sinr=np.divide(desired,denominator,out=np.zeros_like(desired),where=denominator>0)
    sinr=np.where((denominator==0)&(desired>0),np.inf,sinr)
    user_se=np.log2(1+sinr)
    invalid=_energy(h,-1)==0
    return {'sinr':sinr,'user_se':user_se,'sum_se':user_se.sum(axis=-1),
            'invalid_true':invalid,'effective_channel':effective}


def beamforming_gain(true_future,prediction):
    h=_complex(true_future);pred=_complex(prediction)
    if h.shape!=pred.shape:raise ValueError('Beam channel shapes differ')
    power=_energy(pred,-1);truth=_energy(h,-1)
    fallback=power==0
    unit=np.divide(pred.conj(),np.sqrt(power)[...,None],out=np.zeros_like(pred),where=power[...,None]>0)
    fixed=np.zeros(pred.shape[-1],dtype=np.complex128);fixed[0]=1
    unit=np.where(fallback[...,None],fixed,unit)
    received=np.sum(h*unit,axis=-1)
    gain=_energy(received,axis=())
    eta=np.divide(gain,truth,out=np.full_like(truth,np.nan),where=truth>0)
    eta=np.clip(eta,0,1)
    with np.errstate(divide='ignore',invalid='ignore'):
        loss=-10*np.log10(eta)
    return {'efficiency':eta,'loss_db':loss,'fallback':fallback,'invalid_true':truth==0}


def evaluate_link(true_future,prediction,snr_db,channel_scale,scheme='ZF',power=1.):
    h=_complex(true_future,4);pred=_complex(prediction,4)
    if h.ndim!=4 or h.shape!=pred.shape:raise ValueError('Expected matching [B,U,Nt,Nf]')
    scale=_scalar(channel_scale,'training channel_scale')
    snr_db=float(snr_db)
    if not np.isfinite(snr_db):raise ValueError('Finite nominal SNR required')
    noise=10**(-snr_db/10)
    ht=(h/scale).transpose(0,3,1,2)
    hp=(pred/scale).transpose(0,3,1,2)
    prec=make_precoder(hp,scheme,noise,power)
    rates=spectral_efficiency(ht,prec['W'],noise)
    beam=beamforming_gain(ht,hp)
    return {'sum_se':rates['sum_se'].mean(axis=1),
            'user_se':rates['user_se'].mean(axis=1),
            'beam_loss_db':beam['loss_db'].mean(axis=1),
            'beam_efficiency':beam['efficiency'].mean(axis=1),
            'precoder_fallback_fraction':np.mean(prec['fallback'],axis=1),
            'su_fallback_fraction':np.mean(beam['fallback'],axis=1),
            'invalid_true_count':np.sum(rates['invalid_true'],axis=(1,2)),
            'W_per_subcarrier':prec['W'],'noise_variance':noise,
            'snr_reference':'training_mean_per_element_power','power_per_subcarrier':float(power)}
