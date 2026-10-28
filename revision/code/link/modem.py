"""Gray square constellations and ideal effective-CSI scalar receive equalization."""
import numpy as np
from .precoding import _complex,_scalar


def _constellation(name):
    if name.upper()=='QPSK':return 2,np.array([-1.,1.]),np.sqrt(2),np.array([[0],[1]],np.uint8)
    if name.upper()=='16QAM':return 4,np.array([-3.,-1.,3.,1.]),np.sqrt(10),np.array([[0,0],[0,1],[1,0],[1,1]],np.uint8)
    raise ValueError('Supported modulations are QPSK and16QAM')


def modulate(bits,modulation):
    bps,levels,scale,_=_constellation(modulation)
    b=np.asarray(bits)
    if (not b.size or b.ndim<1 or b.shape[-1]!=bps
            or not (np.issubdtype(b.dtype,np.integer) or b.dtype==bool)
            or not np.all((b==0)|(b==1))):
        raise ValueError('Bits must be binary integers with final modulation axis')
    axis_bits=bps//2;weights=2**np.arange(axis_bits-1,-1,-1)
    i=np.sum(b[...,:axis_bits]*weights,axis=-1)
    q=np.sum(b[...,axis_bits:]*weights,axis=-1)
    return (levels[i]+1j*levels[q])/scale


def demodulate(symbols,modulation):
    _,levels,scale,lookup=_constellation(modulation)
    s=np.asarray(symbols,dtype=np.complex128)
    if not s.size or not np.isfinite(s).all():raise ValueError('Finite symbols required')
    i=np.argmin(abs(s.real[...,None]*scale-levels),axis=-1)
    q=np.argmin(abs(s.imag[...,None]*scale-levels),axis=-1)
    return np.concatenate((lookup[i],lookup[q]),axis=-1)


def complex_awgn(shape,variance,rng):
    variance=_scalar(variance,'noise variance',True)
    return np.sqrt(variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))


def simulate_ber_batch(true_future,w,bits,noise,modulation):
    h=_complex(true_future,2);w=_complex(w,2)
    symbols=modulate(bits,modulation)
    noise=np.asarray(noise,dtype=np.complex128)
    if (h.shape[:-2]!=w.shape[:-2] or w.shape[-2:]!=(h.shape[-1],h.shape[-2])
            or symbols.shape[:-2]!=h.shape[:-2] or symbols.shape[-2]!=h.shape[-2]
            or noise.shape!=symbols.shape or not np.isfinite(noise).all()):
        raise ValueError('BER channel/precoder/bit/noise axes mismatch')
    effective=h@w
    desired=np.diagonal(effective,axis1=-2,axis2=-1)
    received=effective@symbols+noise
    nonzero=abs(desired)>0
    equalized=np.divide(received,desired[...,None],out=np.zeros_like(received),where=nonzero[...,None])
    decoded=demodulate(equalized,modulation)
    errors=decoded!=np.asarray(bits)
    return {'errors':int(np.count_nonzero(errors)),'bits':int(errors.size),
            'errors_by_user':np.sum(errors,axis=(-2,-1)),
            'bits_per_user':int(symbols.shape[-1]*np.asarray(bits).shape[-1]),
            'zero_effective_gain_count':int(np.count_nonzero(~nonzero)),
            'receiver':'ideal_current_effective_diagonal_CSI_scalar_equalizer',
            'ici':'ignored_block_fading'}
