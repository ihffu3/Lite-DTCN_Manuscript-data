"""Fit mixed training statistics with equal weight per independent environment."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .arrays import complex_to_ri,record_parts


def _real_ri(ri):
    x=np.asarray(ri)
    if (x.ndim<1 or x.shape[-1]!=2 or not np.issubdtype(x.dtype,np.number)
            or np.iscomplexobj(x) or not np.isfinite(x).all()):
        raise ValueError('Expected a finite real numeric RI array')
    return x


@dataclass(frozen=True)
class Normalizer:
    mean: tuple
    std: tuple
    training_environment_ids: tuple

    def transform(self,ri):
        x=_real_ri(ri)
        return (x-np.asarray(self.mean))/np.asarray(self.std)

    def inverse(self,ri):
        x=_real_ri(ri)
        return x*np.asarray(self.std)+np.asarray(self.mean)

    def to_dict(self):
        return {'schema_version':'1.0','mean':list(self.mean),'std':list(self.std),
                'training_environment_ids':list(self.training_environment_ids),
                'weighting':'equal_environment','std_floor':1e-12}

    @property
    def sha256(self):
        data=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


def fit_normalizer(training_records):
    groups={}
    for record in training_records:
        h,_,meta=record_parts(record)
        if meta['split']!='train':
            raise ValueError('Normalization may only be fitted on train records')
        x=complex_to_ri(h).astype(np.float64).reshape(-1,2)
        n,mean,m2=groups.get(meta['environment_id'],(0,np.zeros(2),np.zeros(2)))
        # Merge centered second moments for numerical stability, including repeated
        # records belonging to one environment; length does not alter its final weight.
        nx=len(x);mx=x.mean(axis=0);dx=mx-mean;total=n+nx
        m2=m2+np.sum((x-mx)**2,axis=0)+dx*dx*(n*nx/total)
        groups[meta['environment_id']]=(total,mean+dx*(nx/total),m2)
    if not groups:
        raise ValueError('At least one training environment is required')
    ids=tuple(sorted(groups))
    means=np.stack([groups[k][1] for k in ids])
    variances=np.stack([groups[k][2]/groups[k][0] for k in ids])
    mean=means.mean(axis=0)
    variance=np.mean(variances+(means-mean)**2,axis=0)
    std=np.maximum(np.sqrt(variance),1e-12)
    return Normalizer(tuple(float(v) for v in mean),tuple(float(v) for v in std),ids)
