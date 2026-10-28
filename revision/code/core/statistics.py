"""Crossed resampling over independent environment and model-seed axes."""
import numpy as np


def paired_bootstrap(differences,n_bootstrap=2000,seed=20261015):
    x=np.asarray(differences,dtype=np.float64)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():
        raise ValueError('differences must be finite nonempty [seed,environment]')
    if isinstance(n_bootstrap,bool) or not isinstance(n_bootstrap,(int,np.integer)) or n_bootstrap<1:
        raise ValueError('n_bootstrap must be a positive integer')
    rng=np.random.default_rng(seed)
    ns,ne=x.shape
    estimates=np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        s=rng.integers(ns,size=ns)
        e=rng.integers(ne,size=ne)
        estimates[i]=x[np.ix_(s,e)].mean()
    low,high=np.quantile(estimates,[.025,.975])
    return {'mean':float(x.mean()),'ci_low':float(low),'ci_high':float(high),
            'n_seeds':ns,'n_environments':ne,'n_bootstrap':int(n_bootstrap),
            'seed':int(seed),'unit':'crossed_seed_environment','confidence':.95}
