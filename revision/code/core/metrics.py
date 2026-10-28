"""Physical CSI metrics with explicit undefined/perfect cases; no error floor."""
import numpy as np


def prediction_metrics(prediction_complex,target_complex):
    pred=np.asarray(prediction_complex)
    target=np.asarray(target_complex)
    if (pred.shape!=target.shape or not pred.size or not np.iscomplexobj(pred)
            or not np.iscomplexobj(target) or not np.isfinite(pred).all()
            or not np.isfinite(target).all()):
        raise ValueError('Prediction and target must be finite same-shape complex arrays')
    pred=pred.astype(np.complex128,copy=False)
    target=target.astype(np.complex128,copy=False)
    try:
        with np.errstate(over='raise',invalid='raise'):
            error=float(np.sum(np.abs(pred-target)**2,dtype=np.float64))
            truth=float(np.sum(np.abs(target)**2,dtype=np.float64))
            power=float(np.sum(np.abs(pred)**2,dtype=np.float64))
    except FloatingPointError as exc:
        raise ValueError('Metric energy overflow; prediction is numerically invalid') from exc
    if not np.isfinite([error,truth,power]).all():
        raise ValueError('Nonfinite derived metric energy')
    if error==0 and np.any(pred!=target):
        raise ValueError('Nonzero residual energy underflow; not a perfect prediction')
    if (truth==0 and np.any(target!=0)) or (power==0 and np.any(pred!=0)):
        raise ValueError('Metric energy underflow for nonzero channel')
    result={'error_energy':error,'target_energy':truth,'prediction_energy':power,
            'n_elements':target.size,'nmse':np.nan,'nmse_db':np.nan,
            'correlation':np.nan,'status':'ok'}
    if truth==0:
        result['status']='zero_target_energy'
        return result
    nmse=error/truth
    if not np.isfinite(nmse) or (nmse==0 and error>0):
        raise ValueError('NMSE ratio overflow or underflow')
    result.update(nmse=nmse,nmse_db=float(10*np.log10(nmse)) if nmse>0 else -np.inf)
    if power==0:
        result['status']='zero_prediction_energy'
    else:
        # Only roundoff outside the mathematical [0,1] correlation range is clipped.
        correlation=abs(np.vdot((target/np.sqrt(truth)).ravel(),(pred/np.sqrt(power)).ravel()))
        result['correlation']=float(np.clip(correlation,0,1))
        if error==0:result['status']='perfect_prediction'
    return result


def aggregate_metrics(environment_metrics):
    values=list(environment_metrics)
    if not values:
        raise ValueError('No environments to aggregate')
    invalid=sum(not np.isfinite(x['nmse']) for x in values)
    result={'n_environments':len(values),'n_invalid':invalid,
            'mean_environment_nmse_db':np.nan,'pooled_nmse_db':np.nan}
    if invalid:
        return result
    result['mean_environment_nmse_db']=float(np.mean([x['nmse_db'] for x in values]))
    error=sum(x['error_energy'] for x in values)
    truth=sum(x['target_energy'] for x in values)
    result['pooled_nmse_db']=float(10*np.log10(error/truth)) if error else -np.inf
    return result
