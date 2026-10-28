"""Retain environmental counts; binomial interval is a reference only."""
import numpy as np
from scipy.stats import beta


def _count(x,name):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(x)


def ber_interval(errors,bits,confidence=.95):
    errors=_count(errors,'errors');bits=_count(bits,'bits')
    if bits<1 or errors>bits or not 0<confidence<1:raise ValueError('Invalid BER counts/confidence')
    alpha=1-confidence
    low=0. if errors==0 else float(beta.ppf(alpha/2,errors,bits-errors+1))
    high=1. if errors==bits else float(beta.ppf(1-alpha/2,errors+1,bits-errors))
    return {'low':low,'high':high,'confidence':confidence,'method':'two_sided_Clopper_Pearson_reference',
            'limitation':'Correlated bits require environment-group uncertainty analysis; this is not a substitute.'}


class BERCounter:
    def __init__(self,min_errors=200,max_bits=10000000,min_environments=1):
        self.min_errors=_count(min_errors,'min_errors');self.max_bits=_count(max_bits,'max_bits')
        self.min_environments=_count(min_environments,'min_environments')
        if self.min_errors<1 or self.max_bits<1 or self.min_environments<1:
            raise ValueError('Positive stopping budgets and environment coverage required')
        self.environments={}

    def add(self,environment_id,errors,bits):
        errors=_count(errors,'errors');bits=_count(bits,'bits')
        if bits<1 or errors>bits or not isinstance(environment_id,str) or not environment_id:
            raise ValueError('Invalid environment bit/error counts')
        if self.should_stop:raise ValueError('Counter already stopped')
        if self.bits+bits>self.max_bits:raise ValueError('Batch exceeds frozen max_bits; truncate before simulation')
        previous=self.environments.get(environment_id,{'errors':0,'bits':0})
        self.environments[environment_id]={'errors':previous['errors']+errors,'bits':previous['bits']+bits}

    @property
    def errors(self):return sum(x['errors'] for x in self.environments.values())

    @property
    def bits(self):return sum(x['bits'] for x in self.environments.values())

    @property
    def should_stop(self):
        return (self.errors>=self.min_errors and len(self.environments)>=self.min_environments) or self.bits>=self.max_bits

    def result(self):
        covered=len(self.environments)>=self.min_environments
        reason='min_errors' if self.errors>=self.min_errors and covered else ('max_bits' if self.bits>=self.max_bits else 'running')
        return {'errors':self.errors,'bits':self.bits,'ber':self.errors/self.bits if self.bits else None,
                'stop_reason':reason,'n_environments':len(self.environments),
                'required_environments':self.min_environments,'coverage_complete':covered,
                'environment_counts':dict(self.environments),
                'interval':ber_interval(self.errors,self.bits) if self.bits else None}
