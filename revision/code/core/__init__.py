"""Shared data boundaries and physical-scale evaluation, contract v1.0."""
from .arrays import complex_to_ri, ri_to_complex, window_indices, build_windows
from .normalization import Normalizer, fit_normalizer
from .metrics import prediction_metrics, aggregate_metrics
from .statistics import paired_bootstrap

__all__=['complex_to_ri','ri_to_complex','window_indices','build_windows',
         'Normalizer','fit_normalizer','prediction_metrics','aggregate_metrics',
         'paired_bootstrap']
