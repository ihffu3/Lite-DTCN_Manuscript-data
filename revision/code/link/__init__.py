"""Controlled block-fading downlink tools; see contract1.2 assumptions."""
from .precoding import make_precoder,spectral_efficiency,beamforming_gain,evaluate_link
from .modem import modulate,demodulate,complex_awgn,simulate_ber_batch
from .counts import ber_interval,BERCounter
