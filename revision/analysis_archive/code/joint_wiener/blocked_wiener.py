"""Zero-intercept complex Wiener heads coupling antennas and frequency blocks."""
import numpy as np


def blocks(history, target=None, block_width=4):
    h = np.asarray(history, dtype=np.complex128)
    if h.ndim != 4 or not np.iscomplexobj(history) or not np.isfinite(h).all():
        raise ValueError('Expected finite complex [batch,history,antenna,frequency]')
    batch, length, antennas, frequencies = h.shape
    if block_width < 1 or frequencies % block_width:
        raise ValueError('Frequency blocks must partition all subcarriers')
    if target is not None:
        target = np.asarray(target, dtype=np.complex128)
        if target.shape != (batch, antennas, frequencies) or not np.isfinite(target).all():
            raise ValueError('Target shape/values differ from history')
    result = []
    for start in range(0, frequencies, block_width):
        x = h[..., start:start + block_width].reshape(batch, length * antennas * block_width)
        y = None if target is None else target[..., start:start + block_width].reshape(batch, antennas * block_width)
        result.append((x, y))
    return result


def statistics(history, target, block_width=4):
    return [(x.conj().T @ x, x.conj().T @ y) for x, y in blocks(history, target, block_width)]


def solve_heads(sufficient_statistics, ridge):
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Relative ridge must be finite and positive')
    heads = []
    for gram, cross in sufficient_statistics:
        gram = (gram + gram.conj().T) / 2
        scale = float(np.trace(gram).real / len(gram))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Training input power is not positive and finite')
        head = np.linalg.solve(gram + ridge * scale * np.eye(len(gram)), cross)
        if not np.isfinite(head).all():
            raise ValueError('Nonfinite fitted head')
        heads.append(head)
    return heads


class BlockedWiener:
    def __init__(self, heads, *, history_length=10, n_tx=8, n_freq=32, block_width=4):
        self.shape = (history_length, n_tx, n_freq)
        self.block_width = block_width
        self.heads = [np.asarray(w, dtype=np.complex128) for w in heads]
        expected = (history_length * n_tx * block_width, n_tx * block_width)
        if len(self.heads) != n_freq // block_width or any(w.shape != expected or not np.isfinite(w).all() for w in self.heads):
            raise ValueError('Fitted heads do not match the declared block architecture')

    def __call__(self, history):
        h = np.asarray(history)
        if h.shape[1:] != self.shape:
            raise ValueError('Prediction history differs from the fitted architecture')
        prediction = np.empty((len(h), *self.shape[1:]), dtype=np.complex128)
        for index, ((x, _), w) in enumerate(zip(blocks(h, block_width=self.block_width), self.heads)):
            start = index * self.block_width
            prediction[..., start:start + self.block_width] = (x @ w).reshape(len(h), self.shape[1], self.block_width)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite forecast; no rows may be omitted')
        return prediction
