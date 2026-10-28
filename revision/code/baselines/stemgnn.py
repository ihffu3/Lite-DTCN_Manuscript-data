"""Per-sample graph/spectral STEM-GNN adaptation for normalized CSI snapshots.

The module preserves the local exp14a block design while removing all data,
training, plotting, checkpoint and output-file side effects.  Its graph is
learned independently for each history window to satisfy the frozen batch
isolation contract.
"""

from copy import deepcopy
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


_CONFIGS = {
    "legacy_adaptation": {
        "id": "legacy_adaptation",
        "feature_dim": 16,
        "chebyshev_order": 4,
        "blocks": 2,
        "multi_layer": 5,
        "dropout": 0.5,
        "optimizer": "RMSprop",
        "lr": 1e-4,
        "batch_size": 50,
    },
    "legacy_adaptation_lr3e4": {
        "id": "legacy_adaptation_lr3e4",
        "feature_dim": 16,
        "chebyshev_order": 4,
        "blocks": 2,
        "multi_layer": 5,
        "dropout": 0.5,
        "optimizer": "RMSprop",
        "lr": 3e-4,
        "batch_size": 50,
    },
}


def _resolve_config(config):
    if config is None:
        key = "legacy_adaptation"
    elif isinstance(config, str):
        key = config
    elif isinstance(config, Mapping):
        key = config.get("id", "legacy_adaptation")
        if key not in _CONFIGS:
            raise ValueError(f"unknown STEM-GNN configuration: {key!r}")
        expected = _CONFIGS[key]
        unknown = set(config) - set(expected)
        mismatched = {
            name: (config[name], expected[name])
            for name in set(config) & set(expected)
            if config[name] != expected[name]
        }
        if unknown or mismatched:
            raise ValueError(
                f"STEM-GNN config must match the frozen specification; "
                f"unknown={sorted(unknown)}, mismatched={mismatched}"
            )
    else:
        raise TypeError("config must be None, a frozen config id, or a mapping")
    try:
        return deepcopy(_CONFIGS[key])
    except KeyError as exc:
        raise ValueError(f"unknown STEM-GNN configuration: {key!r}") from exc


class GatedLinearUnit(nn.Module):
    """Two affine projections followed by the legacy sigmoid gate."""

    def __init__(self, input_width, output_width):
        super().__init__()
        self.left = nn.Linear(input_width, output_width)
        self.right = nn.Linear(input_width, output_width)

    def forward(self, value):
        return self.left(value) * torch.sigmoid(self.right(value))


class _SpectralGraphBlock(nn.Module):
    """FFT/real-imaginary GLU block with the local exp14a spectral layout."""

    def __init__(self, history_length, feature_dim, chebyshev_order, multi_layer, index):
        super().__init__()
        self.history_length = history_length
        self.feature_dim = feature_dim
        self.chebyshev_order = chebyshev_order
        self.multi_layer = multi_layer
        self.index = index

        input_width = chebyshev_order * history_length
        self.spectral_width = chebyshev_order * multi_layer * history_length
        self.graph_width = multi_layer * history_length

        def make_stream():
            return nn.ModuleList(
                [
                    GatedLinearUnit(input_width, self.spectral_width),
                    GatedLinearUnit(self.spectral_width, self.spectral_width),
                    GatedLinearUnit(self.spectral_width, self.spectral_width),
                ]
            )

        self.real_glus = make_stream()
        self.imag_glus = make_stream()
        self.graph_weight = nn.Parameter(
            torch.empty(chebyshev_order, self.graph_width, self.graph_width)
        )
        nn.init.xavier_normal_(self.graph_weight)

        self.forecast = nn.Linear(self.graph_width, self.graph_width)
        self.forecast_result = nn.Linear(self.graph_width, history_length)
        # exp14a registers this shortcut on both blocks; block 1 never consumes it.
        # Keep the dormant second-block parameters for a faithful parameter count.
        self.backcast_shortcut = nn.Linear(history_length, history_length)
        if index == 0:
            # The legacy stack backcasts only from its first block.
            self.backcast = nn.Linear(self.graph_width, history_length)

    def _spectral_sequence(self, graph_spectral):
        # [B,D,K,N,L] -> [B*D,N,K*L] so spectral weights are shared over nodes.
        batch, features, orders, nodes, length = graph_spectral.shape
        merged = graph_spectral.reshape(batch * features, orders, nodes, length)
        transformed = torch.fft.fft(merged, dim=-1)
        real = transformed.real.permute(0, 2, 1, 3).reshape(
            batch * features, nodes, orders * length
        )
        imag = transformed.imag.permute(0, 2, 1, 3).reshape(
            batch * features, nodes, orders * length
        )
        for real_glu, imag_glu in zip(self.real_glus, self.imag_glus):
            real = real_glu(real)
            imag = imag_glu(imag)
        real = real.reshape(
            batch * features, nodes, orders, self.multi_layer * length
        ).permute(0, 2, 1, 3)
        imag = imag.reshape(
            batch * features, nodes, orders, self.multi_layer * length
        ).permute(0, 2, 1, 3)
        restored = torch.fft.ifft(torch.complex(real, imag), dim=-1).real
        return restored.reshape(
            batch, features, orders, nodes, self.multi_layer * length
        )

    def forward(self, value, chebyshev):
        # The graph basis is batched: no graph is averaged across CSI windows.
        graph_spectral = torch.einsum("bknm,bdmt->bdknt", chebyshev, value)
        spectral_temporal = self._spectral_sequence(graph_spectral)
        graph_convolved = torch.einsum(
            "bdkns,ksr->bdknr", spectral_temporal, self.graph_weight
        ).sum(dim=2)

        forecast_source = torch.sigmoid(self.forecast(graph_convolved))
        forecast = self.forecast_result(forecast_source)
        if self.index == 0:
            backcast = torch.sigmoid(
                self.backcast(graph_convolved) - self.backcast_shortcut(value)
            )
        else:
            backcast = None
        return forecast, backcast


class STEMGNN(nn.Module):
    """STEM-GNN CSI adapter with a per-example latent graph.

    Input is normalized real/imag CSI ``[B,L,Nt,Nf,2]``; output is the direct
    normalized RI snapshot ``[B,Nt,Nf,2]``.  The node axis denotes BS Tx
    antennas for the new MISO topology.
    """

    def __init__(
        self,
        history_length=10,
        n_tx=8,
        n_freq=32,
        config=None,
    ):
        super().__init__()
        self.config = _resolve_config(config)
        self.history_length = int(history_length)
        self.n_tx = int(n_tx)
        self.n_freq = int(n_freq)
        self.feature_dim = int(self.config["feature_dim"])
        self.chebyshev_order = int(self.config["chebyshev_order"])
        self.multi_layer = int(self.config["multi_layer"])
        self.dropout_rate = float(self.config["dropout"])
        if self.history_length < 1 or self.n_tx < 1 or self.n_freq < 1:
            raise ValueError("history_length, n_tx and n_freq must be positive")

        raw_node_width = 2 * self.n_freq
        self.node_encoder = nn.Linear(raw_node_width, self.feature_dim)
        # Nt nodes are the GRU sequence; each node receives its own L history values.
        self.correlation_gru = nn.GRU(
            input_size=self.history_length,
            hidden_size=self.n_tx,
            batch_first=True,
        )
        self.weight_key = nn.Parameter(torch.empty(self.n_tx, 1))
        self.weight_query = nn.Parameter(torch.empty(self.n_tx, 1))
        nn.init.xavier_uniform_(self.weight_key, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query, gain=1.414)
        self.blocks = nn.ModuleList(
            [
                _SpectralGraphBlock(
                    self.history_length,
                    self.feature_dim,
                    self.chebyshev_order,
                    self.multi_layer,
                    index,
                )
                for index in range(self.config["blocks"])
            ]
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(self.history_length, self.history_length),
            nn.LeakyReLU(0.2),
            nn.Linear(self.history_length, 1),
        )
        self.node_decoder = nn.Linear(self.feature_dim, raw_node_width)
        self.optimizer_config = {
            "name": self.config["optimizer"],
            "lr": self.config["lr"],
            "batch_size": self.config["batch_size"],
            "eps": 1e-8,
            "scheduler": {
                "name": "ExponentialLR",
                "step_every_epochs": 5,
                "gamma": 0.5,
            },
            "max_epochs": 200,
            "early_stopping_patience": 30,
        }

    def apply_edge_dropout(self, probabilities, keep_mask=None):
        """Drop graph edges in training; restore any completely dropped row.

        ``keep_mask`` is a deterministic testing hook; normal forward calls let
        the module sample the edge mask itself.
        """
        if not self.training:
            return probabilities
        if keep_mask is None:
            keep_mask = torch.rand_like(probabilities) >= self.dropout_rate
        if keep_mask.shape != probabilities.shape:
            raise ValueError("edge keep mask must match the adjacency shape")
        dropped = probabilities * keep_mask.to(dtype=probabilities.dtype)
        has_edge = dropped.sum(dim=-1, keepdim=True) > 0
        return torch.where(has_edge, dropped, probabilities)

    def correlation_graph(self, scalar_series, edge_keep_mask=None):
        """Return per-sample [B,4,N,N] basis and symmetric adjacency.

        ``scalar_series`` has the temporal-major shape [B,L,N].  The T0 basis
        follows this local adaptation's historical zero-matrix convention.
        """
        if scalar_series.ndim != 3:
            raise ValueError("scalar_series must have shape [B,L,Nt]")
        batch, length, nodes = scalar_series.shape
        if (length, nodes) != (self.history_length, self.n_tx):
            raise ValueError(
                f"scalar_series trailing shape must be {(self.history_length, self.n_tx)}"
            )
        node_series = scalar_series.transpose(1, 2).contiguous()
        hidden, _ = self.correlation_gru(node_series)
        key = torch.matmul(hidden, self.weight_key)
        query = torch.matmul(hidden, self.weight_query)
        score = F.leaky_relu(
            key + query.transpose(1, 2), negative_slope=0.2
        )
        probabilities = torch.softmax(score, dim=-1)
        if edge_keep_mask is None:
            adjacency = self.apply_edge_dropout(probabilities)
        else:
            adjacency = self.apply_edge_dropout(probabilities, keep_mask=edge_keep_mask)

        symmetric = 0.5 * (adjacency + adjacency.transpose(1, 2))
        degree = symmetric.sum(dim=-1).clamp_min(torch.finfo(symmetric.dtype).tiny)
        inv_sqrt = torch.rsqrt(degree)
        identity = torch.eye(nodes, dtype=symmetric.dtype, device=symmetric.device)
        normalized_adjacency = (
            inv_sqrt.unsqueeze(-1) * symmetric * inv_sqrt.unsqueeze(-2)
        )
        laplacian = identity.unsqueeze(0) - normalized_adjacency

        zero = torch.zeros_like(laplacian)
        first = laplacian
        second = 2.0 * torch.bmm(laplacian, first) - zero
        third = 2.0 * torch.bmm(laplacian, second) - first
        basis = torch.stack((zero, first, second, third), dim=1)
        return basis, symmetric

    def forward(self, history, conditions=None):
        del conditions  # Dedicated baseline; conditions do not enter this model.
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise ValueError("history must be a tensor with shape [B,L,Nt,Nf,2]")
        expected = (self.history_length, self.n_tx, self.n_freq, 2)
        if tuple(history.shape[1:]) != expected:
            raise ValueError(f"history trailing shape must be {expected}")

        batch = history.shape[0]
        encoded = self.node_encoder(
            history.reshape(batch, self.history_length, self.n_tx, 2 * self.n_freq)
        )
        chebyshev, _ = self.correlation_graph(encoded.mean(dim=-1))
        block_input = encoded.permute(0, 3, 2, 1).contiguous()
        forecasts = []
        for block in self.blocks:
            forecast, backcast = block(block_input, chebyshev)
            forecasts.append(forecast)
            if backcast is not None:
                block_input = backcast

        combined = torch.stack(forecasts, dim=0).sum(dim=0)
        direct = self.forecast_head(combined).squeeze(-1).transpose(1, 2)
        decoded = self.node_decoder(direct)
        return decoded.reshape(batch, self.n_tx, self.n_freq, 2)


def build_stemgnn(config=None, *, history_length=10, n_tx=8, n_freq=32):
    """Construct one of the two predeclared STEM-GNN training configurations."""
    return STEMGNN(
        history_length=history_length,
        n_tx=n_tx,
        n_freq=n_freq,
        config=config,
    )


def stemgnn_parameter_count(
    history_length=10,
    n_tx=8,
    n_freq=32,
    feature_dim=16,
    chebyshev_order=4,
    blocks=2,
    multi_layer=5,
):
    """Analytical trainable-parameter count for the frozen local architecture."""
    length = int(history_length)
    nodes = int(n_tx)
    raw_width = 2 * int(n_freq)
    feature = int(feature_dim)
    orders = int(chebyshev_order)
    depth = int(multi_layer)
    graph_width = depth * length
    input_width = orders * length
    spectral_width = orders * depth * length

    def linear_parameters(inputs, outputs):
        return inputs * outputs + outputs

    def glu_parameters(inputs, outputs):
        return 2 * linear_parameters(inputs, outputs)

    stream_parameters = (
        glu_parameters(input_width, spectral_width)
        + 2 * glu_parameters(spectral_width, spectral_width)
    )
    stem_block_parameters = (
        2 * stream_parameters
        + orders * graph_width * graph_width
        + linear_parameters(graph_width, graph_width)
        + linear_parameters(graph_width, length)
    )
    backcast_parameters = linear_parameters(graph_width, length) + blocks * linear_parameters(
        length, length
    )
    gru_parameters = (
        3 * nodes * length + 3 * nodes * nodes + 6 * nodes
    )
    total = (
        linear_parameters(raw_width, feature)
        + linear_parameters(feature, raw_width)
        + gru_parameters
        + 2 * nodes
        + blocks * stem_block_parameters
        + backcast_parameters
        + linear_parameters(length, length)
        + linear_parameters(length, 1)
    )
    return int(total)


def complexity(model):
    """Return exact trainable parameter count and the analytical architecture count."""
    actual = sum(parameter.numel() for parameter in model.parameters())
    analytical = stemgnn_parameter_count(
        history_length=model.history_length,
        n_tx=model.n_tx,
        n_freq=model.n_freq,
        feature_dim=model.feature_dim,
        chebyshev_order=model.chebyshev_order,
        blocks=len(model.blocks),
        multi_layer=model.multi_layer,
    )
    return {
        "parameters": actual,
        "analytical_parameters": analytical,
        "history_length": model.history_length,
        "n_tx": model.n_tx,
        "n_freq": model.n_freq,
        "graph_tokens": model.n_tx,
        "spectral_blocks": len(model.blocks),
        "measured_latency": False,
    }
