import torch
import torch.nn as nn


class VisualLayerFusion(nn.Module):
    """
    Fuse multiple ViT hidden layers into one visual feature stream.

    Inputs are same-shape patch features from one vision tower:
      [B, N, C] for each selected layer.

    The fused output keeps [B, N, C], so it can feed the existing mm_projector.
    """
    def __init__(self, hidden_size, num_layers, fusion_type="mean"):
        super().__init__()
        self.fusion_type = str(fusion_type or "mean").lower()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        if self.fusion_type in {"learned_weighted", "weighted", "softmax_weighted"}:
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        elif self.fusion_type in {"mean", "sum"}:
            self.register_parameter("layer_weights", None)
        else:
            raise ValueError(
                "Unsupported vision layer fusion type "
                f"{fusion_type!r}; expected mean, sum, or learned_weighted."
            )

    def forward(self, layer_features):
        if len(layer_features) != len(self.layer_norms):
            raise ValueError(
                f"Expected {len(self.layer_norms)} layer features, got {len(layer_features)}."
            )
        normalized = [
            norm(features)
            for norm, features in zip(self.layer_norms, layer_features)
        ]
        stacked = torch.stack(normalized, dim=0)
        if self.fusion_type == "sum":
            return stacked.sum(dim=0)
        if self.layer_weights is not None:
            weights = torch.softmax(self.layer_weights, dim=0).to(dtype=stacked.dtype)
            return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return stacked.mean(dim=0)
