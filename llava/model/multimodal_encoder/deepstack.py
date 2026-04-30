import torch
import torch.nn as nn


class DeepStack(nn.Module):
    """
    Fuses multi-level ViT features as introduced in Qwen3-VL.

    Instead of using only the last layer's output, DeepStack selects features
    from multiple intermediate ViT layers, applies layer-wise normalization,
    and fuses them via learned weights.

    Args:
        hidden_size: feature dimension of each ViT layer (e.g. 1024 for DINOv2-large)
        num_selected_layers: number of layers selected for fusion
        out_hidden_size: if not None, adds a final linear projection to this dim.
                         Default None means output dim == hidden_size.
    """
    def __init__(self, hidden_size, num_selected_layers, out_hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_selected_layers = num_selected_layers

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_selected_layers)
        ])
        self.layer_weights = nn.Parameter(torch.ones(num_selected_layers) / num_selected_layers)

        if out_hidden_size is not None and out_hidden_size != hidden_size:
            self.output_proj = nn.Linear(hidden_size, out_hidden_size)
        else:
            self.output_proj = None

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: list of tensors, each [B, N, hidden_size]
                                N = num_patches (excluding CLS token)

        Returns:
            fused: [B, N, hidden_size] or [B, N, out_hidden_size]
        """
        outputs = []
        for i, hs in enumerate(hidden_states_list):
            hs = self.layer_norms[i](hs)
            outputs.append(hs * self.layer_weights[i])
        fused = torch.stack(outputs, dim=0).sum(dim=0)
        if self.output_proj is not None:
            fused = self.output_proj(fused)
        return fused


def build_deepstack(config, num_selected_layers):
    """
    Factory to build a DeepStack module from config.

    config should have:
        mm_hidden_size (hidden_size) — ViT feature dim
        Optionally deepstack_out_hidden_size for output projection
    """
    hidden_size = getattr(config, 'mm_hidden_size', None)
    if hidden_size is None:
        hidden_size = config.hidden_size
    out_hidden_size = getattr(config, 'deepstack_out_hidden_size', None)
    return DeepStack(hidden_size, num_selected_layers, out_hidden_size)
