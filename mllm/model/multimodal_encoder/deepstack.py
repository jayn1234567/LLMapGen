import torch
import torch.nn as nn


class DeepStackMerger(nn.Module):
    """
    Maps a single ViT layer's patch features to LLM hidden space.

    Each selected ViT layer has its OWN merger — they are NOT shared.
    This is the correct DeepStack architecture from Qwen3-VL:
      - Shallow ViT features → shallow LLM layers (residual addition)
      - Deep ViT features → deeper LLM layers (residual addition)

    Architecture: LayerNorm → Linear → GELU → Linear
    """
    def __init__(self, vit_hidden_size, llm_hidden_size):
        super().__init__()
        self.merger = nn.Sequential(
            nn.LayerNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, vit_hidden_size]  (N = num_patches, CLS removed)
        Returns:
            [B, N, llm_hidden_size]
        """
        return self.merger(x)


def build_deepstack_mergers(vit_hidden_size, llm_hidden_size, num_mergers):
    """
    Build a ModuleList of DeepStackMerger instances — one per selected ViT layer.
    """
    return nn.ModuleList([
        DeepStackMerger(vit_hidden_size, llm_hidden_size)
        for _ in range(num_mergers)
    ])
