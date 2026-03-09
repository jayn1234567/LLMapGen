from typing import Tuple

import torch
import torch.nn as nn


class SimpleBEVEncoder(nn.Module):
    def __init__(self, in_ch: int, channels, d_model: int, out_hw: Tuple[int, int]) -> None:
        super().__init__()
        layers = []
        prev = in_ch
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, c, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                ]
            )
            prev = c
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Conv2d(prev, d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(out_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        x = self.pool(x)  # [B, D, Hm, Wm]
        b, d, h, w = x.shape
        x = x.view(b, d, h * w).transpose(1, 2).contiguous()  # [B, M, D]
        return x


class UniMapGenV1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        cnn_channels=(64, 128, 256),
        memory_tokens_hw=(8, 8),
        use_pv: bool = False,
        pv_cnn_channels=(64, 128, 256),
        pv_memory_tokens_hw=(2, 4),
        use_text_prompt: bool = False,
        num_prompt_types: int = 4,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_pv = bool(use_pv)
        self.use_text_prompt = bool(use_text_prompt)
        self.encoder = SimpleBEVEncoder(3, cnn_channels, d_model=d_model, out_hw=tuple(memory_tokens_hw))
        if self.use_pv:
            self.pv_encoder = SimpleBEVEncoder(3, pv_cnn_channels, d_model=d_model, out_hw=tuple(pv_memory_tokens_hw))
        else:
            self.pv_encoder = None
        if self.use_text_prompt:
            self.prompt_emb = nn.Embedding(int(num_prompt_types), d_model)
        else:
            self.prompt_emb = None
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(2048, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        image: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        pv_images: torch.Tensor = None,
        prompt_types: torch.Tensor = None,
        prompt_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        memory = self.encoder(image)  # [B, M, D]
        memory_pad_mask = torch.zeros((memory.shape[0], memory.shape[1]), device=memory.device, dtype=torch.bool)
        if self.use_pv and pv_images is not None:
            # v1 pv_images shape: [B, L, C, H, W], currently L=1.
            b, l, c, h, w = pv_images.shape
            pv_flat = pv_images.view(b * l, c, h, w)
            pv_mem = self.pv_encoder(pv_flat)
            pv_mem = pv_mem.view(b, l * pv_mem.shape[1], pv_mem.shape[2])
            memory = torch.cat([memory, pv_mem], dim=1)
            pv_mask = torch.zeros((pv_mem.shape[0], pv_mem.shape[1]), device=memory.device, dtype=torch.bool)
            memory_pad_mask = torch.cat([memory_pad_mask, pv_mask], dim=1)
        if self.use_text_prompt and prompt_types is not None:
            p = self.prompt_emb(prompt_types.long()).unsqueeze(1)  # [B, 1, D]
            memory = torch.cat([memory, p], dim=1)
            p_mask = torch.zeros((p.shape[0], p.shape[1]), device=memory.device, dtype=torch.bool)
            memory_pad_mask = torch.cat([memory_pad_mask, p_mask], dim=1)
        if self.use_text_prompt and prompt_tokens is not None:
            p_tok = prompt_tokens.long()
            p_tok_emb = self.tok_emb(p_tok)
            p_pos = torch.arange(p_tok.shape[1], device=decoder_input_ids.device).unsqueeze(0).expand(p_tok.shape[0], -1)
            p_tok_emb = p_tok_emb + self.pos_emb(p_pos)
            memory = torch.cat([memory, p_tok_emb], dim=1)
            p_tok_mask = p_tok.eq(self.pad_id)
            memory_pad_mask = torch.cat([memory_pad_mask, p_tok_mask], dim=1)
        b, t = decoder_input_ids.shape
        pos = torch.arange(t, device=decoder_input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(decoder_input_ids) + self.pos_emb(pos)

        # Causal mask for autoregressive decoding.
        causal = torch.triu(torch.ones(t, t, device=decoder_input_ids.device, dtype=torch.bool), diagonal=1)
        pad_mask = decoder_input_ids.eq(self.pad_id)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        out = self.norm(out)
        return self.head(out)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 512,
        pv_images: torch.Tensor = None,
        prompt_ids: torch.Tensor = None,
        prompt_types: torch.Tensor = None,
        prompt_tokens: torch.Tensor = None,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        device = image.device
        if prompt_ids is not None:
            out = prompt_ids.to(device)
        else:
            out = torch.full((image.shape[0], 1), int(bos_id), dtype=torch.long, device=device)
        finished = torch.zeros((image.shape[0],), dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            logits = self.forward(
                image,
                out,
                pv_images=pv_images,
                prompt_types=prompt_types,
                prompt_tokens=prompt_tokens,
            )
            next_logits = logits[:, -1, :]
            if float(temperature) > 1e-6 and float(temperature) != 1.0:
                next_logits = next_logits / float(temperature)

            if float(repetition_penalty) > 1.0:
                for b in range(out.shape[0]):
                    seen = torch.unique(out[b])
                    next_logits[b, seen] = next_logits[b, seen] / float(repetition_penalty)

            if step < int(min_new_tokens):
                next_logits[:, int(eos_id)] = -1e9

            if int(top_k) > 1:
                k = min(int(top_k), next_logits.shape[-1])
                vals, idxs = torch.topk(next_logits, k=k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                next_tok = idxs.gather(1, pick)
            else:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
            finished = finished | next_tok.squeeze(1).eq(eos_id)
            if bool(finished.all()):
                break
        return out
