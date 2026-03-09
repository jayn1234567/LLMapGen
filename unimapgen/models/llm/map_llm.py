import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None


class MapLLM(nn.Module):
    """
    Wrapper for map generation LLM.
    - Preferred: HuggingFace causal LM (Qwen2.5 family).
    - Fallback: lightweight Transformer decoder.
    """

    def __init__(
        self,
        llm_name: str,
        vocab_size: int,
        pad_id: int,
        local_files_only: bool = False,
        use_fallback: bool = False,
        fallback_dim: int = 256,
        fallback_heads: int = 8,
        fallback_layers: int = 6,
        fallback_ff: int = 1024,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.use_fallback = bool(use_fallback) or (AutoModelForCausalLM is None)

        if not self.use_fallback:
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    trust_remote_code=True,
                    local_files_only=bool(local_files_only),
                )
                self.llm.resize_token_embeddings(vocab_size)
                self.hidden_size = int(self.llm.config.hidden_size)
            except Exception:
                self.use_fallback = True

        if self.use_fallback:
            self.hidden_size = int(fallback_dim)
            self.tok_emb = nn.Embedding(vocab_size, self.hidden_size, padding_idx=self.pad_id)
            self.pos_emb = nn.Embedding(4096, self.hidden_size)
            layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=int(fallback_heads),
                dim_feedforward=int(fallback_ff),
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=int(fallback_layers))
            self.norm = nn.LayerNorm(self.hidden_size)
            self.head = nn.Linear(self.hidden_size, vocab_size)

    def input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.use_fallback:
            b, t = token_ids.shape
            pos = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, t)
            return self.tok_emb(token_ids) + self.pos_emb(pos)
        return self.llm.get_input_embeddings()(token_ids)

    def forward(self, memory: torch.Tensor, decoder_input_ids: torch.Tensor, memory_key_padding_mask=None) -> torch.Tensor:
        if self.use_fallback:
            x = self.input_embeddings(decoder_input_ids)
            t = decoder_input_ids.shape[1]
            causal = torch.triu(torch.ones(t, t, device=decoder_input_ids.device, dtype=torch.bool), diagonal=1)
            pad_mask = decoder_input_ids.eq(self.pad_id)
            out = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            out = self.norm(out)
            return self.head(out)

        tok_emb = self.input_embeddings(decoder_input_ids)
        b = decoder_input_ids.shape[0]
        tok_mask = decoder_input_ids.ne(self.pad_id)
        if memory_key_padding_mask is None:
            mem_mask = torch.ones((b, memory.shape[1]), device=decoder_input_ids.device, dtype=torch.bool)
        else:
            mem_mask = ~memory_key_padding_mask
        inputs_embeds = torch.cat([memory, tok_emb], dim=1)
        attn_mask = torch.cat([mem_mask, tok_mask], dim=1).long()
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
        return out.logits[:, -decoder_input_ids.shape[1] :, :]
