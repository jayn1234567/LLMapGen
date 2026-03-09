import torch
import torch.nn as nn

from .adapters.vision_adapter import VisionAdapter
from .encoders.pv_encoder import PVEncoder
from .encoders.satellite_encoder import SatelliteEncoder
from .llm.map_llm import MapLLM


class UniMapGenPaper(nn.Module):
    """
    Paper-aligned scaffold:
    - BEV encoder: DINOv2 family
    - Decoder: Qwen2.5 causal LM
    - Multi-modal fusion: visual/text prompt prefix embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        bev_encoder_name: str = "facebook/dinov2-large",
        llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        local_files_only: bool = False,
        use_fallback: bool = False,
        use_pv: bool = False,
        pv_cnn_channels=(64, 128, 256),
        pv_memory_tokens_hw=(2, 4),
        use_text_prompt: bool = False,
        num_prompt_types: int = 4,
        d_model_fallback: int = 256,
        bev_token_hw=(8, 8),
        bev_patch_size: int = 14,
        bev_drop_cls_token: bool = True,
        bev_normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.use_pv = bool(use_pv)
        self.use_text_prompt = bool(use_text_prompt)
        self.sat_encoder = SatelliteEncoder(
            model_name=bev_encoder_name,
            local_files_only=bool(local_files_only),
            use_fallback=bool(use_fallback),
            fallback_channels=tuple(pv_cnn_channels),
            fallback_hw=tuple(bev_token_hw),
            fallback_dim=int(d_model_fallback),
            out_hw=tuple(bev_token_hw),
            patch_size=int(bev_patch_size),
            drop_cls_token=bool(bev_drop_cls_token),
            normalize_input=bool(bev_normalize_input),
        )
        self.llm = MapLLM(
            llm_name=llm_name,
            vocab_size=int(vocab_size),
            pad_id=self.pad_id,
            local_files_only=bool(local_files_only),
            use_fallback=bool(use_fallback),
            fallback_dim=int(d_model_fallback),
        )
        self.hidden_size = int(self.llm.hidden_size)
        self.sat_adapter = VisionAdapter(in_dim=int(self.sat_encoder.hidden_size), out_dim=self.hidden_size)
        if self.use_pv:
            self.pv_encoder = PVEncoder(
                d_model=self.hidden_size,
                cnn_channels=tuple(pv_cnn_channels),
                memory_tokens_hw=tuple(pv_memory_tokens_hw),
            )
        else:
            self.pv_encoder = None

        if self.use_text_prompt:
            self.prompt_type_emb = nn.Embedding(int(num_prompt_types), self.hidden_size)
        else:
            self.prompt_type_emb = None

    def forward(
        self,
        image: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        pv_images: torch.Tensor = None,
        prompt_types: torch.Tensor = None,
        prompt_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        b = decoder_input_ids.shape[0]
        prefix_list = []
        prefix_mask_list = []

        bev_raw = self.sat_encoder(image)
        bev = self.sat_adapter(bev_raw)
        prefix_list.append(bev)
        prefix_mask_list.append(torch.ones((b, bev.shape[1]), device=image.device, dtype=torch.bool))

        if self.use_pv and pv_images is not None:
            pv_mem = self.pv_encoder(pv_images)
            prefix_list.append(pv_mem)
            prefix_mask_list.append(torch.ones((b, pv_mem.shape[1]), device=image.device, dtype=torch.bool))

        if self.use_text_prompt and prompt_types is not None:
            p = self.prompt_type_emb(prompt_types.long()).unsqueeze(1)
            prefix_list.append(p)
            prefix_mask_list.append(torch.ones((b, 1), device=image.device, dtype=torch.bool))

        if self.use_text_prompt and prompt_tokens is not None:
            p_tok = prompt_tokens.long()
            p_tok_emb = self.llm.input_embeddings(p_tok)
            p_tok_mask = p_tok.ne(self.pad_id)
            prefix_list.append(p_tok_emb)
            prefix_mask_list.append(p_tok_mask)

        memory = torch.cat(prefix_list, dim=1)
        memory_mask = ~torch.cat(prefix_mask_list, dim=1)
        return self.llm(memory=memory, decoder_input_ids=decoder_input_ids, memory_key_padding_mask=memory_mask)

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
                image=image,
                decoder_input_ids=out,
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
