import os
import time
from typing import Dict, Sequence

import torch
import torch.nn as nn

from unimapgen.models.encoders.satellite_encoder import SatelliteEncoder
from unimapgen.models.encoders.pv_encoder import PVEncoder
from unimapgen.models.hf_utils import resolve_hf_snapshot_path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from transformers import AutoModelForCausalLM
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc


class QwenSatelliteMapGenerator(nn.Module):
    def __init__(
        self,
        dino_model_path: str,
        qwen_model_path: str,
        vocab_size: int,
        allowed_map_token_ids: Sequence[int],
        map_eos_token_id: int,
        local_files_only: bool = True,
        freeze_satellite: bool = True,
        freeze_llm: bool = False,
        sat_token_hw=(8, 8),
        sat_patch_size: int = 14,
        sat_drop_cls_token: bool = True,
        sat_normalize_input: bool = True,
        use_pv: bool = False,
        pv_cnn_channels=(32, 64, 128),
        pv_memory_tokens_hw=(2, 4),
        pv_num_frames: int = 1,
        pv_pool_frames_per_camera: bool = False,
        pv_use_camera_embedding: bool = False,
        pv_max_camera_groups: int = 16,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "QwenSatelliteMapGenerator failed to import transformers.AutoModelForCausalLM. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR!r}"
            )

        self.dino_model_path = resolve_hf_snapshot_path(dino_model_path)
        self.qwen_model_path = resolve_hf_snapshot_path(qwen_model_path)
        print(f"[Init] Satellite backbone path: {self.dino_model_path}", flush=True)
        self.sat_encoder = SatelliteEncoder(
            model_name=self.dino_model_path,
            local_files_only=bool(local_files_only),
            use_fallback=False,
            out_hw=tuple(sat_token_hw),
            patch_size=int(sat_patch_size),
            drop_cls_token=bool(sat_drop_cls_token),
            normalize_input=bool(sat_normalize_input),
        )
        print(f"[Init] Qwen model path: {self.qwen_model_path}", flush=True)
        llm_load_start = time.time()
        llm_load_kwargs = dict(
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_path,
                low_cpu_mem_usage=True,
                **llm_load_kwargs,
            )
        except TypeError:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_path,
                **llm_load_kwargs,
            )
        print(
            f"[Init] Qwen LLM loaded in {time.time() - llm_load_start:.1f}s "
            f"(dtype={getattr(self.llm, 'dtype', 'unknown')})",
            flush=True,
        )
        self.llm.resize_token_embeddings(int(vocab_size))
        self.hidden_size = int(self.llm.config.hidden_size)
        self.sat_proj = nn.Linear(int(self.sat_encoder.hidden_size), self.hidden_size)
        self.use_pv = bool(use_pv)
        if self.use_pv:
            self.pv_encoder = PVEncoder(
                d_model=self.hidden_size,
                cnn_channels=tuple(pv_cnn_channels),
                memory_tokens_hw=tuple(pv_memory_tokens_hw),
                num_frames_per_camera=int(pv_num_frames),
                pool_frames_per_camera=bool(pv_pool_frames_per_camera),
                use_camera_embedding=bool(pv_use_camera_embedding),
                max_camera_groups=int(pv_max_camera_groups),
            )
            self.pv_proj = nn.Identity()
        else:
            self.pv_encoder = None
            self.pv_proj = None
        self.llm_embed_dtype = self.llm.get_input_embeddings().weight.dtype
        self.map_eos_token_id = int(map_eos_token_id)
        self.register_buffer(
            "allowed_map_token_ids",
            torch.tensor(sorted(set(int(x) for x in allowed_map_token_ids)), dtype=torch.long),
            persistent=False,
        )

        if bool(freeze_satellite):
            for p in self.sat_encoder.parameters():
                p.requires_grad = False
        if bool(freeze_llm):
            for p in self.llm.parameters():
                p.requires_grad = False
            for p in self.sat_proj.parameters():
                p.requires_grad = True

        if bool(gradient_checkpointing) and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm.config, "use_cache"):
                self.llm.config.use_cache = False

    @torch.no_grad()
    def semantic_initialize_new_embeddings(self, qwen_map_tokenizer) -> Dict[str, int]:
        input_emb = self.llm.get_input_embeddings()
        output_emb = self.llm.get_output_embeddings()
        if input_emb is None:
            return {"initialized": 0, "skipped": 0}
        input_weight = input_emb.weight.data
        output_weight = None
        if output_emb is not None and hasattr(output_emb, "weight") and output_emb.weight.shape == input_weight.shape:
            output_weight = output_emb.weight.data

        initialized = 0
        skipped = 0
        for spec in qwen_map_tokenizer.semantic_init_specs():
            target_id = int(spec["qwen_id"])
            phrase_ids = qwen_map_tokenizer.tokenizer.encode(str(spec["text"]), add_special_tokens=False)
            phrase_ids = [int(x) for x in phrase_ids if 0 <= int(x) < int(qwen_map_tokenizer.base_vocab_size)]
            if not phrase_ids:
                skipped += 1
                continue
            mean_vec = input_weight[phrase_ids].mean(dim=0)
            input_weight[target_id].copy_(mean_vec)
            if output_weight is not None:
                output_weight[target_id].copy_(mean_vec)
            initialized += 1

        return {"initialized": initialized, "skipped": skipped}

    def encode_prefix(
        self,
        image: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        pv_images: torch.Tensor = None,
        state_input_ids: torch.Tensor = None,
        state_attention_mask: torch.Tensor = None,
    ):
        sat_requires_grad = any(p.requires_grad for p in self.sat_encoder.parameters())
        with torch.set_grad_enabled(bool(sat_requires_grad)):
            sat_tokens = self.sat_encoder(image)
        sat_tokens = self.sat_proj(sat_tokens).to(dtype=self.llm_embed_dtype)
        sat_mask = torch.ones(
            (image.shape[0], sat_tokens.shape[1]),
            device=image.device,
            dtype=torch.long,
        )
        if self.use_pv and pv_images is not None and self.pv_encoder is not None:
            pv_tokens = self.pv_encoder(pv_images).to(dtype=self.llm_embed_dtype)
            pv_tokens = self.pv_proj(pv_tokens)
            pv_mask = torch.ones(
                (image.shape[0], pv_tokens.shape[1]),
                device=image.device,
                dtype=torch.long,
            )
        else:
            pv_tokens = sat_tokens.new_zeros((image.shape[0], 0, self.hidden_size))
            pv_mask = sat_mask.new_zeros((image.shape[0], 0))

        if prompt_input_ids.shape[1] > 0:
            prompt_embeds = self.llm.get_input_embeddings()(prompt_input_ids)
            prompt_mask = prompt_attention_mask.long()
        else:
            prompt_embeds = sat_tokens.new_zeros((image.shape[0], 0, self.hidden_size))
            prompt_mask = sat_mask.new_zeros((image.shape[0], 0))
        if state_input_ids is not None and state_input_ids.shape[1] > 0:
            state_embeds = self.llm.get_input_embeddings()(state_input_ids).to(dtype=self.llm_embed_dtype)
            state_mask = state_attention_mask.long()
        else:
            state_embeds = sat_tokens.new_zeros((image.shape[0], 0, self.hidden_size))
            state_mask = sat_mask.new_zeros((image.shape[0], 0))

        prefix_embeds = torch.cat([sat_tokens, pv_tokens, prompt_embeds, state_embeds], dim=1)
        prefix_mask = torch.cat([sat_mask, pv_mask, prompt_mask, state_mask], dim=1)
        return prefix_embeds, prefix_mask

    def forward(
        self,
        image: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        pv_images: torch.Tensor,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        map_input_ids: torch.Tensor,
        map_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prefix_embeds, prefix_mask = self.encode_prefix(
            image=image,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            pv_images=pv_images,
            state_input_ids=state_input_ids,
            state_attention_mask=state_attention_mask,
        )
        map_embeds = self.llm.get_input_embeddings()(map_input_ids).to(dtype=self.llm_embed_dtype)
        inputs_embeds = torch.cat([prefix_embeds, map_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, map_attention_mask.long()], dim=1)

        prefix_labels = torch.full(
            (image.shape[0], prefix_embeds.shape[1]),
            -100,
            device=image.device,
            dtype=torch.long,
        )
        map_labels = map_input_ids.masked_fill(map_attention_mask.eq(0), -100)
        labels = torch.cat([prefix_labels, map_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "prefix_length": torch.tensor(prefix_embeds.shape[1], device=image.device, dtype=torch.long),
        }

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        pv_images: torch.Tensor = None,
        state_input_ids: torch.Tensor = None,
        state_attention_mask: torch.Tensor = None,
        max_new_tokens: int = 256,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        grammar_helper=None,
        grammar_min_points_per_line: int = 2,
        grammar_max_lines: int = None,
        return_token_meta: bool = False,
    ) -> torch.Tensor:
        self.eval()
        device = image.device
        prefix_embeds, prefix_mask = self.encode_prefix(
            image=image,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            pv_images=pv_images,
            state_input_ids=state_input_ids,
            state_attention_mask=state_attention_mask,
        )
        generated = torch.zeros((image.shape[0], 0), dtype=torch.long, device=device)
        allowed = self.allowed_map_token_ids.to(device=device)
        allowed_list = [int(x) for x in allowed.tolist()]
        finished = torch.zeros((image.shape[0],), dtype=torch.bool, device=device)
        token_meta = [[] for _ in range(image.shape[0])] if bool(return_token_meta) else None
        category_qwen_ids = []
        if grammar_helper is not None and hasattr(grammar_helper, "category_qwen_ids"):
            category_qwen_ids = [int(x) for x in getattr(grammar_helper, "category_qwen_ids", [])]
        category_tensor = (
            torch.tensor(category_qwen_ids, dtype=torch.long, device=device)
            if category_qwen_ids
            else None
        )

        for step in range(int(max_new_tokens)):
            if generated.shape[1] > 0:
                gen_embeds = self.llm.get_input_embeddings()(generated).to(dtype=self.llm_embed_dtype)
                inputs_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
                gen_mask = torch.ones_like(generated, dtype=torch.long)
                attention_mask = torch.cat([prefix_mask, gen_mask], dim=1)
            else:
                inputs_embeds = prefix_embeds
                attention_mask = prefix_mask

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            raw_next_logits = outputs.logits[:, -1, :]
            next_logits = raw_next_logits.clone()
            if float(repetition_penalty) > 1.0 and generated.shape[1] > 0:
                for b in range(generated.shape[0]):
                    seen = torch.unique(generated[b])
                    vals = next_logits[b, seen]
                    penalized = torch.where(vals > 0, vals / float(repetition_penalty), vals * float(repetition_penalty))
                    next_logits[b, seen] = penalized

            restricted = next_logits.index_select(dim=1, index=allowed)
            if float(temperature) > 1e-6 and float(temperature) != 1.0:
                restricted = restricted / float(temperature)
            if step < int(min_new_tokens):
                eos_loc = allowed.eq(int(self.map_eos_token_id)).nonzero(as_tuple=False)
                if eos_loc.numel() > 0:
                    restricted[:, int(eos_loc[0].item())] = -1e9
            if grammar_helper is not None:
                for b in range(generated.shape[0]):
                    valid_qwen_ids = grammar_helper.valid_next_qwen_map_ids(
                        generated_qwen_ids=generated[b].tolist(),
                        min_points_per_line=int(grammar_min_points_per_line),
                        max_lines=grammar_max_lines,
                    )
                    if not valid_qwen_ids:
                        continue
                    valid_set = set(int(x) for x in valid_qwen_ids)
                    for j, tok_id in enumerate(allowed_list):
                        if int(tok_id) not in valid_set:
                            restricted[b, j] = -1e9

            if int(top_k) > 1:
                k = min(int(top_k), restricted.shape[-1])
                vals, idxs = torch.topk(restricted, k=k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                chosen_idx = idxs.gather(1, pick)
                next_tok = allowed[chosen_idx]
            else:
                chosen_idx = restricted.argmax(dim=-1, keepdim=True)
                next_tok = allowed[chosen_idx]

            if token_meta is not None:
                token_probs = torch.softmax(restricted, dim=-1)
                chosen_prob = token_probs.gather(1, chosen_idx).squeeze(1)
                category_probs = None
                if category_tensor is not None and category_tensor.numel() > 0:
                    category_logits = raw_next_logits.index_select(dim=1, index=category_tensor)
                    category_probs = torch.softmax(category_logits, dim=-1)
                for b in range(image.shape[0]):
                    tok_id = int(next_tok[b, 0].item())
                    if bool(finished[b].item()):
                        token_meta[b].append(
                            {
                                "qwen_id": tok_id,
                                "token_score": 1.0,
                                "category_score": None,
                            }
                        )
                        continue
                    cat_score = None
                    if category_probs is not None:
                        cat_match = category_tensor.eq(tok_id).nonzero(as_tuple=False)
                        if cat_match.numel() > 0:
                            cat_score = float(category_probs[b, int(cat_match[0].item())].item())
                    token_meta[b].append(
                        {
                            "qwen_id": tok_id,
                            "token_score": float(chosen_prob[b].item()),
                            "category_score": cat_score,
                        }
                    )

            next_tok = next_tok.masked_fill(finished.unsqueeze(1), int(self.map_eos_token_id))
            generated = torch.cat([generated, next_tok], dim=1)
            finished = finished | next_tok.squeeze(1).eq(int(self.map_eos_token_id))
            if bool(finished.all()):
                break
        if token_meta is not None:
            return generated, token_meta
        return generated
