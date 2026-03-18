import os
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc

from unimapgen.models.hf_utils import resolve_hf_snapshot_path


class QwenMapTokenizer:
    def __init__(
        self,
        qwen_model_path: str,
        map_tokenizer,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        if AutoTokenizer is None:
            raise RuntimeError(
                "QwenMapTokenizer failed to import transformers.AutoTokenizer. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR!r}"
            )

        self.map_tokenizer = map_tokenizer
        self.qwen_model_path = resolve_hf_snapshot_path(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        vocab = self.tokenizer.get_vocab()
        self.base_vocab_size = int(len(vocab))
        new_tokens = [tok for tok in self.map_tokenizer.itos if tok not in vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens, special_tokens=False)
        self.new_map_tokens = list(new_tokens)

        self.map_token_to_qwen_id: Dict[str, int] = {}
        self.map_id_to_qwen_id: Dict[int, int] = {}
        self.qwen_id_to_map_id: Dict[int, int] = {}
        for tok, map_id in self.map_tokenizer.stoi.items():
            qwen_id = int(self.tokenizer.convert_tokens_to_ids(tok))
            if qwen_id < 0:
                raise ValueError(f"Map token was not registered in Qwen tokenizer: {tok}")
            self.map_token_to_qwen_id[tok] = qwen_id
            self.map_id_to_qwen_id[int(map_id)] = qwen_id
            self.qwen_id_to_map_id[qwen_id] = int(map_id)

        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.base_eos_token_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else -1
        self.map_bos_token_id = int(self.map_token_to_qwen_id["<bos>"])
        self.map_eos_token_id = int(self.map_token_to_qwen_id["<eos>"])
        self.allowed_map_token_ids = sorted(
            int(v)
            for k, v in self.map_token_to_qwen_id.items()
            if k != "<pad>"
        )
        self.category_qwen_ids = [
            int(self.map_id_to_qwen_id[int(x)])
            for x in self.map_tokenizer.category_ids
            if int(x) in self.map_id_to_qwen_id
        ]

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode_prompt(self, text: str, max_length: int = None) -> List[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if max_length is not None and max_length > 0:
            ids = ids[: int(max_length)]
        return [int(x) for x in ids]

    def encode_map_token_ids(self, map_token_ids: Sequence[int]) -> List[int]:
        out = []
        for idx in map_token_ids:
            tok = self.map_tokenizer.itos[int(idx)]
            out.append(int(self.map_token_to_qwen_id[tok]))
        return out

    def decode_qwen_map_ids_to_custom_ids(self, qwen_ids: Iterable[int]) -> List[int]:
        out = []
        for idx in qwen_ids:
            idx = int(idx)
            if idx in self.qwen_id_to_map_id:
                out.append(int(self.qwen_id_to_map_id[idx]))
        return out

    def valid_next_qwen_map_ids(
        self,
        generated_qwen_ids: Sequence[int],
        min_points_per_line: int = 2,
        max_lines: int = None,
    ) -> List[int]:
        custom_ids = self.decode_qwen_map_ids_to_custom_ids(generated_qwen_ids)
        valid_custom_ids = self.map_tokenizer.valid_next_token_ids(
            prefix_ids=custom_ids,
            min_points_per_line=min_points_per_line,
            max_lines=max_lines,
        )
        return [int(self.map_id_to_qwen_id[int(x)]) for x in valid_custom_ids if int(x) in self.map_id_to_qwen_id]

    def semantic_init_specs(self) -> List[Dict[str, object]]:
        specs: List[Dict[str, object]] = []
        for tok in self.new_map_tokens:
            qwen_id = self.map_token_to_qwen_id.get(tok)
            if qwen_id is None:
                continue
            phrase = self._semantic_init_text(tok)
            if not phrase:
                continue
            specs.append(
                {
                    "token": str(tok),
                    "qwen_id": int(qwen_id),
                    "text": str(phrase),
                }
            )
        return specs

    @staticmethod
    def _semantic_init_text(tok: str) -> str:
        if tok == "<bos>":
            return "begin map"
        if tok == "<eos>":
            return "end map"
        if tok == "<state>":
            return "previous map state"
        if tok == "<txt_xy>":
            return "xy coordinates"
        if tok == "<txt_trace>":
            return "trace points"
        if tok == "<txt_end>":
            return "end text"
        if tok == "<line>":
            return "polyline"
        if tok == "<pts>":
            return "points"
        if tok == "<eol>":
            return "end of polyline"
        if tok.startswith("<cat_") and tok.endswith(">"):
            return tok[len("<cat_") : -1].replace("_", " ")
        if tok.startswith("<lt_") and tok.endswith(">"):
            return tok[len("<lt_") : -1].replace("_", " ")
        if tok.startswith("<s_") and tok.endswith(">"):
            return "start " + tok[len("<s_") : -1].replace("_", " ")
        if tok.startswith("<e_") and tok.endswith(">"):
            return "end " + tok[len("<e_") : -1].replace("_", " ")
        if tok.startswith("<x_") and tok.endswith(">"):
            return "x coordinate " + tok[len("<x_") : -1]
        if tok.startswith("<y_") and tok.endswith(">"):
            return "y coordinate " + tok[len("<y_") : -1]
        if tok.startswith("<a_") and tok.endswith(">"):
            return "angle " + tok[len("<a_") : -1]
        return ""
