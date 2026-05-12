import json
import os
import shutil
import hashlib
import time

from safetensors.torch import load_file, save_file
from llava.model.qwen_token_utils import normalize_qwen_config_dict


EXTRACT_DONE_FILE = ".extract_complete"
EXTRACT_LOCK_FILE = ".extract_lock"


def is_qwen3vl_checkpoint(path):
    try:
        with open(os.path.join(path, 'config.json')) as f:
            cfg = json.load(f)
        return cfg.get('model_type') == 'qwen3_vl'
    except Exception:
        return False


def is_llava_checkpoint(path):
    try:
        with open(os.path.join(path, 'config.json')) as f:
            cfg = json.load(f)
        return cfg.get('model_type', '').startswith('llava_')
    except Exception:
        return False


def get_extracted_path(vl_path):
    h = hashlib.md5(os.path.abspath(vl_path).encode()).hexdigest()[:8]
    return os.path.join(os.path.dirname(vl_path), f'.qwen3_llm_extracted_{h}')


def is_extracted_llm_ready(output_path):
    if not os.path.exists(os.path.join(output_path, EXTRACT_DONE_FILE)):
        return False
    return (
        os.path.exists(os.path.join(output_path, "model.safetensors"))
        or os.path.exists(os.path.join(output_path, "model.safetensors.index.json"))
    )


def ensure_extracted_llm_from_qwen3vl(vl_path, timeout=7200, poll_interval=2):
    output_path = get_extracted_path(vl_path)
    if is_extracted_llm_ready(output_path):
        return output_path

    os.makedirs(output_path, exist_ok=True)
    lock_path = os.path.join(output_path, EXTRACT_LOCK_FILE)
    start = time.time()
    owns_lock = False
    while not owns_lock:
        if is_extracted_llm_ready(output_path):
            return output_path
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"pid={os.getpid()} time={time.time()}\n")
            owns_lock = True
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for Qwen3-VL LLM extraction: {output_path}")
            time.sleep(poll_interval)

    try:
        if not is_extracted_llm_ready(output_path):
            extract_llm_from_qwen3vl(vl_path, output_path)
        return output_path
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def extract_llm_from_qwen3vl(vl_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(vl_path, 'config.json')) as f:
        vl_config = json.load(f)

    text_config = vl_config['text_config'].copy()
    text_config['model_type'] = 'qwen3'
    text_config['architectures'] = ['Qwen3ForCausalLM']
    text_config.pop('dtype', None)
    normalize_qwen_config_dict(text_config)

    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(text_config, f, indent=2, ensure_ascii=False)

    weight_file = os.path.join(vl_path, 'model.safetensors')
    if os.path.exists(weight_file):
        all_weights = load_file(weight_file)
        llm_weights = {}
        prefix = 'model.language_model.'
        for k, v in all_weights.items():
            if k.startswith(prefix):
                llm_weights['model.' + k[len(prefix):]] = v
            elif k == 'lm_head.weight':
                llm_weights[k] = v
        save_file(llm_weights, os.path.join(output_path, 'model.safetensors'))
    else:
        sharded = sorted(
            [f for f in os.listdir(vl_path) if f.startswith('model-') and f.endswith('.safetensors')]
        )
        for shard_name in sharded:
            all_weights = load_file(os.path.join(vl_path, shard_name))
            llm_weights = {}
            prefix = 'model.language_model.'
            for k, v in all_weights.items():
                if k.startswith(prefix):
                    llm_weights['model.' + k[len(prefix):]] = v
                elif k == 'lm_head.weight':
                    llm_weights[k] = v
            save_file(llm_weights, os.path.join(output_path, shard_name))
        index_path = os.path.join(vl_path, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            new_index = {'weight_map': {}}
            for k, v in index['weight_map'].items():
                if k.startswith(prefix):
                    new_index['weight_map']['model.' + k[len(prefix):]] = v
                elif k == 'lm_head.weight':
                    new_index['weight_map'][k] = v
            new_index['metadata'] = index.get('metadata', {})
            with open(os.path.join(output_path, 'model.safetensors.index.json'), 'w') as f:
                json.dump(new_index, f, indent=2, ensure_ascii=False)

    for fname in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                  'generation_config.json', 'chat_template.json']:
        src = os.path.join(vl_path, fname)
        if os.path.exists(src):
            dst = os.path.join(output_path, fname)
            shutil.copy2(src, dst)
            if fname == 'generation_config.json':
                with open(dst, 'r', encoding='utf-8') as f:
                    generation_config = json.load(f)
                normalize_qwen_config_dict(text_config, generation_config)
                with open(dst, 'w', encoding='utf-8') as f:
                    json.dump(generation_config, f, indent=2, ensure_ascii=False)

    generation_dst = os.path.join(output_path, 'generation_config.json')
    if not os.path.exists(generation_dst):
        _, generation_config = normalize_qwen_config_dict(text_config, {})
        with open(generation_dst, 'w', encoding='utf-8') as f:
            json.dump(generation_config, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_path, EXTRACT_DONE_FILE), 'w', encoding='utf-8') as f:
        f.write(f"source={os.path.abspath(vl_path)}\n")

    return output_path
