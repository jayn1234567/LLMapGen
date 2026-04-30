import json
import os
import shutil
import hashlib

from safetensors.torch import load_file, save_file


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


def extract_llm_from_qwen3vl(vl_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(vl_path, 'config.json')) as f:
        vl_config = json.load(f)

    text_config = vl_config['text_config'].copy()
    text_config['model_type'] = 'qwen3'
    text_config['architectures'] = ['Qwen3ForCausalLM']
    text_config.pop('dtype', None)

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
            shutil.copy2(src, os.path.join(output_path, fname))

    return output_path
