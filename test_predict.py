import json
import torch
from PIL import Image
import os
import argparse
from transformers import AutoImageProcessor

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict(model_path, image_file, prompt, conv_mode="qwen_2_centerline_coord", output_file=None):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    device = "cuda" if torch.cuda.is_available() else "npu" if hasattr(torch, 'npu') and torch.npu.is_available() else "cpu"
    tokenizer, model, _, context_len = load_pretrained_model(model_path, None, model_name, device_map={"": device}, device=device)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map={"": device})
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    image = Image.open(image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(device),
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    if output_file:
        result = {"image": image_file, "prediction": outputs}
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        try:
            from PIL import ImageDraw
            
            text = outputs.strip()
            if '\n' in text:
                text = text.split('\n')[0]
            
            pred_data = json.loads(text)
            if not isinstance(pred_data, list):
                pred_data = [pred_data]
            
            draw = ImageDraw.Draw(image)
            colors = ['red', 'blue', 'green', 'yellow']
            
            for i, line in enumerate(pred_data):
                if isinstance(line, dict):
                    color = colors[i % len(colors)]
                    points = line.get('points', [])
                    if len(points) >= 2:
                        for j in range(len(points) - 1):
                            draw.line([tuple(points[j]), tuple(points[j+1])], fill=color, width=3)
            
            vis_file = output_file.replace('.json', '_vis.png')
            image.save(vis_file)
            print(f"Visualization saved to {vis_file}")
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        print(f"Result saved to {output_file}")
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Extract centerline coordinates from the image. Return as JSON.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()
    
    result = predict(args.model_path, args.image_file, args.prompt, output_file=args.output)
    print("Prediction:", result)