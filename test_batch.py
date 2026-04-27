import json
import torch
from PIL import Image
import os
import glob
from tqdm import tqdm

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict(model_path, image_file, conv_mode="qwen_2_centerline_coord"):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, context_len = load_pretrained_model(model_path, None, model_name, device_map={"": "cuda:0"}, device="cuda:0")
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map={"": "cuda:0"})
    vision_tower.to(device="cuda:0", dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    qs = "Extract centerline coordinates from the image. Return as JSON."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    image = Image.open(image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs, image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/batch_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))[:args.num_samples]
    print(f"Found {len(image_files)} images")
    
    all_results = []
    
    for i, img_file in enumerate(tqdm(image_files)):
        try:
            outputs, image = predict(args.model_path, img_file)
            
            text = outputs.strip()
            
            result = {
                "id": i,
                "image": img_file,
                "prediction": text
            }
            all_results.append(result)
            
            from PIL import ImageDraw
            try:
                if '\n' in text:
                    text = text.split('\n')[0]
                
                pred_data = json.loads(text)
                if not isinstance(pred_data, list):
                    pred_data = [pred_data]
                
                draw = ImageDraw.Draw(image)
                colors = ['red', 'blue', 'green', 'yellow']
                
                for j, line in enumerate(pred_data):
                    if isinstance(line, dict):
                        color = colors[j % len(colors)]
                        points = line.get('points', [])
                        if len(points) >= 2:
                            for k in range(len(points) - 1):
                                draw.line([tuple(points[k]), tuple(points[k+1])], fill=color, width=3)
                
                vis_path = os.path.join(args.output_dir, f"result_{i:03d}_vis.png")
                image.save(vis_path)
            except json.JSONDecodeError as je:
                print(f"JSON error for {img_file}: {je}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    result_file = os.path.join(args.output_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {result_file}")
    print(f"Visualizations saved to {args.output_dir}/")