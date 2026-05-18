import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Relation Level Emotion Classification")
    parser.add_argument('--json_path', type=str, required=True, help="关系特征描述JSON路径")
    parser.add_argument('--output_path', type=str, default="results_relation.json")
    parser.add_argument('--local_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--use_description", action="store_true", help="是否先生成关系描述再预测情感")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def extract_emotion_confidence(response):
    try:
        clean_resp = response.replace('*', '')
        match = re.search(r"Emotion:\s*(positive|negative)\s*;\s*Confidence:\s*(\d*\.?\d+)", clean_resp, re.IGNORECASE)
        if match:
            return match.group(1).lower(), float(match.group(2))
        
        text_lower = response.lower()
        conf_match = re.search(r"(\d\.\d+)", text_lower)
        conf = float(conf_match.group(1)) if conf_match else 0.50
        
        if 'negative' in text_lower and 'positive' not in text_lower: 
            return 'negative', conf
        return "positive", conf
    except Exception:
        return "positive", 0.50

def run_local_vlm_batch(model, processor, image_paths, prompts, max_tokens=64):
    try:
        images = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            img.thumbnail((512, 512))
            background = Image.new('RGB', (512, 512), (0, 0, 0))
            offset_x = (512 - img.size[0]) // 2
            offset_y = (512 - img.size[1]) // 2
            background.paste(img, (offset_x, offset_y))
            images.append(background)
            
        texts = [
            processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        
        processor.tokenizer.padding_side = "left"
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=False,
                use_cache=True
            )
            
        trimmed_ids = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)]
        results = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        return [r.strip() for r in results]
    except Exception as e:
        print(f"\n[推理异常]: {e}")
        return ["Emotion: positive; Confidence: 0.50"] * len(prompts)

def get_object_names(filepath):
    """提取关联对象名称"""
    try:
        raw_parts = os.path.basename(filepath).split("nearby_")[1].rsplit(".", 1)[0].split("_")
        obj_parts = [part for part in raw_parts if not part.isdigit()]
        return " and ".join(obj_parts) if obj_parts else "objects"
    except:
        return "objects"

def main():
    args = parse_args()
    set_seed()

    if not os.path.exists(args.json_path):
        print(f"错误: 找不到输入特征文件 {args.json_path}")
        return

    with open(args.json_path, 'r') as f:
        full_data = json.load(f)
    
    all_image_paths = [p for p in full_data.keys() if os.path.exists(p)]
    if not all_image_paths:
        print("错误: JSON 文件中提供的图片路径在本地均不存在。")
        return
        
    print(f"有效测试样本数量: {len(all_image_paths)}")

    print("正在加载本地模型...")
    processor = AutoProcessor.from_pretrained(args.local_model_path, trust_remote_code=True, max_pixels=512*512)
    processor.tokenizer.padding_side = "left"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True
    ).eval()

    progress_file = "checkpoint_rel_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_results = json.load(f)
        print(f"已加载历史进度，恢复已处理的 {len(processed_results)} 个样本。")
    else:
        processed_results = {}

    tasks = [p for p in all_image_paths if p not in processed_results]

    if tasks:
        print(f"开始关系层级情感分析 (Batch Size: {args.batch_size})")
        for i in tqdm(range(0, len(tasks), args.batch_size), desc="推理进度"):
            batch_paths = tasks[i:i+args.batch_size]
            
            descriptions = [""] * len(batch_paths)
            if args.use_description:
                rel_prompts = []
                for p in batch_paths:
                    obj_str = get_object_names(p)
                    rel_prompts.append(f"As an emotional psychologist, analyze the relationship between the {obj_str} depicted within the green bounding box. Provide a concise description of how these objects relate to each other. Output format: Description: xxx")
                
                res_desc = run_local_vlm_batch(model, processor, batch_paths, rel_prompts, max_tokens=128)
                descriptions = [r.split("Description: ")[1] if "Description: " in r else r for r in res_desc]
            
            emo_prompts = []
            for p, d in zip(batch_paths, descriptions):
                prompt = (
                    f"As an emotional psychologist, carefully analyze the image. "
                    f"{f'Consider this relationship description: {d}. ' if d else ''}"
                    f"Determine the underlying emotion is Negative (sadness, anger, etc.) "
                    f"or Positive (happiness, joy, etc.), and assign a confidence score (a float from 0 to 1). "
                    f"Follow this exact output format and the emotion must only be 'positive' or 'negative': "
                    f"Emotion: xxx; Confidence: x.xx"
                )
                emo_prompts.append(prompt)
                
            responses = run_local_vlm_batch(model, processor, batch_paths, emo_prompts, max_tokens=64)
            
            for path, resp in zip(batch_paths, responses):
                emotion, conf = extract_emotion_confidence(resp)
                processed_results[path] = {
                    "prediction": emotion,
                    "confidence": conf
                }
                
            if (i // args.batch_size) % 5 == 0:
                with open(progress_file, 'w') as f:
                    json.dump(processed_results, f)
                    
        with open(progress_file, 'w') as f:
            json.dump(processed_results, f)

    with open(args.output_path, 'w') as f:
        json.dump(processed_results, f, indent=4)
        
    print(f"分析完成，结果已保存至 {args.output_path}")

if __name__ == "__main__":
    main()
