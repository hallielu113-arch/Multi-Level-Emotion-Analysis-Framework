import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Global Level Emotion Classification")
    parser.add_argument('--data_folder', type=str, required=True, help="存放调整尺寸后图像的目录")
    parser.add_argument('--output_path', type=str, default="results_global.json")
    parser.add_argument('--local_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="模型路径或HuggingFace Hub名称")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批次大小")
    parser.add_argument("--pos_prefix", type=str, default="0.", help="正向样本文件名前缀")
    parser.add_argument("--neg_prefix", type=str, default="1.", help="负向样本文件名前缀")
    return parser.parse_args()

def extract_emotion_confidence(response):
    """解析模型输出的情感极性与置信度"""
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

def run_local_vlm_batch(model, processor, image_paths, prompts):
    """执行本地视觉语言模型批处理推理"""
    try:
        images = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            # 保持宽高比缩放并填充背景，防止图像形变
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
                max_new_tokens=64, 
                do_sample=False,
                use_cache=True
            )
            
        trimmed_ids = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)]
        results = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        return [r.strip() for r in results]
    except Exception as e:
        print(f"\n[推理异常]: {e}")
        return ["Emotion: positive; Confidence: 0.50"] * len(prompts)

def main():
    args = parse_args()
    set_seed()

    if not os.path.exists(args.data_folder):
        print(f"错误: 找不到指定的数据目录 {args.data_folder}")
        return

    print(f"正在加载模型: {args.local_model_path}")
    processor = AutoProcessor.from_pretrained(args.local_model_path, trust_remote_code=True, max_pixels=512*512)
    processor.tokenizer.padding_side = "left"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True
    ).eval()
    print("模型加载完成。")

    filenames, labels = [], []
    for filename in os.listdir(args.data_folder):
        if filename.endswith((".png", ".jpg")):
            if filename.startswith(args.pos_prefix):
                filenames.append(os.path.join(args.data_folder, filename))
                labels.append("positive")
            elif filename.startswith(args.neg_prefix):
                filenames.append(os.path.join(args.data_folder, filename))
                labels.append("negative")

    print(f"有效测试样本数量: {len(filenames)}")

    prompt = (
        "As an emotional psychologist, analyze the image carefully and focus on the overall visual characteristics. "
        "Determine the underlying emotion is Negative (sadness, anger, etc.) or Positive (happiness, joy, etc.), "
        "and assign a confidence score (a float from 0 to 1, where 0 means no confidence and 1 means full confidence) "
        "indicating certainty in the emotional interpretation. "
        "Follow this exact output format and the emotion must only be 'positive' or 'negative': Emotion: xxx; Confidence: x.xx"
    )

    results = []
    progress_file = "checkpoint_global_progress.json"
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            results = json.load(f)
        processed_paths = {item['filename'] for item in results}
        print(f"已加载历史进度，恢复已处理的 {len(processed_paths)} 个样本。")
    else:
        processed_paths = set()

    tasks = [(f, l) for f, l in zip(filenames, labels) if f not in processed_paths]

    if tasks:
        print(f"开始全局层级情感分析 (Batch Size: {args.batch_size})")
        for i in tqdm(range(0, len(tasks), args.batch_size), desc="推理进度"):
            batch = tasks[i:i+args.batch_size]
            batch_paths = [t[0] for t in batch]
            batch_labels = [t[1] for t in batch]
            batch_prompts = [prompt] * len(batch)
            
            responses = run_local_vlm_batch(model, processor, batch_paths, batch_prompts)
            
            for path, label, resp in zip(batch_paths, batch_labels, responses):
                emotion, conf = extract_emotion_confidence(resp)
                prediction = emotion if emotion in ["positive", "negative"] else "positive"
                
                results.append({
                    "filename": path,
                    "label": label,
                    "prediction": prediction,
                    "confidence": conf,
                    "description": ""
                })
            
            if (i // args.batch_size) % 5 == 0:
                with open(progress_file, 'w') as f:
                    json.dump(results, f)
        
        with open(progress_file, 'w') as f:
            json.dump(results, f)

    valid_true = [item['label'] for item in results]
    valid_pred = [item['prediction'] for item in results]

    acc = accuracy_score(valid_true, valid_pred)
    p_prec = precision_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    p_rec = recall_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    p_f1 = f1_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    n_prec = precision_score(valid_true, valid_pred, pos_label='negative', zero_division=0)
    n_rec = recall_score(valid_true, valid_pred, pos_label='negative', zero_division=0)
    n_f1 = f1_score(valid_true, valid_pred, pos_label='negative', zero_division=0)

    print("\n" + "="*50)
    print("全局层级 (Global Level) 评估结果")
    print("-" * 50)
    print(f"总体准确率 (Accuracy) : {acc:.4f}")
    print(f"正向 F1-Score         : {p_f1:.4f}")
    print(f"负向 F1-Score         : {n_f1:.4f}")
    print(f"负向召回率 (Recall)   : {n_rec:.4f}")
    print("="*50)

    summary = {
        "total_images": len(valid_true),
        "accuracy_positive": acc,
        "precision_positive": p_prec,
        "recall_positive": p_rec,
        "f1_positive": p_f1,
        "accuracy_negative": acc,
        "precision_negative": n_prec,
        "recall_negative": n_rec,
        "f1_negative": n_f1,
    }
    
    with open(args.output_path, 'w') as f:
        json.dump({"global_results": results, "summary": summary}, f, indent=4)
    
    if os.path.exists(progress_file):
        os.remove(progress_file)
        
    print(f"分析完成，结果已保存至 {args.output_path}")

if __name__ == "__main__":
    main()
