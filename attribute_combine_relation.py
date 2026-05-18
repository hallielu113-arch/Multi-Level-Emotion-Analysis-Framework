import argparse
import json
import os
import re
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import warnings
 
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Object Level Emotion Classification")
    parser.add_argument("--json_path", type=str, required=True, help="提取的对象特征JSON文件路径")
    parser.add_argument("--csv_path", type=str, required=True, help="知识库CSV文件路径")
    parser.add_argument("--output_path", type=str, default="results_object.json")
    parser.add_argument("--valid_words", type=str, nargs="+", default=["house", "tree", "person"])
    parser.add_argument("--local_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def load_json_data(json_path):
    if not os.path.exists(json_path): return {}
    with open(json_path, 'r') as f: return json.load(f)

def save_json_data(data, json_path):
    with open(json_path, 'w') as f: json.dump(data, f, indent=4)

def clean_attributes(attrs):
    """清理无效属性描述，保留有效特征"""
    clean_list = []
    for a in attrs:
        s = str(a).strip()
        if len(s) > 40 or "\n" in s or "```" in s or "/" in s:
            continue
        clean_list.append(s)
    
    if not clean_list:
        return ["Position", "Size"]
    return list(dict.fromkeys(clean_list))

def select_instance(json_data, valid_words):
    cleaned_data = {}
    for fp, attrs in json_data.items():
        if any(w in os.path.basename(fp) for w in valid_words):
            cleaned_data[fp] = clean_attributes(attrs)
    return cleaned_data

def extract_emotion_confidence(response):
    text_lower = response.lower()
    conf = 0.50
    conf_match = re.search(r"confidence[*\s:]*(\d*\.?\d+)", text_lower)
    if conf_match: conf = float(conf_match.group(1))

    match = re.search(r"emotion[*\s:]*(positive|negative)", text_lower)
    if match: return match.group(1), conf

    has_pos = "positive" in text_lower
    has_neg = "negative" in text_lower
    if has_neg and not has_pos: return "negative", conf
    if has_pos and not has_neg: return "positive", conf
    if has_pos and has_neg:
        return ("positive" if text_lower.find("positive") < text_lower.find("negative") else "negative"), conf
    return "positive", conf

def get_embedding_batch(client, texts, model="text-embedding-3-large"):
    """批量获取文本向量表示"""
    if not texts:
        return []
    for _ in range(3):
        try:
            res = client.embeddings.create(input=texts, model=model).data
            return [d.embedding for d in sorted(res, key=lambda x: x.index)]
        except Exception as e:
            print(f"Embedding API 异常: {e}")
            time.sleep(2)
    return [[0.0] * 3072] * len(texts)

def extract_emotion_kb_batch(client, csv_data, global_kb_embeddings, descriptions):
    """批量计算描述与知识库的相似度"""
    embs = get_embedding_batch(client, descriptions)
    results = []
    if not embs: return []
    
    sims_batch = cosine_similarity(embs, global_kb_embeddings)
    
    for i in range(len(embs)):
        idx = sims_batch[i].argsort()[-1]
        similarity = sims_batch[i][idx]
        label = json.loads(csv_data.iloc[idx]["extracted_label"])
        pos_score, neg_score = float(label["Positive"].strip('%')), float(label["Negative"].strip('%'))
        total = pos_score + neg_score
        results.append(({"pos": pos_score / total, "neg": neg_score / total}, similarity))
    return results

def combine_emotions(e_llm, e_kb, conf, sim):
    """结合大语言模型和知识库的情感得分"""
    combined_pos_scores, combined_neg_scores = [], []
    for i in range(len(e_llm)):
        ld = {"pos": 1.0 if e_llm[i] == "positive" else 0.0, "neg": 1.0 if e_llm[i] == "negative" else 0.0}
        exp_llm, exp_kb = np.exp(conf[i]), np.exp(sim[i])
        total_exp = exp_llm + exp_kb
        llm_weight, kb_weight = exp_llm / total_exp, exp_kb / total_exp
        combined_pos = (llm_weight * ld["pos"]) + (kb_weight * e_kb[i]["pos"])
        combined_neg = (llm_weight * ld["neg"]) + (kb_weight * e_kb[i]["neg"])
        total = combined_pos + combined_neg
        combined_pos_scores.append(combined_pos / total)
        combined_neg_scores.append(combined_neg / total)
    avg_pos, avg_neg = np.mean(combined_pos_scores), np.mean(combined_neg_scores)
    return avg_pos / (avg_pos + avg_neg), avg_neg / (avg_pos + avg_neg)

def run_local_vlm_batch(model, processor, image_paths, prompts, max_tokens=128):
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
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
            
        trimmed_ids = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)]
        results = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        return [r.strip() for r in results]
    except Exception as e:
        print(f"\n[推理异常]: {e}")
        return ["Emotion: positive; Confidence: 0.50"] * len(prompts)

def main():
    args = parse_args()
    set_seed()

    # 初始化 API 客户端
    gpt_base_url = os.getenv("OPENAI_BASE_URL", "[https://api.openai.com/v1](https://api.openai.com/v1)")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY")
        
    gpt_client = OpenAI(api_key=api_key, base_url=gpt_base_url, timeout=30.0)

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
    print("模型加载完成。")

    print("正在预加载知识库矩阵...")
    csv_data = pd.read_csv(args.csv_path)
    csv_data["head_embedding"] = csv_data["head_embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    global_kb_embeddings = np.vstack(csv_data["head_embedding"].values)

    raw_data = load_json_data(args.json_path)
    data = select_instance(raw_data, set(args.valid_words))
    
    step1_file = "checkpoint_step1_descriptions.json"
    if os.path.exists(step1_file):
        print(f"加载已缓存的描述: {step1_file}")
        data = load_json_data(step1_file)
    else:
        print("\n--- 阶段 1: 生成特征描述 ---")
        step1_tasks = [(img_path, attr, i) for img_path, attrs in data.items() for i, attr in enumerate(attrs)]
        for img_path in data: data[img_path] = {"attributes": data[img_path], "descriptions": [None]*len(data[img_path])}
                
        for i in tqdm(range(0, len(step1_tasks), args.batch_size), desc="生成进度"):
            batch = step1_tasks[i : i + args.batch_size]
            img_paths = [t[0] for t in batch]
            prompts = []
            for t in batch:
                obj_name = os.path.basename(t[0]).split("_")[1] if "_" in os.path.basename(t[0]) else "object"
                prompts.append(f"Acting as a emotional psychologist, provide a concise and complete sentence description of the {obj_name} depicted within the green bounding box in the sketch drawing. Think carefully and the sentence should focus on the following: {t[1]}, and should not involve any emotional words. The output structure must be exactly the following: Description: xxx")
            
            responses = run_local_vlm_batch(model, processor, img_paths, prompts, max_tokens=64)
            for j, t in enumerate(batch):
                img_path, attr, idx = t
                desc = responses[j]
                if "Description: " in desc: desc = desc.split("Description: ")[1]
                data[img_path]["descriptions"][idx] = desc
        save_json_data(data, step1_file)

    print("\n--- 阶段 2: 提取情感分布 ---")
    progress_file = "checkpoint_step2_progress.json"
    processed_data = load_json_data(progress_file)
    
    step2_tasks = []
    for img_path, info in data.items():
        if img_path not in processed_data:
            data[img_path].update({"emotion_llm": [None]*len(info["attributes"]), "emotion_distributions_kb": [None]*len(info["attributes"]), "confidences": [None]*len(info["attributes"]), "similarities": [None]*len(info["attributes"])})
            for i, desc in enumerate(info["descriptions"]):
                step2_tasks.append((img_path, info["attributes"][i], desc, i))
        else:
            data[img_path] = processed_data[img_path]

    if step2_tasks:
        for i in tqdm(range(0, len(step2_tasks), args.batch_size), desc="分析进度"):
            batch = step2_tasks[i : i + args.batch_size]
            img_paths = [t[0] for t in batch]
            prompts = []
            
            for t in batch:
                attr, desc = t[1], t[2]
                prompt = (
                    f"As an emotional psychologist, analyze the following: "
                    f"1. the image, 2. this attribute about the object in the bounding box: {attr} "
                    f"3. this description based on the image and attribute: {desc}, "
                    f"ensuring no details are overlooked in making the diagnosis. "
                    f"Think carefully and determine the underlying emotion is Negative (sadness, anger, etc.) "
                    f"or Positive (happiness, joy, etc.), and assign a confidence score (a float from 0 to 1). "
                    f"Follow this exact output format and the emotion must only be 'positive' or 'negative': "
                    f"Emotion: xxx; Confidence: x.xx"
                )
                prompts.append(prompt)
                
            responses = run_local_vlm_batch(model, processor, img_paths, prompts, max_tokens=32)
            kb_results = extract_emotion_kb_batch(gpt_client, csv_data, global_kb_embeddings, [t[2] for t in batch])
                
            for j, t in enumerate(batch):
                img_path, attr, desc, idx = t
                emotion, conf = extract_emotion_confidence(responses[j])
                dist_kb, sim = kb_results[j]
                
                data[img_path]["emotion_llm"][idx] = emotion
                data[img_path]["confidences"][idx] = conf
                data[img_path]["emotion_distributions_kb"][idx] = dist_kb
                data[img_path]["similarities"][idx] = sim
                
                if all(x is not None for x in data[img_path]["emotion_llm"]):
                    processed_data[img_path] = data[img_path]
                    
            if (i // args.batch_size) % 10 == 0: save_json_data(processed_data, progress_file)
        save_json_data(processed_data, progress_file)
        data = processed_data

    print("\n阶段 3: 汇总各实例分布...")
    for image_path, info in data.items():
        avg_pos, avg_neg = combine_emotions(info["emotion_llm"], info["emotion_distributions_kb"], info["confidences"], info["similarities"])
        data[image_path]["instance_distribution"] = {"pos": avg_pos, "neg": avg_neg}
        
    print("阶段 4 & 5: 计算最终评估指标...")
    prefix_distributions = {}
    for image_path, info in data.items():
        filename = os.path.basename(image_path)
        prefix = re.match(r'(\d+\.\d+)', filename).group(1) if re.match(r'(\d+\.\d+)', filename) else None
        
        if prefix not in prefix_distributions: prefix_distributions[prefix] = {"pos_sum": 0, "neg_sum": 0, "count": 0, "paths": []}
        prefix_distributions[prefix]["pos_sum"] += info["instance_distribution"]["pos"]
        prefix_distributions[prefix]["neg_sum"] += info["instance_distribution"]["neg"]
        prefix_distributions[prefix]["count"] += 1
        prefix_distributions[prefix]["paths"].append(image_path)
        
    valid_true, valid_pred = [], []
    for prefix, info in prefix_distributions.items():
        if not info["paths"]: continue
        avg_pos, avg_neg = info["pos_sum"] / info["count"], info["neg_sum"] / info["count"]
        predicted_label = "positive" if avg_pos >= avg_neg else "negative"
        info["predicted_label"] = predicted_label
        info["final_distribution"] = {"pos": avg_pos, "neg": avg_neg}
        
        # 使用通用逻辑提取真值标签
        truth = "positive" if "0." in info["paths"][0].split('/')[-1] else "negative"
        valid_true.append(truth)
        valid_pred.append(predicted_label)
        
    acc = accuracy_score(valid_true, valid_pred)
    p_prec = precision_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    p_rec = recall_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    p_f1 = f1_score(valid_true, valid_pred, pos_label='positive', zero_division=0)
    n_prec = precision_score(valid_true, valid_pred, pos_label='negative', zero_division=0)
    n_rec = recall_score(valid_true, valid_pred, pos_label='negative', zero_division=0)
    n_f1 = f1_score(valid_true, valid_pred, pos_label='negative', zero_division=0)

    print("\n" + "="*50)
    print("对象层级 (Object Level) 评估结果")
    print("-" * 50)
    print(f"总体准确率 (Accuracy) : {acc:.4f}")
    print(f"正向 F1-Score         : {p_f1:.4f}")
    print(f"负向 F1-Score         : {n_f1:.4f}")
    print(f"负向召回率 (Recall)   : {n_rec:.4f}")
    print("="*50)

    summary = {
        "accuracy": acc, "pos_f1": p_f1, "neg_f1": n_f1, "neg_recall": n_rec,
        "pos_prec": p_prec, "pos_recall": p_rec, "neg_prec": n_prec
    }
    save_json_data({"instance_level": data, "prefix_level": prefix_distributions, "summary": summary}, args.output_path)
    print(f"分析完成，结果已保存至 {args.output_path}")

if __name__ == "__main__":
    main()
