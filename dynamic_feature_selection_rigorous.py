import os
os.environ["OMP_NUM_THREADS"] = "1"

# 环境兼容性修复补丁
import sys
try:
    import transformers.utils.import_utils
    import transformers.utils
    def skip_check(*args, **kwargs): return None
    transformers.utils.import_utils.check_torch_load_is_safe = skip_check
    transformers.utils.check_torch_load_is_safe = skip_check
    import transformers.trainer
    transformers.trainer.check_torch_load_is_safe = skip_check
except Exception:
    pass

import argparse
import re
import gc
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from datasets import Dataset, Image as HFImage
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback 
) 
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RealTimeLogCallback(TrainerCallback):
    """训练过程实时监控回调"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs.get("loss", 0.0)
            reward = logs.get("reward", 0.0)
            kl = logs.get("kl", 0.0)
            print(f"\n[训练状态 - Step {step}] Loss: {loss:.4f} | Reward: {reward:.4f} | KL Divergence: {kl:.6f}")
            
            if kl > 0.15:
                print(f"[警告] KL散度 ({kl:.3f}) 超过设定阈值，已触发学习率衰减机制。")
                optimizer = kwargs.get('optimizer')
                if optimizer:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.9

def predict(model, tokenizer, text, device):
    if not text: return 0.5, 0.5
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=1)
        probs = torch.exp(log_probs)
    return probs[0, 0].item(), probs[0, 1].item()

def highest_frequency_word(words):
    if not words: return ""
    counts = Counter(words)
    max_freq = max(counts.values())
    for word in words:
        if counts[word] == max_freq: return word
    return words[0]

def extract_completion_text(completion) -> str:
    if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    elif isinstance(completion, dict):
        return str(completion.get("content", ""))
    else:
        parts = str(completion).split("\nassistant\n")
        return parts[-1] if len(parts) > 1 else str(completion)

def reward_func(prompts, completions, **kwargs):
    """主奖励计算函数"""
    rewards = []
    grouped_answers = defaultdict(list)
    image_paths = kwargs.get("image_path", [])
    
    banned_emotion_words = ["depress", "sad", "happy", "happi", "cheer", "lonel", "anxi", "fear", "angr", "joy"]
    lazy_template_words = ["curve", "curved", "line", "lines", "cross", "crossed", "shape", "texture", "simple", "normal", "standard", "symmetry", "symmetrical", "outline", "appearance"]
    
    for idx, (path, completion) in enumerate(zip(image_paths, completions)):
        gen_ans = extract_completion_text(completion).strip(" \n\"'*")
        ans_lower = gen_ans.lower()
        
        if not gen_ans or "```" in gen_ans or len(gen_ans) > 25 or len(gen_ans) < 2:
            rewards.append(0.0); continue
            
        if re.search(r'[^\x00-\x7F]', gen_ans):
            rewards.append(0.0); continue
            
        if any(banned in ans_lower for banned in banned_emotion_words):
            rewards.append(0.0); continue
            
        grouped_answers[path].append(gen_ans)
        
        rm_input = f"Description: A drawing showing {gen_ans}."
        pos_score, neg_score = predict(reward_model, reward_tokenizer, rm_input, device)
        
        base_reward = (max(pos_score, neg_score) - 0.5) * 2.0
        
        lazy_penalty = 1.0
        for lazy_word in lazy_template_words:
            if lazy_word in ans_lower:
                lazy_penalty = 0.8
                break
                
        final_score = base_reward * lazy_penalty
        rewards.append(final_score)

    for path, ans in grouped_answers.items():
        if not ans: continue
        best = highest_frequency_word(ans)
        if path in extract_feature_dict:
            if len(extract_feature_dict[path]) < 4: extract_feature_dict[path].append(best)
            else: extract_feature_dict[path][-1] = best

    return rewards

def load_checkpoint(checkpoint_path, model_name, device):
    print("正在加载奖励模型权重...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS", modules_to_save=["score"])
    model = get_peft_model(base_model, lora_config)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = cp.get('model_state_dict', cp)
    new_sd = {}
    expected_keys = model.state_dict().keys()
    for k, v in sd.items():
        nk = k
        if "base_model.model." in nk: nk = nk[nk.find("base_model.model."):]
        if "score.modules_to_save.default" in nk: nk = nk[nk.find("score.modules_to_save.default"):]
        if nk in expected_keys: new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False); model.eval()
    tk = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    return model, tk

def dataset_prepare(extract_feature_dict, data_folder, processor):
    rows = []
    for f in sorted(os.listdir(data_folder)):
        if not f.endswith(".png"): continue
        path = os.path.join(data_folder, f)
        obj_name = os.path.basename(f).split("nearby")[-1].lstrip('_').split('_')[0] if "nearby" in f else "object"
        excl = extract_feature_dict.get(path, ["Position", "Size"])
        text = (f"Analyze the sketch. Focus on '{obj_name}' in green box. Identify ONE highly specific UNIQUE physical detail.\n"
                f"GOOD: 'barred window', 'drooping branches'. BAD: 'curved', 'sad'.\n"
                f"DO NOT output: {', '.join(excl)} or Color. Output 1-3 words short phrase.")
        rows.append({"image_path": path, "text": text})
    ds = Dataset.from_list(rows)
    def render(ex):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["text"]}]}]
        p = processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        return {"prompt": p, "image": ex["image_path"], "image_path": ex["image_path"]}
    return ds.map(render, num_proc=16).cast_column("image", HFImage())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help="奖励模型检查点路径")
    parser.add_argument('--data_folder', type=str, required=True, help="训练图像文件夹路径")
    parser.add_argument('--extract_json_path', type=str, default="extracted_features.json", help="特征提取输出路径")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/GRPO_model", help="模型保存路径")
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="基础大模型路径")
    parser.add_argument('--resume_from', type=str, default=None, help="恢复训练的检查点路径")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(args.extract_json_path):
        with open(args.extract_json_path, 'r') as f: extract_feature_dict = json.load(f)
    else:
        extract_feature_dict = {os.path.join(args.data_folder, f): ["Position", "Size"] for f in os.listdir(args.data_folder) if f.endswith(".png")}
        
    global reward_model, reward_tokenizer
    reward_model, reward_tokenizer = load_checkpoint(args.checkpoint_path, "Qwen/Qwen2.5-7B-Instruct", device)

    avg_feat_len = np.mean([len(v) for v in extract_feature_dict.values()])
    start_round = 0 if avg_feat_len < 2.5 else 1
    
    for k in range(start_round, 2):
        print(f"\n准备开启 Round {k+1} 训练流程...")
        proc = AutoProcessor.from_pretrained(args.base_model, max_pixels=512*512)
        proc.tokenizer.padding_side = "left"
        dataset = dataset_prepare(extract_feature_dict, args.data_folder, proc)

        current_output_dir = f"{args.output_dir}_R{k+1}" if k == 1 else args.output_dir

        training_args = GRPOConfig(
            output_dir=current_output_dir, run_name=f"GRPO_R{k}",
            learning_rate=3e-6, beta=0.2, warmup_ratio=0.15,
            bf16=True, per_device_train_batch_size=8, num_generations=4, 
            max_completion_length=12, num_train_epochs=1, save_steps=50, logging_steps=1,
            report_to="none", gradient_checkpointing=True, 
            save_total_limit=3
        )
        
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="auto"
        )
        base_model.config.use_cache = False 
        base_model.enable_input_require_grads() 
        
        peft_config = LoraConfig(r=16, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], task_type="CAUSAL_LM", lora_dropout=0.05)
        
        if k == 1:
            r2_resume_ckpt = args.resume_from if args.resume_from else get_last_checkpoint(current_output_dir)
            
            if r2_resume_ckpt:
                print(f"检测到历史检查点：{r2_resume_ckpt}，将恢复训练进度。")
                update_model = get_peft_model(base_model, peft_config)
                resume_ckpt = r2_resume_ckpt
            else:
                last_round_ckpt = get_last_checkpoint(args.output_dir)
                if last_round_ckpt:
                    print(f"继承上一阶段模型权重：{last_round_ckpt}")
                    update_model = PeftModel.from_pretrained(base_model, last_round_ckpt, is_trainable=True)
                else:
                    update_model = get_peft_model(base_model, peft_config)
                resume_ckpt = None 
        else:
            update_model = get_peft_model(base_model, peft_config)
            resume_ckpt = args.resume_from if args.resume_from else get_last_checkpoint(args.output_dir)

        trainer = GRPOTrainer(model=update_model, processing_class=proc, reward_funcs=[reward_func],
                             args=training_args, train_dataset=dataset)
        
        trainer.add_callback(RealTimeLogCallback())
        print(f"执行 Round {k+1} 训练...")
        
        trainer.train(resume_from_checkpoint=resume_ckpt)
        
        with open(args.extract_json_path, "w") as f: json.dump(extract_feature_dict, f, indent=4)
        del update_model, trainer, base_model; gc.collect(); torch.cuda.empty_cache()
