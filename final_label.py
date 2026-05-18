import json
import os
import math
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def load_json_data(file_paths):
    """加载 JSON 数据文件"""
    data_from_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_from_files.append(data)
        else:
            print(f"[Warning] File not found: {file_path}")
    return data_from_files

def calculate_distribution_entropy(distribution):
    """计算情感分布的信息熵"""
    entropy = 0
    for prob in distribution.values():
        if prob > 0:
            # 引入极小值 1e-9 防止 log(0) 异常
            entropy -= prob * math.log2(prob + 1e-9)
    return entropy

def combine_data(object_data, relation_data, global_data):
    """基于信息熵加权的多层级数据融合"""
    combined_results = {}
    true_labels = {}
    predictions = {}

    # 1. 初始化 Global 层级数据
    for item in global_data.get("global_results", []):
        filename = item.get("filename", "")
        prefix = os.path.basename(filename).split('.png')[0].split('.jpg')[0]
        true_labels[prefix] = 0 if item.get("label") == "positive" else 1

        confidence = item.get("confidence", 0.5)
        global_distribution = {
            "pos": confidence if item.get("prediction") == "positive" else 1 - confidence,
            "neg": 1 - confidence if item.get("prediction") == "positive" else confidence
        }
        global_entropy = calculate_distribution_entropy(global_distribution)
        combined_results[prefix] = {
            "global": {
                "distribution": global_distribution,
                "weight": 1, 
                "entropy": global_entropy
            },
            "object": None,
            "relation": None
        }

    # 2. 注入 Object 层级数据
    if "prefix_level" in object_data:
        for prefix, data in object_data["prefix_level"].items():
            if prefix in combined_results:
                dist = data.get("final_distribution")
                if not dist and "pos_sum" in data and data.get("count", 0) > 0:
                    cnt = data["count"]
                    dist = {"pos": data["pos_sum"] / cnt, "neg": data["neg_sum"] / cnt}
                
                if dist:
                    object_entropy = calculate_distribution_entropy(dist)
                    combined_results[prefix]["object"] = {
                        "distribution": dist,
                        "weight": data.get("count", 1), 
                        "entropy": object_entropy
                    }

    # 3. 注入 Relation 层级数据
    if "prefix_level" in relation_data:
        for prefix, data in relation_data["prefix_level"].items():
            if prefix in combined_results:
                dist = data.get("final_distribution")
                if not dist and "pos_sum" in data and data.get("count", 0) > 0:
                    cnt = data["count"]
                    dist = {"pos": data["pos_sum"] / cnt, "neg": data["neg_sum"] / cnt}
                
                if dist:
                    relation_entropy = calculate_distribution_entropy(dist)
                    combined_results[prefix]["relation"] = {
                        "distribution": dist,
                        "weight": data.get("count", 1), 
                        "entropy": relation_entropy
                    }

    # 4. 执行基于信息熵的加权融合计算
    for prefix, data in combined_results.items():
        total_weight = 0
        pos_weighted_sum = 0
        neg_weighted_sum = 0

        for level_name, level_data in data.items():
            if level_data is not None:
                # 权重公式：N_l * (1 - Entropy)
                effective_weight = level_data["weight"] * (1 - level_data["entropy"])
                total_weight += effective_weight
                pos_weighted_sum += effective_weight * level_data["distribution"]["pos"]
                neg_weighted_sum += effective_weight * level_data["distribution"]["neg"]

        if total_weight > 0:
            final_pos = pos_weighted_sum / total_weight
            final_neg = neg_weighted_sum / total_weight
            
            # 归一化处理
            norm = final_pos + final_neg
            if norm > 0:
                final_pos /= norm
                final_neg /= norm
            predictions[prefix] = 0 if final_pos >= final_neg else 1

    # 按照 prefix 排序，确保评估结果可对齐与复现
    aligned_prefixes = sorted([p for p in true_labels if p in predictions])
    y_true_list = [true_labels[p] for p in aligned_prefixes]
    y_pred_list = [predictions[p] for p in aligned_prefixes]
    
    return y_true_list, y_pred_list

def main(args):
    print("[Info] Loading JSON data for mathematical fusion evaluation...")
    
    try:
        with open(args.object_level, "r", encoding="utf-8") as f: 
            object_data = json.load(f)
        with open(args.relation_level, "r", encoding="utf-8") as f: 
            relation_data = json.load(f)
        with open(args.global_level, "r", encoding="utf-8") as f: 
            global_data = json.load(f)
    except FileNotFoundError as e:
        print(f"[Error] Missing necessary input files: {e}")
        return

    valid_true, valid_pred = combine_data(object_data, relation_data, global_data)

    if not valid_true:
        print("[Error] No valid data found to evaluate. Please check data alignment.")
        return

    # 计算评估指标 (0: Positive, 1: Negative)
    total_acc = accuracy_score(valid_true, valid_pred)
    
    p_p = precision_score(valid_true, valid_pred, pos_label=0, zero_division=0)
    p_r = recall_score(valid_true, valid_pred, pos_label=0, zero_division=0)
    p_f1 = f1_score(valid_true, valid_pred, pos_label=0, zero_division=0)
    
    n_p = precision_score(valid_true, valid_pred, pos_label=1, zero_division=0)
    n_r = recall_score(valid_true, valid_pred, pos_label=1, zero_division=0)
    n_f1 = f1_score(valid_true, valid_pred, pos_label=1, zero_division=0)

    # 打印格式化评估报告
    print("\n" + "="*50)
    print("Multi-Level Fusion Evaluation Report")
    print("="*50)
    print(f"{'Overall Accuracy':<25}: {total_acc*100:.2f}%")
    print(f"{'Total Samples Analyzed':<25}: {len(valid_true)}")
    print("-" * 50)
    print("Positive Class (Healthy):")
    print(f"  - Precision            : {p_p*100:.2f}%")
    print(f"  - Recall               : {p_r*100:.2f}%")
    print(f"  - F1-Score             : {p_f1*100:.2f}%")
    print("-" * 50)
    print("Negative Class (Distress):")
    print(f"  - Precision            : {n_p*100:.2f}%")
    print(f"  - Recall               : {n_r*100:.2f}%")
    print(f"  - F1-Score             : {n_f1*100:.2f}%")
    print("="*50)

    # 导出统计数据
    summary = {
        "accuracy": total_acc,
        "sample_count": len(valid_true),
        "positive": {"precision": p_p, "recall": p_r, "f1": p_f1},
        "negative": {"precision": n_p, "recall": n_r, "f1": n_f1}
    }
    
    os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"valid_true": valid_true, "valid_pred": valid_pred, "summary": summary}, f, indent=4)
    print(f"[Info] Evaluation results successfully saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Emotion Framework (Mathematical Fusion Version)")
    parser.add_argument("--object_level", type=str, required=True, help="Path to object-level results JSON")
    parser.add_argument("--relation_level", type=str, required=True, help="Path to relation-level results JSON")
    parser.add_argument("--global_level", type=str, required=True, help="Path to global-level results JSON")
    parser.add_argument("--output_json", type=str, default="final_fusion_results.json", help="Path for output JSON file")
    
    args = parser.parse_args()
    main(args)
