# Multi-Level Emotion Analysis Framework

This repository contains the official implementation of an advanced Multi-Level Emotion Analysis Framework.

## Repository Structure

The core modules of this repository are split into the following files:

- `attribute_combine_global.py`: Analyzes the overall composition of images (Global Level) to determine baseline emotional distribution using Vision-Language Models.
    
- `attribute_combine_object.py`: Extracts and evaluates emotional traits from specific localized objects within the image, matching text embeddings against a custom knowledge base.
    
- `attribute_combine_relation.py`: Evaluates the spatial and semantic relationships between multiple interacting objects (Relation Level) to adjust emotional interpretation.
    
- `final_label.py`: The final mathematical fusion module. It integrates Global, Object, and Relation level predictions using an Information Entropy weighted formula.
    
- `reward_model_train.py`: Trains the sequence classification Reward Model used during the reinforcement learning phase.
    
- `dynamic_feature_selection_rigorous.py`: Implements GRPO (Group Relative Policy Optimization) to fine-tune the model for dynamic feature selection based on the trained reward model.
    
- `requirements.txt`: Contains all necessary Python dependencies.
    

## Installation

We recommend using Python 3.9+ and a virtual environment (e.g., conda or venv).

```
# Clone the repository
git clone <your-repository-url>
cd <your-repository-directory>

# Install required packages
pip install -r requirements.txt
```

_(Optional) If you plan to use OpenAI API or similar endpoints, copy `.env.example` to `.env` and configure your API keys:_

```
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-base-url"
```

## Usage Instructions

### 1. Multi-Level Feature Extraction

You need to process your dataset through the three distinct levels. Please ensure you prepare your dataset and bounding boxes before running these scripts.

**Global Level:**

```
python attribute_combine_global.py \
    --data_folder /path/to/your/images \
    --output_path ./results/global_results.json
```

**Object Level:**

```
python attribute_combine_object.py \
    --json_path /path/to/extracted_features.json \
    --csv_path /path/to/knowledge_base.csv \
    --output_path ./results/object_results.json
```

**Relation Level:**

```
python attribute_combine_relation.py \
    --json_path /path/to/extracted_features.json \
    --output_path ./results/relation_results.json \
    --use_description
```

### 2. Final Label Fusion (Entropy Weighted)

Once you have generated the JSON results from the three levels above, fuse them using our entropy-based mathematical formula:

```
python final_label.py \
    --global_level ./results/global_results.json \
    --object_level ./results/object_results.json \
    --relation_level ./results/relation_results.json \
    --output_json ./results/final_evaluation_report.json
```

### 3. Dynamic Feature Selection (Advanced)

To train your own dynamic feature selector, you must first train the reward model, followed by the GRPO reinforcement learning loop:

```
# Train the Reward Model
python reward_model_train.py --csv_file /path/to/soft_labels.csv

# Run GRPO Fine-Tuning
python dynamic_feature_selection_rigorous.py \
    --checkpoint_path /path/to/reward_model.pt \
    --data_folder /path/to/images
```
