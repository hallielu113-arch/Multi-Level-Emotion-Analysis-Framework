<<<<<<< HEAD
{\rtf1\ansi\ansicpg936\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Bold;\f1\froman\fcharset0 Times-Roman;\f2\fmodern\fcharset0 Courier;
\f3\froman\fcharset0 Times-Italic;\f4\fmodern\fcharset0 Courier-Oblique;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c0\c93333;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat0\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat0\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa321\partightenfactor0

\f0\b\fs48 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Multi-Level Emotion Analysis Framework\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 This repository contains the official implementation of an advanced Multi-Level Emotion Analysis Framework.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Acknowledgment\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 This project is built upon the foundational {\field{\*\fldinst{HYPERLINK "https://github.com/YanbeiJiang/PICK"}}{\fldrslt 
\f0\b \cf3 \ul \ulc3 \strokec3 PICK Framework}} originally developed by YanbeiJiang. We extend the original architecture by introducing:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls1\ilvl0
\f0\b \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Rigorous Mathematical Fusion:
\f1\b0  A novel entropy-based weighting mechanism for combining multi-level visual features.\
\ls1\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Dynamic Feature Selection (GRPO):
\f1\b0  Advanced reinforcement learning fine-tuning to dynamically select key object features.\
\ls1\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Optimized Generalization:
\f1\b0  Codebase refactoring to support broader local and API-based visual-language models seamlessly.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Repository Structure\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The core modules of this repository are split into the following files:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls2\ilvl0
\f2\fs26 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 attribute_combine_global.py
\f1\fs24 : Analyzes the overall composition of images (Global Level) to determine baseline emotional distribution using Vision-Language Models.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 attribute_combine_object.py
\f1\fs24 : Extracts and evaluates emotional traits from specific localized objects within the image, matching text embeddings against a custom knowledge base.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 attribute_combine_relation.py
\f1\fs24 : Evaluates the spatial and semantic relationships between multiple interacting objects (Relation Level) to adjust emotional interpretation.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 final_label.py
\f1\fs24 : The final mathematical fusion module. It integrates Global, Object, and Relation level predictions using an Information Entropy weighted formula.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 reward_model_train.py
\f1\fs24 : Trains the sequence classification Reward Model used during the reinforcement learning phase.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 dynamic_feature_selection_rigorous.py
\f1\fs24 : Implements GRPO (Group Relative Policy Optimization) to fine-tune the model for dynamic feature selection based on the trained reward model.\
\ls2\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 requirements.txt
\f1\fs24 : Contains all necessary Python dependencies.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Installation\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 We recommend using Python 3.9+ and a virtual environment (e.g., conda or venv).\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 # Clone the repository\
git clone <your-repository-url>\
cd <your-repository-directory>\
\
# Install required packages\
pip install -r requirements.txt\
\pard\pardeftab720\sa240\partightenfactor0

\f3\i\fs24 \cf0 (Optional) If you plan to use OpenAI API or similar endpoints, copy 
\f4\fs26 .env.example
\f3\fs24  to 
\f4\fs26 .env
\f3\fs24  and configure your API keys:
\f1\i0 \
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 export OPENAI_API_KEY="your-api-key"\
export OPENAI_BASE_URL="your-api-base-url"\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Usage Instructions\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 1. Multi-Level Feature Extraction\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 You need to process your dataset through the three distinct levels. Please ensure you prepare your dataset and bounding boxes before running these scripts.\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b \cf0 Global Level:
\f1\b0 \
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 python attribute_combine_global.py \\\
    --data_folder /path/to/your/images \\\
    --output_path ./results/global_results.json\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b\fs24 \cf0 Object Level:
\f1\b0 \
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 python attribute_combine_object.py \\\
    --json_path /path/to/extracted_features.json \\\
    --csv_path /path/to/knowledge_base.csv \\\
    --output_path ./results/object_results.json\
\pard\pardeftab720\sa240\partightenfactor0

\f0\b\fs24 \cf0 Relation Level:
\f1\b0 \
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 python attribute_combine_relation.py \\\
    --json_path /path/to/extracted_features.json \\\
    --output_path ./results/relation_results.json \\\
    --use_description\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 2. Final Label Fusion (Entropy Weighted)\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Once you have generated the JSON results from the three levels above, fuse them using our entropy-based mathematical formula:\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 python final_label.py \\\
    --global_level ./results/global_results.json \\\
    --object_level ./results/object_results.json \\\
    --relation_level ./results/relation_results.json \\\
    --output_json ./results/final_evaluation_report.json\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 3. Dynamic Feature Selection (Advanced)\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 To train your own dynamic feature selector, you must first train the reward model, followed by the GRPO reinforcement learning loop:\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 # Train the Reward Model\
python reward_model_train.py --csv_file /path/to/soft_labels.csv\
\
# Run GRPO Fine-Tuning\
python dynamic_feature_selection_rigorous.py \\\
    --checkpoint_path /path/to/reward_model.pt \\\
    --data_folder /path/to/images\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 License\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Please refer to the original {\field{\*\fldinst{HYPERLINK "https://github.com/YanbeiJiang/PICK"}}{\fldrslt \cf3 \ul \ulc3 \strokec3 PICK repository}} for primary licensing constraints. Any additional modifications provided in this repository are released under the MIT License unless otherwise specified.\
}
=======
# Multi-Level-Emotion-Analysis-Framework
>>>>>>> 0c8a9da8582b5ba7bd44d2f3253981e1c4a047fe
