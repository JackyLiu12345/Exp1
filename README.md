### Comparative Analysis of Transformer Architectures and Learning Techniques for Multilingual Political and Fake News Classification

#### Supervised fine-tuning experiment

This experiment focuses on using supervised learning to fine-tune LoRA (Low-Rank Adaptation) adapters for sequence classification across various datasets. We conducted all of our tests on a single A40 30GB NVIDIA GPU.

##### Models tested:

| Model name                 | Model Architecture | Huggingface ID                        | Note                    |
| -------------------------- | ------------------ | ------------------------------------- | ----------------------- |
| Llama-3.1-8B               | Decoder            | meta-llama/Meta-Llama-3.1-8B          | Access request required |
| Llama3.1-8B-instruct       | Decoder            | meta-llama/Meta-Llama-3.1-8B-Instruct | Access request required |
| Mistral-Nemo-Instruct-2407 | Decoder            | mistralai/Mistral-Nemo-Instruct-2407  | Access request required |
| Qwen2.5-7B-Instruct        | Decoder            | Qwen/Qwen2.5-7B-Instruct              | -                       |
| POLITICS                   | Encoder            | launch/POLITICS                       | -                       |
| RoBERTa-base               | Encoder            | FacebookAI/roberta-base               | -                       |
| RoBERTa-large              | Encoder            | FacebookAI/roberta-large              | -                       |
| mlx-RoBERTa-base           | Encoder            | FacebookAI/xlm-roberta-base           | -                       |
| mdeBERTa-base              | Encoder            | microsoft/mdeberta-v3-base            | -                       |
| ModernBERT                 | Encoder            | answerdotai/ModernBERT-base           | -                       |

##### Result reproduction

To reproduce our findings for any of the datasets, follow these steps:

1. Install dependencies: `pip install -r requirements.txt`
2. Use `python prepare_data.py --dataset_name DATASET_NAME --language LANGUAGE` to prepare the dataset you want to test.
3. Execute the training script:
   - For encoder models: `python train_encoder_seq_cls.py --model_name MODEL --epochs 3 --runs 5`
   - For decoder models: `python train_decoder_seq_cls.py --model_name MODEL --epochs 3 --runs 5`

Where MODEL is the model's id on huggingface, described in the table above.

**Note**: for the Clef 1C dataset: you have to also add the "--language LANGUAGE" flag when executing the training scripts. If it's not included, it defaults to English. The languages are: English, Dutch, Bulgarian, Arabic.  
_Example:_ `python train_encoder_seq_cls.py --model_name MODEL --language Bulgarian --epochs 3 --runs 5`

#### ICL experiment

This experiment investigates the effectiveness of prompts with varying levels of reasoning complexity, utilizing Few-shot learning and Chain-of-Thought (CoT) approaches. The prompts, detailed in the Appendix tables, range from simple to more elaborate reasoning structures. To conduct this experiment, we employed a computing infrastructure consisting of two Tesla P40 GPUs and one NVIDIA GeForce RTX 2080 Ti GPU.

##### Models tested:

| Model name                 | Model Architecture | Huggingface ID                        | Note                    |
| -------------------------- | ------------------ | ------------------------------------- | ----------------------- |
| Llama3.1-8B-instruct       | Decoder            | meta-llama/Meta-Llama-3.1-8B-Instruct | Access request required |
| Mistral-Nemo-Instruct-2407 | Decoder            | mistralai/Mistral-Nemo-Instruct-2407  | Access request required |
| Qwen2.5-7B-Instruct        | Decoder            | Qwen/Qwen2.5-7B-Instruct              | -                       |

##### Result reproduction

To reproduce our findings for any of the datasets, follow these steps:

1. Install dependencies: `pip install -r requirements.txt`
2. Use `python prepare_data.py --dataset_name DATASET_NAME --language LANGUAGE` to prepare the dataset you want to test.
3. Execute the training script for the corresponding configuration:
   1. Zero-shot/CoT: `python eval_zero_cot.py --model_name MODEL_NAME --dataset_name DATASET_NAME --configuration zero_shot_generic --task_labels LABELS --label_type LABEL_TYPE --language LANGUAGE --verbose`
   2. Few-shot: `python eval_few_shot.py --model_name MODEL_NAME --dataset_name DATASET_NAME --configuration CONFIGURATION --task_labels LABELS --label_type LABEL_TYPE --verbose --language LANGUAGE`

The args to compile the training scripts are the following:

Zero-shot/CoT:
   -`language: 'en', 'bg', 'ar', 'es', 'pt'`
   -`configuration: zero_shot_generic", "zero_shot_specific", "codebook", "cot`
   -`dataset_name: use the corresponding dataset's folder`
   -`model_name: refer to HuggingFace ID`
   -`task_labels: "hp", "pl", "ht", "fn"`

Few-shot:
   -`dataset_name: use the corresponding dataset's folder`
   -`configuration: "fs_dpp", "fs_random"`
   -`language: 'bg', 'en', 'pt'`
   -`task_labels: "hp", "pl", "ht", "fn"`
