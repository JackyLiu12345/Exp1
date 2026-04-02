#!/bin/bash
# -------- CONFIGURABLE PARAMETERS --------
DATASET_NAME="fake_news_net"
LABEL_TYPE="string"  # New variable independent from argparse
CONFIG="zero_shot_generic"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
TASK_LABELS="fn"
LANGUAGE="en"  # Default language

#mistralai/Mistral-7B-Instruct-v0.3
#meta-llama/Llama-3.1-8B-Instruct
#Qwen/Qwen2.5-Omni-7B
# -----------------------------------------

# Validate task-language combinations and determine if language parameter is needed
LANGUAGE_ARG=""

# Special case for clef_1c dataset which maps to ht task
if [[ "$DATASET_NAME" == "clef_1c" ]]; then
    TASK_LABELS="ht"
    
    # Validate language for clef_1c (ht task) - only en, bg, ar are supported
    if [[ "$LANGUAGE" != "en" && "$LANGUAGE" != "bg" && "$LANGUAGE" != "ar" ]]; then
        echo "Error: For dataset clef_1c (ht task), only languages 'en', 'bg', and 'ar' are supported."
        echo "You specified: $LANGUAGE"
        exit 1
    fi
    LANGUAGE_ARG="--language ${LANGUAGE}"
fi

# Check for fn task language support (en, es, pt)
if [[ "$TASK_LABELS" == "fn" ]]; then
    if [[ "$LANGUAGE" != "en" && "$LANGUAGE" != "es" && "$LANGUAGE" != "pt" ]]; then
        echo "Error: For 'fn' task, only languages 'en', 'es', and 'pt' are supported."
        echo "You specified: $LANGUAGE"
        exit 1
    fi
    LANGUAGE_ARG="--language ${LANGUAGE}"
fi

# For ht task not coming from clef_1c dataset
if [[ "$TASK_LABELS" == "ht" && "$DATASET_NAME" != "clef_1c" ]]; then
    if [[ "$LANGUAGE" != "en" && "$LANGUAGE" != "bg" && "$LANGUAGE" != "ar" ]]; then
        echo "Error: For 'ht' task, only languages 'en', 'bg', and 'ar' are supported."
        echo "You specified: $LANGUAGE"
        exit 1
    fi
    LANGUAGE_ARG="--language ${LANGUAGE}"
fi

# Label type variable - independent from argparse
LABEL_TYPE="string"  # Default value, can be set when calling the script

MODEL_TAG=$(echo "$MODEL_NAME" | tr '/' '_' | tr ':' '_')
JOB_TAG="${DATASET_NAME}_${LABEL_TYPE}_${CONFIG}_${MODEL_TAG}"
if [[ -n "$LANGUAGE_ARG" ]]; then
    JOB_TAG="${JOB_TAG}_${LANGUAGE}"
fi

JOB_SCRIPT="job_${JOB_TAG}.sh"

# Generate SLURM script dynamically
cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=regular
#SBATCH --nodelist=hpc-gpu1
#SBATCH --job-name=${JOB_TAG}
#SBATCH --output=${JOB_TAG}_%j.out
#SBATCH --error=${JOB_TAG}_%j.err

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=""
huggingface-cli login --token \$HF_TOKEN

source /mnt/beegfs/home/michele.maggini/miniconda3/bin/activate unsloth_env

python eval_zero_cot.py \\
  --dataset_name "${DATASET_NAME}" \\
  --configuration "${CONFIG}" \\
  --task_labels "${TASK_LABELS}" \\
  --label_type "${LABEL_TYPE}" \\
  ${LANGUAGE_ARG} \\
  --model_name "${MODEL_NAME}"
EOF

# Submit the job
sbatch "$JOB_SCRIPT"
echo "Submitted job: $JOB_SCRIPT"

