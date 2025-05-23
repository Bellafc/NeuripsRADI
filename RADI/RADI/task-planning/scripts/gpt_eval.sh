eval "$(conda shell.bash hook)"
conda activate MLDT

base_port=4045

python FADI/FADI/task-planning/llm_eval.py \
--base-port ${base_port} \
--eval \
--interactive_eval \
--max_retry 1 \
--llm gpt-4 \
--lora None \
--mode multi-layer \
--demo \
--api "your key"