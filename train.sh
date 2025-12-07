OPTIMIZER_TYPE="Adafactor"

python train.py \
  --model_name_or_path /code/models/R1-Distill-Qwen-1.5B/ \
  --data_path /code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/outputs/R1-Distill-Qwen-7B/gsm8k_and_math/gsm8k/train_r1-distill-series_-1_seed0_t0.0_s0_e-1.jsonl \
  --output_dir SFT-Train/${OPTIMIZER_TYPE} \
  --strategy lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --optimizer_type ${OPTIMIZER_TYPE}
