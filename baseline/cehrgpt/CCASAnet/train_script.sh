#################training
export CEHR_GPT_MODEL_DIR=./CCASAnet/model2/
export CEHR_GPT_DATA_DIR=./CCASAnet/pretrain/
export TRANSFORMERS_VERBOSITY=info
export SEED=42 #42 vs 41


# nohup deepspeed src/cehrgpt/runners/hf_cehrgpt_pretrain_runner.py \
# nohup torchrun --nnodes=1 --nproc-per-node=2 src/cehrgpt/runners/hf_cehrgpt_pretrain_runner.py \
nohup python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $CEHR_GPT_MODEL_DIR \
  --data_folder "$CEHR_GPT_DATA_DIR" \
  --dataset_prepared_path "$CEHR_GPT_DATA_DIR" \
  --do_train true --seed $SEED \
  --continue_pretrain true \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 \
  --evaluation_strategy epoch --save_strategy epoch \
  --warmup_steps 50 --weight_decay 0.01 \
  --num_train_epochs 100 --learning_rate 5e-5 \
  --use_early_stopping --early_stopping_threshold 0.001 \
  --load_best_model_at_end \
  --report_to none \
  --gradient_accumulation_steps 1 \
  --sample_packing \
  --per_device_train_batch_size 2\
  --per_device_eval_batch_size 2 \
  --max_position_embeddings 3072 \
  --max_tokens_per_batch 7000   \
  --gradient_checkpointing true \
  --logging_steps 10 &> $CEHR_GPT_MODEL_DIR/nohup.out &
