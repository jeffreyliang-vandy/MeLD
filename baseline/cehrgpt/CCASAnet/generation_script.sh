export TRANSFORMERS_VERBOSITY=info
export CEHR_GPT_MODEL_DIR=./CCASAnet/model1/
export CEHR_GPT_DATA_DIR=./CCASAnet/pretrain/
export SYNTHETIC_DATA_OUTPUT_DIR=./CCASAnet/generated/10/
export TRANSFORMERS_VERBOSITY=info

python -u -m cehrgpt.generation.generate_batch_hf_gpt_sequence \
  --model_folder $CEHR_GPT_MODEL_DIR \
  --tokenizer_folder $CEHR_GPT_MODEL_DIR \
  --output_folder $SYNTHETIC_DATA_OUTPUT_DIR \
  --num_of_patients 43000 \
  --batch_size 32 \
  --buffer_size 128 \
  --context_window 3072 \
  --sampling_strategy TopPStrategy \
  --top_p 1.0 --temperature 1.0 --repetition_penalty 1.0 \
  --epsilon_cutoff 0.00 \
  --demographic_data_path $CEHR_GPT_DATA_DIR

# export PYTHONUNBUFFERED=1
# accelerate launch --num_processes 2 --mixed_precision fp16 -m cehrgpt.generation.generate_batch_hf_gpt_sequence_acc \
#   --model_folder $CEHR_GPT_MODEL_DIR \
#   --tokenizer_folder $CEHR_GPT_MODEL_DIR \
#   --output_folder $SYNTHETIC_DATA_OUTPUT_DIR \
#   --num_of_patients 43000 \
#   --batch_size 32 \
#   --buffer_size 128 \
#   --context_window 3072 \
#   --sampling_strategy TopPStrategy \
#   --top_p 1.0 --temperature 1.0 --repetition_penalty 1.0 \
#   --epsilon_cutoff 0.00 \
#   --demographic_data_path $CEHR_GPT_DATA_DIR

python ./CCASAnet/collect.py --input_dir "${SYNTHETIC_DATA_OUTPUT_DIR}/top_p10000/generated_sequences/" \
      --output_dir "${SYNTHETIC_DATA_OUTPUT_DIR}/../collected/"