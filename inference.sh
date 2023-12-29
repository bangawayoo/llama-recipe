export CUDA_VISIBLE_DEVICES="4"
#python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name "meta-llama/Llama-2-7b-hf" --output_dir ./ckpt

MODEL="meta-llama/Llama-2-13b-hf"
python examples/chat_completion/chat_stream.py --model_name "${MODEL}" \
      --prompt_file examples/chat_completion/maru_chat.json \
      --quantization \
      --peft_model "./ckpt"
