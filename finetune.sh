export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_API_KEY="2570d8af822487be5bd6478ecc3c153ac9beede5"
wandb online
huggingface-cli login --token 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 --token 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


#MODEL="kfkas/Llama-2-ko-7b-Chat" NousResearch/Nous-Hermes-Llama2-13b hyunseoki/ko-en-llama2-13b kyujinpy/KO-Platypus2-13B
#MODEL="kyujinpy/KO-Platypus2-13B NaverWebtoon/llama-2-ko-novel-7b"
#MODEL="./ckpt/webtoon-pretrained-peft/Ep80/"
MODEL="hyunseoki/ko-en-llama2-13b"

DATASET="chat_dataset"
# GPU4, BS2, GA2
LR=4e-4
EPOCHS=10
WARMUP=100
BS=2
GA=1
PEFT="True"

LORA_FLAG="False"
OUTPUT_DIR="./llama-13b/1102"
EXP_NAME="./1102-13b-volcano"

torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500  examples/finetuning.py \
        --enable_fsdp \
        --use_peft ${PEFT} \
        --peft_method "lora" \
        --model_from_lora $LORA_FLAG \
        --model_name "${MODEL}" \
        --pure_bf16 \
        --batch_size_training $BS \
        --gradient_accumulation_steps $GA \
        --num_epochs ${EPOCHS} \
        --warmup_steps ${WARMUP} \
        --lr ${LR} \
        --output_dir ./ckpt/${OUTPUT_DIR} \
        --dataset $DATASET \
        --dist_checkpoint_folder ${OUTPUT_DIR} \
        --exp_name $EXP_NAME \
        --low_cpu_fsdp


