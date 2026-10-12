#!/bin/bash
set -e

dataset_name="$1"  # 10-10-ov + _top_k
step="$2"  # 0, 1, 2, ...
rank="$3"  # 8, 16, 32, 64
learning_rate="$4"  # 1e-4, 5e-5, 1e-5
num_epochs="$5"  # 5, 10, 20
seed="$6"

BASE_ARGS=(
    # base settings
        --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
        --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"
        --num_validation_images=8
        --validation_prompt="RANDOM_SELECTION"
        --report_to="wandb"
        --seed="$seed"
    # training and model settings
        # --train_text_encoder=False
        # --image_interpolation_mode="lanczos"
        # --center_crop=False
        --random_flip
        # --lr_scheduler="constant"
        --lr_warmup_steps=0
        --train_batch_size=1
        --gradient_accumulation_steps=1
        # --snr_gamma=None
    # performance
        --use_8bit_adam
        --mixed_precision="fp16"
        # --enable_xformers_memory_efficient_attention
        # --gradient_checkpointing
    # testing & logging
        # --max_train_samples=10
        --checkpointing_steps=500
    # unimportant stuff
        # --validation_epochs=1
        # --revision=None
        # --variant=None
        # --dataset_name=None
        # --dataset_config_name=None
        # --image_column="image"
        # --caption_column="text"
        # --cache_dir=None
        # --resolution=1024
        # --checkpoints_total_limit=None
        # --resume_from_checkpoint=None
        # --scale_lr=False
        # --allow_tf32=False
        # --dataloader_num_workers=6
        # --adam_beta1=0.9
        # --adam_beta2=0.999
        # --adam_weight_decay=1e-2
        # --adam_epsilon=1e-08
        # --max_grad_norm=1.0
        # --push_to_hub=False
        # --hub_token=None
        # --prediction_type=None
        # --hub_model_id=None
        # --logging_dir="logs"
        # --local_rank=-1
        # --enable_npu_flash_attention=False
        # --noise_offset=0
        # --debug_loss=False
        # --max_train_steps=None
)


if [ "$step" = "all" ]; then
    task=$(echo "$dataset_name" | cut -d'_' -f1)
    n_init=$(echo "$task" | cut -d'-' -f1)
    n_incr=$(echo "$task" | cut -d'-' -f2)
    n_steps=$(( (20 - n_init) / n_incr ))
    for (( i=0; i<n_steps; i++ ))
    do
        MY_ARGS=(
            "${BASE_ARGS[@]}"
            --train_data_dir="lora_prep_data_seed_$seed/$dataset_name/$i"
            --output_dir="lora_outputs_sdxl_seed_$seed/$dataset_name/$i"
            --rank="$rank"
            --learning_rate="$learning_rate"
            --num_train_epochs="$num_epochs"
        )
        accelerate launch train_text_to_image_lora_sdxl.py "${MY_ARGS[@]}"
    done
elif [ "$step" = "incr" ]; then
    task=$(echo "$dataset_name" | cut -d'_' -f1)
    n_init=$(echo "$task" | cut -d'-' -f1)
    n_incr=$(echo "$task" | cut -d'-' -f2)
    n_steps=$(( (20 - n_init) / n_incr ))
    for (( i=1; i<n_steps; i++ ))
    do
        MY_ARGS=(
            "${BASE_ARGS[@]}"
            --train_data_dir="lora_prep_data_seed_$seed/$dataset_name/$i"
            --output_dir="lora_outputs_sdxl_seed_$seed/$dataset_name/$i"
            --rank="$rank"
            --learning_rate="$learning_rate"
            --num_train_epochs="$num_epochs"
        )
        accelerate launch train_text_to_image_lora_sdxl.py "${MY_ARGS[@]}"
    done
else
    MY_ARGS=(
        "${BASE_ARGS[@]}"
        --train_data_dir="lora_prep_data_seed_$seed/$dataset_name/$step"
        --output_dir="lora_outputs_sdxl_seed_$seed/$dataset_name/$step"
        --rank="$rank"
        --learning_rate="$learning_rate"
        --num_train_epochs="$num_epochs"
    )
    accelerate launch train_text_to_image_lora_sdxl.py "${MY_ARGS[@]}"
fi