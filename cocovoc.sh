#!/bin/bash
set -e

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

echo Running on $(nvidia-smi -L | wc -l) GPUs

which="$1"  # base, incr, all (which steps to train and possibly generate)
name="$2"  # name under which the incremental models are saved, e.g. Incr
base_name=${3:-"Base"}  # name of the base model, e.g. Base

replay_root=${4:-""}  # path to replay data, e.g. replay_data
rho=${5:-"0.5"}  # fraction of replay data per batch, e.g. 0.5

seed=${6:-"42"}  # random seed
resample=${7:-"4"}  # how often to resample, e.g. 4 (0 to deactivate)
shuffle=${8:-"shuffle"}  # shuffle, noshuffle
mode=${9:-"gen"}  # set this to real to use actual training data for replay instead of generated data
gen_model=${10:-"sd21"}  # sd21, sdxl
img_gen_bs=${11:-"8"}  # e.g. 8 for sd21, 3 for sdxl
resolution=${12:-"512"}  # 512 for sd21, 1024 for sdxl
diff_steps=${13:-"50"}  # number of diffusion steps, e.g. 50
prompt_skeleton=${14:-"photo of a CLASS"}  # image generation prompt, "CLASS" will be replaced by the actual class(es)
neg_prompt=${15:-"none"}  # e.g. "vintage,grayscale,low quality" ("none" to deactivate)
regen_base_step=${16:-"false"}  # "true" or "false", whether to regenerate replay data for step 0 or not
top_k=${17:-"10"}  # the top_k used in LoRA training and select_top_k.py

dataset=coco-voc
epochs=30
task=voc
lr_init=0.01
lr=0.001

ov=""

path=checkpoints/step/${dataset}-${task}/
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 --epochs ${epochs} $ov --val_interval 5"

if [ $which = base ] || [ $which = all ]; then
  exp --name ${base_name} --step 0 --lr ${lr_init} ${dataset_pars} --bce
fi

if [ $which = incr ] || [ $which = all ]; then
    pretr=${path}${base_name}_0.pth
  if [ "$replay_root" != "" ]; then
    if [ "$regen_base_step" == "true" ]; then
      accelerate launch generate.py ${replay_root} coco-voc voc disjoint 0 ${resample} ${shuffle} ${mode} ${gen_model} ${img_gen_bs} ${resolution} ${diff_steps} "$prompt_skeleton" "$neg_prompt" ${base_name} ${name} 24 ${seed} ${top_k}
    fi
    exp --name ${name} --step 1 --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --random_seed ${seed} --replay_root ${replay_root} --rho ${rho}
  else
    exp --name ${name} --step 1 --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --random_seed ${seed}
  fi
fi