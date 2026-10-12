#!/bin/bash
set -e

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

echo Running on $(nvidia-smi -L | wc -l) GPUs

task="$1"  # e.g. 15-5
overlap="$2"  # overlap, disjoint
which="$3"  # base, incr, all (which steps to train and possibly generate)
name="$4"  # name under which the incremental models are saved, e.g. Incr
base_name=${5:-"Base"}  # name of the base model, e.g. Base

replay_root=${6:-""}  # path to replay data, e.g. replay_data
rho=${7:-"0.5"}  # fraction of replay data per batch, e.g. 0.5

seed=${8:-"42"}  # random seed
resample=${9:-"4"}  # how often to resample, e.g. 4 (0 to deactivate)
shuffle=${10:-"shuffle"}  # shuffle, noshuffle
mode=${11:-"gen"}  # set this to real to use actual training data for replay instead of generated data
gen_model=${12:-"sd21"}  # one of: sd21, sdxl, sd21_lora, sdxl_lora
img_gen_bs=${13:-"8"}  # e.g. 8 for sd21, 3 for sdxl
resolution=${14:-"512"}  # 512 for sd21, 1024 for sdxl
diff_steps=${15:-"50"}  # number of diffusion steps, e.g. 50
prompt_skeleton=${16:-"photo of a CLASS"}  # image generation prompt, "CLASS" will be replaced by the actual class(es)
neg_prompt=${17:-"none"}  # e.g. "vintage,grayscale,low quality" ("none" to deactivate)
regen_base_step=${18:-"false"}  # "true" or "false", whether to regenerate replay data for step 0 or not
top_k=${19:-"10"}  # the top_k used in LoRA training and select_top_k.py

dataset=voc
epochs=40
lr_init=0.01
lr=0.001

n_init=$(echo "$task" | cut -d'-' -f1)
n_incr=$(echo "$task" | cut -d'-' -f2)
n_steps=$(( (20 - n_init) / n_incr ))

if [ "$overlap" = "overlap" ]; then
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
else
  path=checkpoints/step/${dataset}-${task}/
  ov=""
  echo "Disjoint"
fi

dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 5"

if [ $which = base ] || [ $which = all ]; then
  exp --name ${base_name} --step 0 --bce --lr ${lr_init} ${dataset_pars}  --epochs 30 --random_seed ${seed}
fi

if [ $which = incr ] || [ $which = all ]; then
    pretr=${path}${base_name}_0.pth
  if [ "$replay_root" != "" ]; then
    for (( i=1; i<=n_steps; i++ ))
    do
      if [ "$regen_base_step" == "true" ] || [ $i -gt 1 ]; then
        accelerate launch generate.py ${replay_root} voc ${task} ${overlap} $((i-1)) ${resample} ${shuffle} ${mode} ${gen_model} ${img_gen_bs} ${resolution} ${diff_steps} "$prompt_skeleton" "$neg_prompt" ${base_name} ${name} 24 ${seed} ${top_k}
      fi
      exp --name ${name} --step ${i} --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --random_seed ${seed} --replay_root ${replay_root} --rho ${rho}
      pretr=${path}${name}_${i}.pth
    done
  else
    for (( i=1; i<=n_steps; i++ ))
    do
      exp --name ${name} --step ${i} --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --random_seed ${seed}
      pretr=${path}${name}_${i}.pth
    done
  fi
fi
