#!/usr/bin/bash

set -ex

eid=$(basename $0 .sh)  # Experiment id.

seeds=(11 22 33)
datasets=(cs_merlin de_merlin en_write_and_improve es_cedel2 it_merlin pt_cople2)
plm_restore_ckpt="plms/bert-base-multilingual-uncased"  # TODO: Change with your own plm path.

mkdir -p outs logs
for (( i=0; i<${#datasets[@]}; i++ )); do
  for seed in ${seeds[@]}; do
    source_datasets="${datasets[@]:0:i} ${datasets[@]:i+1}"
    target_datasets="${datasets[i]}"

    dp_outputs="outs/${eid}/${target_datasets}.seed_${seed}"

    python main.py \
      --dp-datasets data/raw \
      --source-datasets ${source_datasets} \
      --target-datasets ${target_datasets} \
      --plm-arch mbert-uncased-for-sequence-classification-with-supcl \
      --plm-restore-ckpt ${plm_restore_ckpt} \
      --max-seq-len 512 \
      --dp-outputs ${dp_outputs} \
      --cl-temp 0.1 \
      --mse-weight 0.9 \
      --cl-weight 0.1 \
      --cl-memory-bank-size 128 \
      --cl-projector linear \
      --seed ${seed} \
      | tee -a logs/${eid}.seed_${seed}.log

  done
done