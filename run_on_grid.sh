#!/bin/bash

#    for model in 'vit_small' 'vit_base' 'deit_tiny' 'deit_small' 'deit_base'
for seed in {0..0}
do
  for w_bit in '3'
  do
    for model in 'vit_small' 'vit_base' 'deit_tiny' 'deit_small' 'deit_base'
    do

      command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bit $w_bit --a_bit 4 --calib-batchsize 32 --coe 10000 --seed $seed)
      echo "${command[@]}"
      nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"

      command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bit $w_bit --a_bit 4 --calib-batchsize 32 --coe 10000 --seed $seed --rotation random)
      echo "${command[@]}"
      nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"

      command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bit $w_bit --a_bit 4 --calib-batchsize 32 --coe 10000 --seed $seed --rotation hadamard)
      echo "${command[@]}"
      nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"
    done
  done
done
