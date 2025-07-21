#!/bin/bash

#    for model in 'vit_small' 'vit_base' 'deit_tiny' 'deit_small' 'deit_base'
for seed in {0..0}
do
  for w_bit in '3' '4'
  do
    for a_bit in '4' '8'
    do
      for model in 'vit_small' 'vit_base' 'deit_tiny' 'deit_small' 'deit_base'
      do

        command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bits $w_bit --a_bits $a_bit --calib-batchsize 32 --coe 10000 --seed $seed --float_evaluation)
        echo "${command[@]}"
        nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"

        command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bits $w_bit --a_bits $a_bit --calib-batchsize 32 --coe 10000 --seed $seed --rotation random --add_linear_bf_head --replace_ln --rotation_float_evaluation --float_evaluation --add_linear_af_embed)
        echo "${command[@]}"
        nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"

        command=(--model $model --train_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train --val_dataset /data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords --w_bits $w_bit --a_bits $a_bit --calib-batchsize 32 --coe 10000 --seed $seed --rotation hadamard --add_linear_bf_head --replace_ln --rotation_float_evaluation --float_evaluation --add_linear_af_embed)
        echo "${command[@]}"
        nc run -C GPUALGALL python test_quant_expand.py "${command[@]}"
      done
    done
  done
done
