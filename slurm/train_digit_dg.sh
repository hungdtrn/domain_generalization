
#!/bin/bash

data_name="digit_dg"
model_name="inlay_binary_sparse"
data_cfg_file="configs/datasets/dg/digits_dg.yaml"
cfg_file="configs/trainers/dg/vanilla/digits_dg.yaml"
other_cfg="--backbone ind --attn inlay --residual_type 2 --resnet_layer 2 --img_size 32 --seed 1802"
domains=("mnist" "mnist_m" "svhn" "syn")

for dst in ${domains[@]}; do
    src=""
    for d in ${domains[@]}; do
        if [[ $d != $dst  ]]; then
            src=${src}" "${d}
        fi
    done
    script="python tools/train.py --root /home/tduy/Workspace/dataset/domain_generalization/ --trainer Vanilla \
            --source-domains $src --target-domains $dst \
            --dataset-config-file $data_cfg_file \
            --config-file $cfg_file --output-dir output/${model_name}_${data_name}_${dst} \
            $other_cfg"

    echo $script 
    sbatch slurm/slurm.sh $script

    sleep 3
done
