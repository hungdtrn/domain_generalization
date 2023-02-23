
#!/bin/bash

data_name="office_home"
model_name="inlay_sparse_new"
data_cfg_file="configs/datasets/dg/office_home_dg.yaml"
cfg_file="configs/trainers/dg/vanilla/office_home_dg.yaml"
other_cfg="--backbone ind --attn inlay --sparse_res --seed 1802"
domains=("art" "clipart" "product" "real_world")

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
