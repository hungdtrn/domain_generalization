
#!/bin/bash

data_name="pacs"
model_name="inlay_sparse_new"
data_cfg_file="configs/datasets/dg/pacs.yaml"
cfg_file="configs/trainers/dg/vanilla/pacs.yaml"
other_cfg="--backbone ind --attn inlay --sparse_res --seed 1802"
# other_cfg="--backbone ind --attn inlay --sparse_res --learnable_dim 32 --seed 1802"

domains=("art_painting" "cartoon" "photo" "sketch")

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
