@echo off
set CUDA_VISIBLE_DEVICES=0
set name=MIND

set output=%name%

set PYTHONPATH=%PYTHONPATH%;./src
python -m torch.distributed.launch ^
    --nproc_per_node=1 ^
    --master_port 1178 ^
    src/pretrain.py ^
        --seed 42 ^
        --train MIND ^
        --valid MIND ^
        --batch_size 16 ^
        --val_batch_size 16 ^
        --optim adamw ^
        --warmup_ratio 0.05 ^
        --lr 1e-3 ^
        --num_workers 4 ^
        --clip_grad_norm 1.0 ^
        --losses sequential ^
        --backbone t5-small ^
        --output %output% ^
        --epoch 30 ^
        --max_text_length 512 ^
        --gen_max_length 64 ^
        --history_length 5 ^
        --pair_weight 0.4 ^
        --whole_word_embed > %name%.log