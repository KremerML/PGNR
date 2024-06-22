@echo off
set CUDA_VISIBLE_DEVICES=0
set name=MIND-eval

set output=snap\%name%

set PYTHONPATH=%PYTHONPATH%;.\src

python src/evaluate.py ^
    --seed 42 ^
    --load MIND/BEST_EVAL_LOSS.pth ^
    --test MIND ^
    --val_batch_size 2000 ^
    --backbone "t5-small" ^
    --output %output% %* ^
    --max_text_length 512 ^
    --gen_max_length 64 ^
    --history_length 5 ^
    --pair_weight 0.4 ^
    --whole_word_embed > %name%.log
