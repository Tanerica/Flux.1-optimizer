#quantize only transformer + torch compile all
python torchao_torchcompile.py \
    --model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --compile_mode "max-autotune" \
    --transformer \
    --text_encoder \
    --text_encoder_2 \
    --vae \
    --steps 8 \
    --height 1024 \
    --width 1024 \
    --seed 62 \
    --output_dir "./torchao_output"

