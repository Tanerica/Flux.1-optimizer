# Run normal - only schnell origin
python torchcompile.py \
    --model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --steps 8 \
    --height 1024 \
    --width 1024 \
    --seed 62 \
    --output_dir "./normal_output"