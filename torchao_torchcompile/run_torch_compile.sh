#Normal
python torchcompile.py \
    --model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --compile_mode "max-autotune" \
    --steps 8 \
    --height 1024 \
    --width 1024 \
    --seed 62 \
    --output_dir "./torchcompile_output"
#Run vae max-autotune
python torchcompile.py \
    --model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --vae \
    --compile_mode "max-autotune" \
    --steps 8 \
    --height 1024 \
    --width 1024 \
    --seed 62 \
    --output_dir "./torchcompile_output"
#Run vae cuda-gpraph
python torchcompile.py \
    --model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --vae \
    --compile_mode "max-autotune" \
    --cuda_graph \
    --steps 8 \
    --height 1024 \
    --width 1024 \
    --seed 62 \
    --output_dir "./torchcompile_output"