# compute score for torchao
python compute_score.py \
 --normal_images_folder './normal_output' \
 --generated_images_folder './torchao_output'

# compute score for torch compile only
python compute_score.py \
 --normal_images_folder './normal_output' \
 --generated_images_folder './torchcompile_output'