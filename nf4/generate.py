from huggingface_hub import hf_hub_download
from accelerate.utils import set_module_tensor_to_device, compute_module_sizes
from accelerate import init_empty_weights
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
from convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, check_quantized_param
from diffusers import FluxTransformer2DModel, FluxPipeline
import safetensors.torch
import gc
import torch
ckpt_flux = "black-forest-labs/flux.1-schnell"
dtype = torch.bfloat16
ckpt_path = hf_hub_download(ckpt_flux, filename="flux1-schnell.safetensors")
original_state_dict = safetensors.torch.load_file(ckpt_path)
converted_state_dict = convert_flux_transformer_checkpoint_to_diffusers(original_state_dict)

del original_state_dict
gc.collect()

with init_empty_weights():
    config = FluxTransformer2DModel.load_config(ckpt_flux, subfolder="transformer")
    model = FluxTransformer2DModel.from_config(config).to(dtype)

_replace_with_bnb_linear(model, "nf4")
for param_name, param in converted_state_dict.items():
    param = param.to(dtype)
    if not check_quantized_param(model, param_name):
        set_module_tensor_to_device(model, param_name, device=0, value=param)
    else:
        create_quantized_param(model, param, param_name, target_device=0)

del converted_state_dict
gc.collect()

print(compute_module_sizes(model)[""] / 1024 / 1204)

pipe = FluxPipeline.from_pretrained(ckpt_flux, transformer=model, torch_dtype=dtype).to('cpu')
# pipe.enable_model_cpu_offload()

prompt = "A mystic cat with a sign that says hello world!"
image = pipe(prompt, guidance_scale=0, num_inference_steps=8, generator=torch.manual_seed(0)).images[0]
image.save("flux-nf4-dev.png")
