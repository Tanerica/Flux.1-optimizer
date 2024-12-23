"""
Some bits are from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
"""
from huggingface_hub import login
login(token="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")
from huggingface_hub import hf_hub_download
from accelerate.utils import set_module_tensor_to_device, compute_module_sizes
from accelerate import init_empty_weights
from convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, check_quantized_param
from diffusers import FluxTransformer2DModel, FluxPipeline
import safetensors.torch
import gc
import torch

dtype = torch.bfloat16
is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
ckpt_path = hf_hub_download("sayakpaul/flux.1-dev-nf4", filename="diffusion_pytorch_model.safetensors")
# ckpt_path = hf_hub_download("lllyasviel/flux1-dev-bnb-nf4", filename="flux1-dev-bnb-nf4.safetensors")
original_state_dict = safetensors.torch.load_file(ckpt_path)

with init_empty_weights():
    config = FluxTransformer2DModel.load_config("sayakpaul/flux.1-dev-nf4")
    model = FluxTransformer2DModel.from_config(config).to(dtype)
    expected_state_dict_keys = list(model.state_dict().keys())

_replace_with_bnb_linear(model, "nf4")

for param_name, param in original_state_dict.items():
    if param_name not in expected_state_dict_keys:
        continue
    
    is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
    if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
        param = param.to(dtype)
    
    if not check_quantized_param(model, param_name):
        set_module_tensor_to_device(model, param_name, device=0, value=param)
    else:
        create_quantized_param(
            model, param, param_name, target_device=0, state_dict=original_state_dict, pre_quantized=True
        )

del original_state_dict
gc.collect()

print(compute_module_sizes(model)[""] / (1024**3))

pipe = FluxPipeline.from_pretrained("black-forest-labs/flux.1-schnell", transformer=model, torch_dtype=dtype).to('cuda')
# torch.compile() don't work
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
import time
# warm-up
dum_prompt = "A cat holding a sign that says hello world"
for i in range(4):
    start_time = time.time()
    dum_out = pipe(
    prompt=dum_prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=8,
    max_sequence_length=256,
).images[0]
    end_time = time.time()
    print(f'Time: {i}', {end_time - start_time})
prompt = "1boy solo, mature man, Jang Ho-joong, shirtless, muscular, hairy chest, shorts, stubble, facial hair, black hair, looking at you, standing, veiny arms, messy hair, black eyes, serious expression, front view, realistic, blake alexander style, best quality, best aesthetic, high details"


start_time = time.time()
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=8,
    max_sequence_length=256,
).images[0]
end_time = time.time()
out.save('nf4.png')
print('Time: ', {end_time - start_time})