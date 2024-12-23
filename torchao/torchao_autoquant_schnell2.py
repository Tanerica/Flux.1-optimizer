from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxPipeline 
from transformers import T5EncoderModel, CLIPTextModel
from torchao.quantization import quantize_, int8_weight_only, autoquant
import torch 
import torch._dynamo
torch._dynamo.config.suppress_errors = True
ckpt_id = "black-forest-labs/FLUX.1-schnell"
# Initialize the pipeline now.
pipe = FluxPipeline.from_pretrained(
	ckpt_id,  
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)


pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# warm-up
dum_prompt = "A cat holding a sign that says hello world"
for _ in range(5):
    dum_out = pipe(
    prompt=dum_prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=8,
    max_sequence_length=256,
).images[0]
prompt = "1boy solo, mature man, Jang Ho-joong, shirtless, muscular, hairy chest, shorts, stubble, facial hair, black hair, looking at you, standing, veiny arms, messy hair, black eyes, serious expression, front view, realistic, blake alexander style, best quality, best aesthetic, high details"

import time

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
out.save('torchao.png')
print('Time: ', {end_time - start_time})