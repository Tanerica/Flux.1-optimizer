from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxPipeline
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to('cuda')

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
    generator = torch.Generator('cuda').manual_seed(62)
).images[0]
end_time = time.time()
out.save('origin_schnell.png')
print('Time final: ', {end_time - start_time})