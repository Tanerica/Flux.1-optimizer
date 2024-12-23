from huggingface_hub import login
login(token="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")
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
ckpt_id = "black-forest-labs/FLUX.1-schnell"
quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)

text_encoder_2_8bit = T5EncoderModel.from_pretrained(
    ckpt_id,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)

transformer_8bit = FluxTransformer2DModel.from_pretrained(
    ckpt_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
pipe = FluxPipeline.from_pretrained(
    ckpt_id,
    transformer=transformer_8bit,
    text_encoder_2=text_encoder_2_8bit,
    torch_dtype=torch.float16,
).to('cuda')

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
    generator = torch.Generator('cuda').manual_seed(62)
).images[0]
end_time = time.time()
out.save('bitnbytefp16.png')
print('Time: ', {end_time - start_time})