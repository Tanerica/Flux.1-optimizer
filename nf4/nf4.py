from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel
import torch 
from huggingface_hub import login
login(token="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")

ckpt_id = "black-forest-labs/FLUX.1-schnell"
nf4_path = "/workspace/nf4/flux1-dev-bnb-nf4.safetensors"
transformer = FluxTransformer2DModel.from_single_file(nf4_path, torch_dtype=torch.bfloat16)

text_encoder = CLIPTextModel.from_pretrained(
    ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
)
# quantize(text_encoder, qfloat8)
# freeze(text_encoder)
text_encoder_2 = T5EncoderModel.from_pretrained(
    ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
# quantize(text_encoder_2, qfloat8)
# freeze(text_encoder_2)
vae = AutoencoderKL.from_pretrained(
    ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16
)
# Initialize the pipeline now.
pipe = FluxPipeline.from_pretrained(
	ckpt_id, 
    transformer=transformer, 
    vae=vae,
    text_encoder=text_encoder, 
    text_encoder_2=text_encoder_2, 
    torch_dtype=torch.bfloat16,
).to('cuda')
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
print('B0: ',torch.cuda.memory_allocated() / (1024 ** 3))
generator = torch.Generator().manual_seed(62)
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
out.save('quantoint4_2.png')
print('Time: ', {end_time - start_time})