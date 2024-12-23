from huggingface_hub import login
login(token="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")
import torch

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16


bfl_repo = "black-forest-labs/FLUX.1-schnell"
revision = "refs/pr/1"
# revision = None
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

# Experimental: Try this to load in 4-bit for <16GB cards.
#
# from optimum.quanto import qint4
# quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
# freeze(transformer)
quantize(transformer, weights=qfloat8)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.to('cuda')

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
out.save('quanto.png')
print('Time: ', {end_time - start_time})