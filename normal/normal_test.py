from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxPipeline
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to('cuda')

# warm-up
dum_prompt = "A cat holding a sign that says hello world"
for _ in range(3):
    dum_out = pipe(
    prompt=dum_prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=8,
    max_sequence_length=256,
).images[0]
prompt1 = "fantasy art, woman , young, messy short blonde hair, blue eyes, sitting on sofa, crossed legs, sexy legs, long legs, Karen Starr, Powergirl, Powergirl (DC) , Powergirl (Justice League) , Powergirl (DC comics) , Powergirl from DC , seductive, full length long white knit sweater that goes down to knees, bare shoulders , cleavage, sitting, white wool thigh highs, slim white body, pale skin, curvy body, big breasts, big boobs"
prompt3 = "old man with glasses portrait, photo, 50mm, f1. 4, natural light, Pathéchrome"
prompt2 = "1boy solo, mature man, Jang Ho-joong, shirtless, muscular, hairy chest, shorts, stubble, facial hair, black hair, looking at you, standing, veiny arms, messy hair, black eyes, serious expression, front view, realistic, blake alexander style, best quality, best aesthetic, high details"
prompt4 = 'Photo of a ultra realistic jeep gladiator, with a large sticker on the door with the legend "Wild Silver Fox", dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john William turner, A man with a medium build and very short gray hair, standing with his back to the viewer, looking towards a dramatic, highly realistic Icelandic landscape with active volcanoes in the distance. Next to him is a yellow Jeep Rubicon, heavily equipped for off-road adventures, featuring rugged tires and outdoor gear. Beside the man is his Siberian Husky, a brown-colored dog, attentively gazing at the volcanic scenery. The man is dressed in outdoor, adventurous clothing—wearing a durable jacket, cargo pants, and sturdy boots, perfect for exploring harsh and remote environments. The atmosphere is awe-inspiring, with dramatic volcanic terrain, ash clouds, and a moody sky'
prompt = [prompt1, prompt2, prompt3, prompt4]
import time
average_time = 0
for i in range(len(prompt)):
    start_time = time.time()
    out = pipe(
    prompt=prompt[i],
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=8,
    max_sequence_length=256,
).images[0]
    end_time = time.time()
    average_time += (end_time - start_time)
    print(f'Time {i}: ', (end_time - start_time))
    out.save(f"imageschnell{i}.png")
print('Average time: ', average_time / len(prompt))