import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
login(token="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")
ckpt = "black-forest-labs/FLUX.1-dev"
pipeline = DiffusionPipeline.from_pretrained(
    ckpt,
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipeline.transformer = torch._inductor.aoti_load_package("./bs_1_1024.pt2")
image = pipeline("cute dog", guidance_scale=0.0, max_sequence_length=256, num_inference_steps=8).images[0]
image.save("aot_compiled.png")