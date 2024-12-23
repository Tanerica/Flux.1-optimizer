from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxPipeline
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to('cuda')

#prompt = "1boy solo, mature man, Jang Ho-joong, shirtless, muscular, hairy chest, shorts, stubble, facial hair, black hair, looking at you, standing, veiny arms, messy hair, black eyes, serious expression, front view, realistic, blake alexander style, best quality, best aesthetic, high details"

import time
prompt0 = "A cat holding a sign that says Tan Ngo is handsome"
prompt1 = "A high-quality image of a leopard gecko that’s all yellow with no black on his yellow skin. His eyes are black and super high-quality IMAX movie style. He has a moist look and has a lot of depth. His pink lounge slightly out licking a yellow banana popsicle."
prompt2 = "1boy solo, mature man, Jang Ho-joong, shirtless, muscular, hairy chest, shorts, stubble, facial hair, black hair, looking at you, standing, veiny arms, messy hair, black eyes, serious expression, front view, realistic, blake alexander style, best quality, best aesthetic, high details"
prompt3 = "A woman seated on a sofa, facing the camera directly, with blonde hair pulled back and a few loose strands. She wears a black deep V-neck dress and a statement necklace. Her hands rest lightly on her lap. The background features large windows with soft, natural light illuminating her face."
prompt4 = "#High image quality, high color saturation #Milan fashion show, runway #Beautiful Japanese female model wearing the sample morning dress #This (beautiful and cute) Japanese female model has big eyes, a high nose, is tall and has long limbs, and the figure of a fashion model #The sample morning dresses come in a variety of patterns, including (Japanese cherry blossom pattern)."
prompt5 = "Ultra-realistic close-up of a tiny baby pig, smaller than a fingernail, sitting on the tip of a person’s finger. The baby pig has soft pink skin with delicate wrinkles, tiny ears, and a cute, upturned snout. Its little hooves rest gently on the fingertip, and the close-up emphasizes the fine details of its skin texture, snout, and hooves, capturing its adorable yet surprisingly strong look. Soft lighting should highlight the contrast between the piglet's miniature size and the human finger, enhancing the charm and detail of the tiny animal."
prompt6 = "An image of a woman with a hyperrealistic style, showcasing an elaborate body paint design. The skin should be painted in segments of black, gold, and black, with patterns that mimic flowing lines and dots. The face features an asymmetrical mask-like effect, with one eye highlighted in gold and the other in black, complemented by matching eyeshadow. The lips are a vibrant red, contrasting with the face paint. The figure's blonde hair is tousled, enhancing the fantastical element. The image should blend the precision of hyperrealism with the creativity of fantasy body art."
prompt7 = "A sleek, photorealistic depiction of a stunning redhead, her vibrant green eyes locked onto the viewer as she confidently poses in a futuristic space station setting. She's clad in a form-fitting Jedi-inspired suit with a flowing hood, showcasing her impressive assets: massive breasts, thick lips, and tantalizing cleavage that threatens to spill over her outfit. Her slender physique is accentuated by the wide crotch of the suit, drawing attention to her toned physique. Makeup is applied sparingly, allowing her natural beauty to shine through as she gazes directly at the viewer with an air of quiet confidence."
prompt8 = "Create a professional, ultra-realistic 4K image of a high-tech mega drone capturing an aerial view of a futuristic city at night. The drone hovers high above, equipped with advanced cameras and sensors, offering us a breathtaking perspective of the city below. The urban landscape is illuminated by vibrant and warm neon lights from towering skyscrapers, with intricate details on the buildings’ facades and the bustling streets far below. The colors are hyper-vivid and warm, enhancing the dynamic atmosphere of the city at night. The drone’s sleek, cutting-edge design is visible in the foreground, showcasing the blend of technology and urban life in a stunning visual display."
prompt9 = "A young woman with long dark hair stands in a snowy forest, her seductive smile illuminated by soft, warm lighting. She wears a white fur coat with a large hood, draped over her shoulders, and a black lingerie set with thigh-high stockings. One hand rests on her hip, the other on her thigh, as she poses confidently. Snowflakes gently fall, creating a dreamy, ethereal atmosphere. The scene is framed with a shallow depth of field, emphasizing her expression and the intricate details of her attire. The lighting is moody and atmospheric, with a soft, diffused key light from the front and a subtle rim light from behind, casting a gentle glow on her skin. The background trees are covered in snow, adding to the serene, otherworldly mood. "
prompts = [prompt0, prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9]
for i in range(len(prompts)):
    start_time = time.time()
    out = pipe(
        prompt=prompts[i],
        guidance_scale=0.,
        height=1024,
        width=1024,
        num_inference_steps=8,
        max_sequence_length=256,
        generator = torch.Generator('cuda').manual_seed(62)
    ).images[0]
    end_time = time.time()
    print(f"Length of prompt {i}: ", len(prompts[i].split(" ")))
    out.save(f'origin_schnell{i}.png')
    print(f'Time compute prompt {i}: ', {end_time - start_time})
# out.save('origin_schnell.png')
# print('Time final: ', {end_time - start_time})