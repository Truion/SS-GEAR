from transformers import AutoProcessor , AutoModelForCausalLM
import requests
from PIL import Image
import glob
import os
import tqdm
import torch

processor = AutoProcessor.from_pretrained('microsoft/git-large-coco')
model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-coco').cuda()

from diffusers import DiffusionPipeline

gen_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True).to('cuda')


folder = 'results/'

for f in tqdm.tqdm(os.listdir(folder)):
    # print(f)
    if 'val' in f or 'test' in f:
        continue
    else:
        file_path=os.path.join(folder, f)
        image= Image.open(file_path)

        pixel_values = processor(images=image, return_tensor='pt').pixel_values[0]
        pixel_values=torch.tensor(pixel_values).cuda()
        pixel_values=pixel_values.unsqueeze(0)

        generated_ids=model.generate(pixel_values=pixel_values, max_length=50)
        gen_cap=processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        images = gen_pipeline([gen_cap]*4).images

        for i, im in enumerate(images):
            im.save(f"aug_images/{f.replace('_out', '').replace('.JPEG', f'_{i}.JPEG')}")
        
        # breakpoint()