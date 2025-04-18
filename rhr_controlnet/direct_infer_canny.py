from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

seed = 2020
generator = torch.Generator("cuda:2").manual_seed(seed)
prompt = "aerial view, an advanced AI data center shaped with smooth, curving structures and glowing corridors, set in a vast synthetic plain, illuminated panels, cyberpunk lighting, dreamlike symmetry"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "checkpoints/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("checkpoints/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "/home/KeyuHu/qualitative_rhr/diffusion_models/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda:2")
# pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image,generator=generator,width=3072,height=3072, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_mc.jpg")
