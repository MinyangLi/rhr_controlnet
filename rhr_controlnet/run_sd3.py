
# import torch
# from PIL import Image
# from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("checkpoint/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16)
# # pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16)
# pipe = pipe.to("cuda")
# pipe.vae.enable_tiling()
# image = pipe(
#     "A capybara holding a sign that reads Hello World",
#     # "A dog holding a sign that reads Hello World",
#     num_inference_steps=28,
#     guidance_scale=3.5,
#     height=3072, width=3072,
#     # strength=0.1,
#     # image=Image.open('capybara.png').resize((2048, 2048)),
# ).images[0]
# image.save("3072.png")





import torch, os
from pipelines.scheduling_flow_match_euler_discrete import FMEDScheduler
from pipelines.pipeline_sd3 import SD3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from utils.preprocess import latent2image, image2latent
from utils.main_tools import quantic_HW, quantic_cfg, quantic_step
from configs import (
    model_id, prompts, generator, It_base_path, num_inference_steps,
    N, cfg_min, cfg_max, M_cfg, T_min, T_max, M_T, res_min, res_max,
)

def main():
    # 1. init
    refresh_step_list = quantic_step(T_min, T_max, N, M_T)
    cfg_list = quantic_cfg(cfg_min, cfg_max, N, M_cfg)
    resolution_list = quantic_HW(res_min, res_max, N)
    os.makedirs(It_base_path, exist_ok=True)
    pipe = SD3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.vae.enable_tiling()
    pipe.scheduler = FMEDScheduler.from_config(pipe.scheduler.config)
    MEMORY = {}

    # 2. run
    for prompt in prompts:
        MEMORY.update({
            'predict_x0_list': [],
        })
        start_latent = randn_tensor((1, 16, res_min[0] // pipe.vae_scale_factor, res_min[1] // pipe.vae_scale_factor), dtype=pipe.dtype, device=pipe.device, generator=generator)
        for i in range(len(resolution_list)):
            if i == 0:
                predict_x0_latent_noisey = start_latent
            else:
                predict_x0_latent = image2latent(latent2image(MEMORY['predict_x0_list'][-1], pipe).resize(resolution_list[i]), pipe)
                noise_ = randn_tensor(predict_x0_latent.shape, dtype=predict_x0_latent.dtype, device=predict_x0_latent.device, generator=generator)
                predict_x0_latent_noisey = pipe.scheduler.scale_noise(predict_x0_latent, pipe.scheduler.timesteps[len(MEMORY['predict_x0_list']) - num_inference_steps].unsqueeze(0), noise_)
            # TODO flux加入pack，并在pipe运行时输入height和width
            hr_output = pipe(
                    prompt=prompt,
                    # negative_prompt="blurry, ugly, duplicated objects, poorly drawn, deformed, mosaic, bad hands, missing fingers, extra fingers, bad feet, unreasonable layout",
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg_list[i],
                    latents=predict_x0_latent_noisey,
                    MEMORY=MEMORY,
                    strength=(num_inference_steps - len(MEMORY['predict_x0_list'])) / num_inference_steps,
                    denoising_end=refresh_step_list[i + 1] / num_inference_steps,
                )
        hr_output.images[0].save(os.path.join(It_base_path, f'{prompt}.jpg'))



if __name__ == '__main__':
    main()



