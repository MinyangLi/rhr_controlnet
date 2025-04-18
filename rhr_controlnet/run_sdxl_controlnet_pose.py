import torch, os
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL,EulerAncestralDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from controlnet_aux import OpenposeDetector
from pipelines.pipelines_sdxl_controlnet_rhr import StableDiffusionXLControlNetPipelineRhr
from diffusers.utils import load_image
from utils.preprocess import latent2image, image2latent
from utils.main_tools import quantic_HW, quantic_cfg, quantic_step
from pipelines.pipeline_sdxl import SDXLPipeline
from configs import (
    model_id, prompts, generator, It_base_path, num_inference_steps,
    N, cfg_min, cfg_max, M_cfg, T_min, T_max, M_T, res_min, res_max,
)
import numpy as np
import cv2
from PIL import Image


def resize_control_image(controlnet_img, target_width, target_height):
    controlnet_img = cv2.resize(controlnet_img, (target_width, target_height))
    controlnet_img = Image.fromarray(controlnet_img)
    return controlnet_img

negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

def main():
    # 1. 初始化
    refresh_step_list = quantic_step(T_min, T_max, N, M_T)
    cfg_list = quantic_cfg(cfg_min, cfg_max, N, M_cfg)
    resolution_list = quantic_HW(res_min, res_max, N)
    os.makedirs(It_base_path, exist_ok=True)

    # 加载 SDXL 管道和 ControlNet
    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("/mnt/data/minyangli/checkpoints/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(
    "/mnt/data/minyangli/checkpoints/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("/mnt/data/minyangli/checkpoints/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipelineRhr.from_pretrained(
        model_id,
        controlnet=controlnet,
        vae = vae,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=eulera_scheduler
    ).to("cuda:1")
    pipe.vae.enable_tiling()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    processor = OpenposeDetector.from_pretrained('/mnt/data/minyangli/checkpoints/lllyasviel-ControlNet/annotator/ckpts')

    # 加载初始图像并生成初始 Canny 边缘图
    controlnet_img = cv2.imread("pose_input1.jpg")
    controlnet_img = processor(controlnet_img, hand_and_face=False, output_type='cv2')
   
    MEMORY = {}

    # 2. 执行
    for prompt in prompts:
        # print(f'prompts: {prompt}')
        MEMORY.update({'predict_x0_list': []})
        # print(f'memory prompts: {MEMORY["predict_x0_list"]}')
        start_latent = randn_tensor(
            (1, 4, res_min[0] // pipe.vae_scale_factor, res_min[1] // pipe.vae_scale_factor),
            dtype=pipe.dtype, device=pipe.device, generator=generator
        )
        # s
        for i in range(len(resolution_list)):
            current_res = resolution_list[i]
            control_image = resize_control_image(controlnet_img, current_res[1], current_res[0])
            if i == 0:
                predict_x0_latent_noisey = start_latent
            else:
                # print(f'memory prompts: {MEMORY["predict_x0_list"]}')
                predict_x0_latent = image2latent(latent2image(MEMORY['predict_x0_list'][-1], pipe).resize((resolution_list[i][1], resolution_list[i][0])), pipe)
                noise_ = randn_tensor(predict_x0_latent.shape, dtype=predict_x0_latent.dtype, device=predict_x0_latent.device, generator=generator)
                predict_x0_latent_noisey = pipe.scheduler.add_noise(predict_x0_latent, noise_, pipe.scheduler.timesteps[len(MEMORY['predict_x0_list']) - num_inference_steps])

            
            hr_output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnet_width=current_res[1],
                controlnet_height=current_res[0],
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg_list[i],
                latents=predict_x0_latent_noisey,
                MEMORY=MEMORY,
                strength=(num_inference_steps - len(MEMORY['predict_x0_list'])) / num_inference_steps,
                denoising_end=refresh_step_list[i + 1] / num_inference_steps,
                image=control_image, 
                controlnet_conditioning_scale=1.0, 
            )
            hr_output.images[0].save(os.path.join(It_base_path, f'{resolution_list[i]}.jpg'))
        
        # 保存最终图像
        hr_output.images[0].save(os.path.join(It_base_path, f'{prompt[:30]}.jpg'))

if __name__ == '__main__':
    main()