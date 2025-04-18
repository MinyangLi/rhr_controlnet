import torch

seed = 2020
num_inference_steps = 50
generator = torch.Generator("cuda:1").manual_seed(seed)
# model_id = 'checkpoint/stable-diffusion-xl-base-1.0'
# model_id = 'checkpoint/stable-diffusion-v1-4'
# model_id = 'checkpoint/stable-diffusion-v1-5'
# model_id = 'checkpoint/stable-diffusion-2-base'
model_id = '/mnt/data/minyangli/checkpoints//stable-diffusion-xl-base-1.0'
It_base_path = f"results/controlnet_pose/woman"

# For 2048 x 2048
# res_min, res_max = (1024, 1024), (2048, 2048)
# N = 2
# cfg_min, cfg_max, M_cfg = 5, 30, 1
# T_min, T_max, M_T = 40, num_inference_steps, 1

# for 3072 x 3072
res_min, res_max = (1024, 1024), (3072, 3072)
N = 3
cfg_min, cfg_max, M_cfg = 5, 40, 1
T_min, T_max, M_T = 40, num_inference_steps, 1

# For 4096 x 4096
# res_min, res_max = (1024, 1024), (4096, 4096)
# N = 3
# cfg_min, cfg_max, M_cfg = 5, 50, 0.5
# T_min, T_max, M_T = 40, num_inference_steps, 0.5

# For 2048 x 4096
# res_min, res_max = (1536, 768), (4096, 2048)
# M_cfg, M_T, T_max, cfg_min = 1, 1, num_inference_steps, 5
# N = 3 # resize次数, 范围(2, 3), teaser图里的那些我都用的3
# cfg_max = 50 # cfg最大的选取, 范围(50, 30), 控制细节的生成，越大细节越多，但是越容易出现过曝现象
# T_min = 40 # 开始进行resize的step, 范围(35, 40), 控制自由度越大自由度越小，但可以生成更少的重复pattern



prompts = [
    "a futuristic cyberpunk woman with glowing neon tattoos, wearing a metallic bodysuit, standing in a rain-soaked alley with holographic billboards, night scene, ultra-detailed, 8K, negative: lowres, cartoon, medieval, extra limbs, nsfw, traditional background",
    
    "a high-fashion model in a dramatic black gown, posing in an avant-garde studio with abstract sculptures and soft shadows, editorial magazine style, 8K, photorealistic, negative: anime, fantasy armor, blurry, nsfw, streetwear",
    
    "a fantasy elf queen with silver hair and glowing blue eyes, standing in an enchanted forest with floating lights, wearing a flowing gown with ethereal patterns, magical atmosphere, ultra-detailed, 8K, negative: sci-fi, modern clothing, lowres, deformed face, nsfw",
    
    "a punk rock singer in ripped fishnets and leather jacket, holding a mic, standing in front of graffiti-covered walls, rebellious vibe, sharp shadows, photorealistic, 8K, negative: anime, dreamy, fantasy costume, blurry, nsfw",
    
    "a retro 80s disco diva in a shiny sequin dress with voluminous hair, standing under colorful spotlights and mirror balls, vibrant nightclub atmosphere, 8K, ultra-detailed, negative: grayscale, medieval, fantasy elements, lowres, nsfw",
    
    "a sci-fi android girl with synthetic skin and LED highlights, standing in a sterile futuristic lab, emotionless expression, sci-fi aesthetics, 8K, photorealistic, negative: fantasy, historical, blurry, extra arms, nsfw",
    
    "a pirate captain woman with a tricorn hat and ornate coat, standing on the deck of a wooden ship with a stormy sky behind her, cinematic lighting, 8K, ultra-detailed, negative: modern, lowres, cartoon, anime style, nsfw",
    
    "a vintage 1920s flapper girl in a fringed dress and feather headband, standing in a smoky jazz club with golden light, glamorous and nostalgic, photorealistic, 8K, negative: futuristic, neon colors, blurry, nsfw, sci-fi armor",
    
    "a warrior princess in golden armor, standing in a grand marble hall with glowing runes and a mystical sword, heroic and powerful, fantasy setting, 8K, ultra-detailed, negative: cartoon, lowres, cyberpunk elements, nsfw",
    
    "a steampunk inventor woman in leather corset and goggles, standing in a gear-filled workshop with steam pipes and glowing machinery, warm lighting, 8K, photorealistic, negative: anime, modern streetwear, blurry, nsfw"
]
