import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import gc


def free_gpu():
    torch.no_grad()
    torch.cuda.empty_cache()
    gc.collect()


torch.manual_seed(1)
if not (Path.home() / '.huggingface' / 'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);


def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device) * 2 - 1)  # Note scaling
    return 0.18215 * latent.latent_dist.sample()


def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]


def build_text_embedding(prompt, batch_size=1):
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def generate_image_prompt(prompt, height=512, width=512, num_inference_steps=30, initial_image=None, start_step=0,
                          batch_size=1):
    # Prep text
    text_embeddings = build_text_embedding(prompt, batch_size=batch_size)
    return generate_image_prompt_embedding(text_embeddings, height=height, width=width,
                                           num_inference_steps=num_inference_steps, initial_image=initial_image,
                                           start_step=start_step, batch_size=batch_size)


def generate_image_prompt_embedding(text_embeddings, height=512, width=512, num_inference_steps=30, initial_image=None,
                                    start_step=0, batch_size=1):

    free_gpu()

    guidance_scale = 7.5
    generator = torch.manual_seed(32)

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    if initial_image is not None:

        # Prep latents (noising appropriately for start_step)
        start_sigma = scheduler.sigmas[start_step]
        latents = pil_to_latent(initial_image)
        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
        latents = latents.to(torch_device).float()

    else:

        # Prep latents
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma  # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):

            if i >= start_step:  # << This is the only modification to the loop we do

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]

                # Scale the latents (preconditioning):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)


def generate_and_display_pair_related_images(prompt1, prompt2):
    image1 = generate_image_prompt(prompt1, height=400, width=400, num_inference_steps=30)
    image2 = generate_image_prompt(prompt2, height=400, width=400, num_inference_steps=30, initial_image=image1, start_step=10)
    image1.save("out/image_pair_1.jpg")
    image2.save("out/image_pair_2.jpg")


def generate_transition_prompt_embeddings(prompt1, prompt2, num_steps=10):
    text_embeddings1 = build_text_embedding(prompt1)
    text_embeddings2 = build_text_embedding(prompt2)
    list_prompt_embeddings = []

    for i in range(num_steps + 1):
        percentage = i / num_steps
        text_embeddings = (1 - percentage) * text_embeddings1 + percentage * text_embeddings2
        list_prompt_embeddings.append(text_embeddings)

    return list_prompt_embeddings


def generate_transition_images(prompt1, prompt2, num_steps=10):
    list_prompt_embeddings = generate_transition_prompt_embeddings(prompt1, prompt2, num_steps=num_steps)

    list_images = []

    for text_embeddings in list_prompt_embeddings:
        image_mix = generate_image_prompt_embedding(text_embeddings, height=400, width=400)
        list_images.append(image_mix)

    return list_images


def generate_transition_images_smooth(prompt1, prompt2, num_steps=10):
    list_prompt_embeddings = generate_transition_prompt_embeddings(prompt1, prompt2, num_steps=num_steps)

    list_images = []
    initial_image = None
    start_step = 0

    for text_embeddings in list_prompt_embeddings:

        image_mix = generate_image_prompt_embedding(text_embeddings, num_inference_steps=30,
                                                    initial_image=initial_image, start_step=start_step, height=400, width=400)
        list_images.append(image_mix)
        if initial_image is None:
            initial_image = image_mix
        start_step = 10

    return list_images


def generate_gif(file_name, list_images):
    gif = []
    for image in list_images:
        gif.append(image)
    gif[0].save(file_name, save_all=True, optimize=False, append_images=gif[1:], loop=0)
    # FIXME duplicate images to make longer the gif


def transform_image(image_path, out_path, prompt, start_step=10):

    with Image.open(image_path) as image:
        image_real = generate_image_prompt(prompt, height=400, width=400, num_inference_steps=30, initial_image=image,
                                           start_step=start_step)
        image_real.save(out_path)


