from stable_diffusion_engine import StableDiffusionEngine as sde
from utils.ImageUtils import ImageUtils

'''
WARNING:

In order to work and not get OutOfMemory Error in CUDA the images must be at most 400x400

'''

commands = ["All"]

if "Generate" in commands or "All" in commands:
    print("Generate")
    prompt = "A photograph of the back of a woman standing in a dark hostile crowded street. Black and white"
    image = sde.generate_image_prompt(prompt, height=400, width=400, num_inference_steps=30)
    image.save(".out/image_generated.jpg")


if "GeneratePairRelated" in commands or "All" in commands:
    print("GeneratePairRelated")
    prompt1 = "A watercolor painting of a girl very detailed"
    prompt2 = "A watercolor painting of a cat very detailed"
    sde.generate_and_display_pair_related_images(prompt1, prompt2, output_path=".out")


if "Mix" in commands or "All" in commands:
    print("Mix")
    text_embeddings1 = sde.build_text_embedding("a photograph of an eagle with long wings")
    text_embeddings2 = sde.build_text_embedding("a photograph of a big dangerous lion")
    text_embeddings = text_embeddings1 + text_embeddings2
    image_mix = sde.generate_image_prompt_embedding(text_embeddings, height=400, width=400)
    image_mix.save(".out/image_mix.jpg")


if "Gif" in commands or "All" in commands:
    print("Gif")
    list_images = sde.generate_transition_images_smooth("a photograph of the face of a cat", "a photograph of the face of an eagle")
    ImageUtils.generate_gif(".out/transition.gif", list_images)


if "Transforms" in commands or "All" in commands:
    print("Transforms")
    sde.transform_image(".images/Dibujo_leon.jpg", ".out/Dibujo_leon_real.jpg", "a very detailed drawing of a lion smiling, head and lateral body view with colors", start_step=14)
    sde.transform_image(".images/Dibujo_fabrica.jpg", ".out/Dibujo_fabrica_real.jpg", "a very detailed drawing of the front part of a big building made with bricks and with chimneys and smoke, with colors", start_step=14)
    sde.transform_image(".images/Dibujo_pato.jpg", ".out/Dibujo_pato_real.jpg", "a very detailed drawing of a duck swimming in a pond, with a beck, with colors", start_step=16)
    sde.transform_image(".images/Alex.jpg", ".out/Alex_out.jpg", "an anime character, manga, boy laying and smiling", start_step=8)

