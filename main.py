from stable_diffusion_engine import StableDiffusionEngine as sde
from utils.ImageUtils import ImageUtils


command = "All"

if command == "Generate" or command == "All":
    print("Generate")
    prompt = "A photograph of a girl standing in a crowded street"
    image = sde.generate_image_prompt(prompt, height=400, width=400, num_inference_steps=30)
    image.save(".out/image_generated.jpg")


if command == "GeneratePairRelated" or command == "All":
    print("GeneratePairRelated")
    prompt1 = "A watercolor painting of a girl very detailed"
    prompt2 = "A watercolor painting of a cat very detailed"
    sde.generate_and_display_pair_related_images(prompt1, prompt2, output_path=".out")


if command == "Mix" or command == "All":
    print("Mix")
    text_embeddings1 = sde.build_text_embedding("a photograph of an inoffensive mouse")
    text_embeddings2 = sde.build_text_embedding("a photograph of a dangerous leopard")
    text_embeddings = text_embeddings1 + text_embeddings2
    image_mix = sde.generate_image_prompt_embedding(text_embeddings, height=400, width=400)
    image_mix.save(".out/image_mix.jpg")


if command == "Gif" or command == "All":
    print("Gif")
    list_images = sde.generate_transition_images_smooth("a photograph of a very cute cat", "a photograph of a big dog")
    ImageUtils.generate_gif(".out/cat_dog_transition.gif", list_images)


if command == "Transforms" or command == "All":
    print("Transforms")
    sde.transform_image("images/Dibujo1.jpg", ".out/Dibujo1_real.jpg", "a photograph of a lion, full body view", start_step=14)
    sde.transform_image("images/Dibujo2.jpg", ".out/Dibujo2_real.jpg", "a photograph of a factory with chimneys", start_step=14)
    sde.transform_image("images/Dibujo3.jpg", ".out/Dibujo3_real.jpg", "a photograph of a duck swimming, lateral view", start_step=16)
    sde.transform_image("images/Alex.jpg", ".out/Alex_out.jpg", "an anime character, manga, boy laying and smiling", start_step=8)

