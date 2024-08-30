import StableDiffusionEngine as sde

print("START")
prompt1 = "A watercolor painting of an otter"
prompt2 = "A watercolor painting of a cat"
sde.generate_and_display_pair_related_images(prompt1, prompt2)
print("END")

'''
text_embeddings1 = sde.build_text_embedding("a mouse")
text_embeddings2 = sde.build_text_embedding("a leopard")
text_embeddings = text_embeddings1 + text_embeddings2
image_mix = sde.generate_image_prompt_embedding(text_embeddings, height=400, width=400)
image_mix.save(".out/image_mix.jpg")


list_images = sde.generate_transition_images_smooth("a photogtaph of a cat", "a photogtaph of a dog")
sde.generate_gif(".out/cat_dog_transition.gif", list_images)


sde.transform_image("images/Dibujo1.jpg", ".out/Dibujo1_real.jpg", "a photograph of a lion, full body view", start_step=14)
sde.transform_image("images/Dibujo2.jpg", ".out/Dibujo2_real.jpg", "a photograph of a factory with chimneys", start_step=14)
sde.transform_image("images/Dibujo3.jpg", ".out/Dibujo3_real.jpg", "a photograph of a duck swimming, lateral view", start_step=16)


sde.transform_image("images/Alex.jpg", ".out/Alex_out.jpg", "an anime character, manga, boy laying and smiling", start_step=8)
'''
