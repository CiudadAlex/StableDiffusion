
class ImageUtils:

    @staticmethod
    def generate_gif(file_name, list_images):
        gif = []
        for image in list_images:
            gif.append(image)

        gif[0].save(file_name, save_all=True, optimize=False, append_images=gif[1:], duration=1000, loop=0)

