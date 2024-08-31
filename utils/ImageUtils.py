
class ImageUtils:

    @staticmethod
    def generate_gif(file_name, list_images):
        gif = []
        repetition = 10
        for image in list_images:
            for i in range(repetition):
                gif.append(image)

        gif[0].save(file_name, save_all=True, optimize=False, append_images=gif[1:], loop=0)

