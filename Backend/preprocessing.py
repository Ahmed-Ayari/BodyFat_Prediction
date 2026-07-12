from io import BytesIO
from PIL import Image
import rembg

def photo_to_mask(image_path, threshold=127):
    """
    Converts a photo to a mask using the rembg library.

    Args:
        image_path (str): The path to the input image.
        threshold (int): Alpha cutoff for turning the soft mask into a binary mask.

    Returns:
        PIL.Image.Image: A single-channel grayscale mask in "L" mode.
    """
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = rembg.remove(input_data)

    rgba_image = Image.open(BytesIO(output_data)).convert("RGBA")
    alpha_channel = rgba_image.getchannel("A")
    binary_mask = alpha_channel.point(lambda pixel: 255 if pixel > threshold else 0, mode="L")

    return binary_mask