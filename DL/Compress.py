from PIL import Image
import os
from tqdm import tqdm

CPS_SIZE = 128

def resize_images(input_folder, output_folder, target_size=(CPS_SIZE, CPS_SIZE)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        resize_image(filename, input_folder, output_folder, target_size)

def resize_image(filename, input_folder, output_folder, target_size=(CPS_SIZE, CPS_SIZE)):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            # Open the image
            img = Image.open(input_path)

            # Resize the image
            img_resized = img.resize(target_size, Image.LANCZOS)

            # Save the resized image
            img_resized.save(output_path, format="PNG")

        except Exception as e:
            print(f"Error processing image '{filename}': {str(e)}")

if __name__=="__main__":
    input_file_path = "./DateSet/Malimg/"
    output_file_path = "./DateSet/CPS_imgs/"
    resize_images(input_file_path, output_file_path)


