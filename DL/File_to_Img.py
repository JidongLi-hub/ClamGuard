from PIL import Image
import os
from tqdm import tqdm



def create_grayscale_image(input_file, output_image_path, image_width=128):
    try:
        # Open the executable file in binary mode
        with open(input_file, 'rb') as file:
            # Read the entire content of the file
            file_content = file.read()

            # Calculate the height of the image based on the total number of bytes
            image_height = (len(file_content) // image_width) + 1

            # Create a new image with the specified width and height
            image = Image.new("L", (image_width, image_height))

            # Iterate through each byte in the file content and set the pixel value
            for i, byte in enumerate(file_content):
                # Calculate the coordinates of the pixel
                x = i % image_width
                y = i // image_width

                # Set the pixel value (grayscale)
                image.putpixel((x, y), byte)

            # Save the image as a PNG file
            image.save(output_image_path, format="PNG")

    except Exception as e:
        print(f"Error creating image: {str(e)}")

if __name__=="__main__":
    input_file_path = "./DateSet/bin/"
    output_image_path = "./DateSet/imgs/XXX.png"
    files = os.listdir(input_file_path)
    for file in tqdm(files):
        create_grayscale_image(os.path.join(input_file_path,file), output_image_path.replace("XXX", file))





