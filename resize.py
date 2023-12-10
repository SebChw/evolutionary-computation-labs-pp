import os

from PIL import Image

# Directory path
directory = "visualizations/"

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        # Open the image
        image = Image.open(os.path.join(directory, filename))

        # Resize the image
        resized_image = image.resize((1000, 1000))

        # Save the resized image
        resized_image.save(os.path.join(directory, f"{filename}"))
