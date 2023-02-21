import os
from PIL import Image
import random

LIMIT = 12000

def cut_images(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Count the amount of images processed
    n = 0

    # Loop through the files in the subdirectories
    for root, dirs, files in os.walk(input_directory):
        for fn in files:
            if fn != "0.jpg":
                # Only take the first picture of each object. 
                # 1.png, 2.png,... are pictures (at different angles) of the same object
                continue

            filepath = os.path.join(root, fn)
            filename = os.path.dirname(filepath).split('\\')[-1]
            fragment_directory = f"{output_directory}/{filename}"

            # Open the image
            image = Image.open(filepath)

            amount_fragments_side = 3
            fragments_size = 96
            margins_size = 24 # This is on both sides, so it's x2 inbetween two fragments
            piece_size = (fragments_size + 2 * margins_size) # Size of the total piece (fragment + margin)
            total_size = amount_fragments_side * piece_size # Minimum size of the total image

            width, height = image.size
            if (width < total_size or height < total_size):
                # Image is too small -> Skip
                continue

            # Cut to correct size, in center
            cut_left = int((width - total_size) / 2)
            cut_top = int((height - total_size) / 2)
            image = image.crop((cut_left, cut_top, cut_left + total_size, cut_top + total_size))

            if not os.path.exists(fragment_directory):
                os.mkdir(fragment_directory)

            for i in range(amount_fragments_side):
                for j in range(amount_fragments_side):
                    # Calculate dimensions of piece (fragment + margins)
                    left = i * piece_size
                    right = left + piece_size
                    top = j * piece_size
                    bottom = top + piece_size

                    # Cut off the margins
                    margin_left = margins_size                      # TODO: Make this random
                    margin_right = margins_size * 2 - margin_left
                    margin_top = margins_size                        # TODO: Make this random
                    margin_bottom = margins_size * 2 - margin_top

                    # Crop and save
                    fragment = image.crop((left + margin_left, top + margin_top, right - margin_right, bottom - margin_bottom))
                    fragment_filename = f"{filename}_{i}_{j}.jpg"
                    fragment_path = os.path.join(fragment_directory, fragment_filename)
                    fragment.save(fragment_path)
                    
            n += 1
            if n % 100 == 0: print(f"{n} images processed...")
            if n >= LIMIT:
                print(f"Reached {LIMIT} images, stopping!")
                return

    print(f"Done! Processed {n} images")

cut_images("datasets/met", "images/out")
