from PIL import Image
import glob
import os
from os.path import join
import shutil


# Run this script if you want resize images and/or reorganize folder

def resize(original_folder, destination_folder, extensions, size):
    """
    resize images in original folder to fit in well in gui for predictions
    """
    image_list = []
    resized_images = []
    names = []
    files = []

    # Look for images extensions in folder
    for ext in extensions:
        files.extend(glob.glob(join(original_folder, ext)))

    # Create directory to store resized images if not all ready exists
    try:
        os.mkdir(destination_folder)

    except FileExistsError:
        print("directory already exists")

    for filename in files:
        name = filename.split('\\')
        name = name[1]
        names.append(name)
        img = Image.open(filename)
        image_list.append(img)

    for image in image_list:
        image = image.resize(size)
        resized_images.append(image)

    for (i, new) in enumerate(resized_images):
        path_to_save = destination_folder + '/'
        new.save('{}{}'.format(path_to_save, names[i]))


def reorganize(original_folder, destination_folder, extensions):
    """
    Run this function if you want move resized images back to its original folder
    function will delete original non resized images and delete resized folder if it gets empty after moving images
    This will recreate the original folder structure with only resized images
    """

    original_files = []
    resized_files = []

    for ext in extensions:
        original_files.extend(glob.glob(join(original_folder, ext)))
        resized_files.extend(glob.glob(join(destination_folder, ext)))

    # Delete non-resized images
    for file in original_files:
        os.unlink(file)

    # Move resized images to its original folder
    for file in resized_files:
        shutil.move(file, original_folder)

    # Delete destination folder if empty
    if len(os.listdir(destination_folder)) == 0:
        shutil.rmtree(destination_folder)


def run():
    # Folder whose images you want to resize
    original_folder = "category/test/test"

    # This folder will get created if not all ready exists
    destination_folder = "category/test/resized"

    # Put images extensions you want resize
    extensions = ('*.jpg', '*.jpeg', '*.png')

    # Resize dimensions
    size = (600, 400)

    resize(original_folder, destination_folder, extensions, size)
    # reorganize(original_folder, destination_folder, extensions)
