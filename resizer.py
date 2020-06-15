from PIL import Image
import glob
import os
import shutil


# Run this script if you want resize test images and/or reorganize test folder

def resize():
    """
    resize images in test folder to fit in well in gui for predictions
    """
    image_list = []
    resized_images = []
    names = []

    # Create directory to store resized images if not all ready exists
    try:
        os.mkdir("category/test/resized")

    except FileExistsError:
        print("directory already exists")

    for filename in glob.glob("category/test/test/*.jpg"):
        name = filename.split('\\')
        name = name[1]
        names.append(name)
        img = Image.open(filename)
        image_list.append(img)

    for image in image_list:
        image = image.resize((600, 400))
        resized_images.append(image)

    for (i, new) in enumerate(resized_images):
        path_to_save = "category/test/resized/"
        new.save('{}{}'.format(path_to_save, names[i]))


def reorganize():
    """
    Run this function if you want to delete folder of non-resized images
    and rename created folder of resized to images to its original name 'test' where images stored originally
    This will recreate the original folder structure with only resized images
    """
    shutil.rmtree("category/test/test")

    source = "category/test/resized"

    # destination file path
    dest = "category/test/test"

    # try renaming the source path
    # to destination path
    # using os.rename() method

    try:
        os.rename(source, dest)
        print("Source path renamed to destination path successfully.")

    # If Source is a file
    # but destination is a directory
    except IsADirectoryError:
        print("Source is a file but destination is a directory.")

    # If source is a directory
    # but destination is a file
    except NotADirectoryError:
        print("Source is a directory but destination is a file.")

    # For permission related errors
    except PermissionError:
        print("Operation not permitted.")

    # For other errors
    except OSError as error:
        print(error)
