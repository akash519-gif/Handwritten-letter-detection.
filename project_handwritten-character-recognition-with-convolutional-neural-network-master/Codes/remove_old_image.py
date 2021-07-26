# import require package
import os


def remove_content(directory):
    for file in os.scandir(directory):
        print(file.path)
        os.remove(file.path)
        print("Old images has been deleted")


upload_dir = './uploads'
remove_content(upload_dir)

images_dir = './static/images'
remove_content(images_dir)
