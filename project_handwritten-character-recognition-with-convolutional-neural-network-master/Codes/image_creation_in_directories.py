"""
Here I am going to convert array to image from it's pixel value and put those images in their respective directory for
both in train and test set.

train set ------->  [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]

test set ------->  [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]

"""

# Import required packages
import os
import numpy as np
import cv2

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
             12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
             24: 'Y', 25: 'Z'}


def test_images_creation():
    # Open file of test.csv in read mode
    file = open('test.csv', 'r')

    count = 0
    labels = []

    # Directory where test image save
    parent_dir = os.path.join(os.getcwd(), 'test')

    while True:
        # read line of file
        line = file.readline()

        # Break if line not found
        if not line:
            break

        # Split line on ',' and create list of row values
        row = line.split(',')

        # extract label and pixel value from row
        # label = str(row[0]) --orignal
        lab_num = int(row[0])
        label = word_dict.get(lab_num)
        pixel = row[1:]

        # Convert pixel in numpy array of 28 x 28
        pixel = np.asarray(pixel, dtype=np.uint8).reshape((28, 28, 1))

        # join path of directories
        path = os.path.join(parent_dir, label)

        # count line number and use with image name
        count += 1

        # list of contents(directory and file both) in directory
        labels = os.listdir(parent_dir)

        if label in labels:
            # save image in its directory
            cv2.imwrite(f'{path}/image_{count}.png', pixel)
            print(f"{count} - not created directory only image add")
        else:
            try:
                os.mkdir(path)
            except OSError as error:
                print(error)
            # save image in its directory
            cv2.imwrite(f'{path}/image_{count}.png', pixel)
            print(f"{count} - created directory and image add")

    file.close()


test_images_creation()

def train_images_creation():
    # Open file of train.csv in read mode
    file = open('train.csv', 'r')

    count = 0
    labels = []

    # Directory where train image save
    parent_dir = os.path.join(os.getcwd(), 'train')

    while True:
        # read line of file
        line = file.readline()

        # Break if line not found
        if not line:
            break

        # Split line on ',' and create list of row values
        row = line.split(',')

        # extract label and pixel value from row
        # label = str(row[0]) --orignal
        lab_num = int(row[0])
        label = word_dict.get(lab_num)
        pixel = row[1:]

        # Convert pixel in numpy array of 28 x 28
        pixel = np.asarray(pixel, dtype=np.uint8).reshape((28, 28, 1))

        # join path of directories
        path = os.path.join(parent_dir, label)

        # count line number and use with image name
        count += 1

        # list of contents(directory and file both) in directory
        labels = os.listdir(parent_dir)

        if label in labels:
            # save image in its directory
            cv2.imwrite(f'{path}/image_{count}.png', pixel)
            print(f"{count} - not created directory only image add")
        else:
            try:
                os.mkdir(path)
            except OSError as error:
                print(error)
            # save image in its directory
            cv2.imwrite(f'{path}/image_{count}.png', pixel)
            print(f"{count} - created directory and image add")

    file.close()


# train_images_creation()
