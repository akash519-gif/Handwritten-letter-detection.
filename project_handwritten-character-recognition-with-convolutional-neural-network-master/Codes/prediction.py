# import required packages
import tensorflow as tf
import numpy as np


def predict():
    # load model for prediction
    model = tf.keras.models.load_model('./model')

    # load image
    img = tf.keras.preprocessing.image.load_img('/home/amit/Downloads/qqq.jpg', color_mode="grayscale", target_size=(28, 28, 1))
    x = tf.keras.preprocessing.image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    pridiction = model.predict(x)

    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                 12: 'M', 13: 'N',
                 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
                 25: 'Z'}
    l1 = list(pridiction[0])

    for i in l1:
        if i == 1:
            print(word_dict.get(l1.index(i)))


predict()