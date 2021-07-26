import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

def predict(path):
    model = tf.keras.models.load_model('./model')
    img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=(28, 28, 1))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # print(model.predict(x), model.predict(x)[0][0])
    pridiction = model.predict(x)
    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                 12: 'M', 13: 'N',
                 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
                 25: 'Z'}
    l1 = list(pridiction[0])
    for i in l1:
        if i == 1:
            print(word_dict.get(l1.index(i)))
            return word_dict.get(l1.index(i))

# predict(img_path)


# Import flask package
from flask import Flask, Response, render_template, request, flash, redirect, url_for

# upload folder
UPLOAD_FOLDER = 'uploads'

# Image folder
image_folder = os.path.join('static', 'images')

# Create a REST server
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# static routing
app.config['images_folder'] = image_folder


# route: mapping of http method and the path
@app.route("/", methods=["GET"])
def root():
    return render_template('index.html')


# Extension allow for prediction
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# method to know allowed file from upload file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/data", methods=["POST"])
def upload_file():
    if request.method == 'POST':
        # upload image from user
        img = request.files['file']
        if img and allowed_file(img.filename):

            img.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename)))

            # use the model and predict the salary
            img_p = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename))
            result = predict(img_p)
            # result = img.filename
            # get the path for image
            # print(img_p)
            import shutil
            image_name = os.path.join(app.config['images_folder'], img.filename)
            shutil.copyfile(img_p, image_name)

            # print(f"Inside post_data {result}")
            return render_template("result.html", char=result, image_name=image_name)
            # return "upload done !"
        else:
            flash('😵 Please select image or Select correct image format')
            # return redirect(request.url)
            return render_template('index.html')


# Start the server
app.run(host="0.0.0.0", port=4000, debug=True)

