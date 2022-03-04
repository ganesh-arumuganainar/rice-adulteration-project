from flask import Flask, request, redirect, render_template
from werkzeug.utils import  secure_filename
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

rice_classes = ['Basmati', 'Gundu Malli']
application = Flask(__name__, static_url_path='/Users/ganesh/Downloads/Rice_Quality_Project_0.1/static')

application.config["IMAGE_UPLOADS"] = 'static'
application.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG"]

model = tf.keras.models.load_model('/Users/ganesh/Downloads/Rice_Quality_Project_0.1/model/keras_model.h5')

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in application.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@application.route("/", methods=["GET", "POST"])

def upload_image():
    if request.method=="POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":

                return redirect(request.url)

            if allowed_image(image.filename):

                filename = secure_filename(image.filename)
                
                image.save(os.path.join(application.config["IMAGE_UPLOADS"], filename))

                return redirect(f'/showing-image/{filename}')

            else:

                return redirect(request.url)

    return render_template("upload_images.html")



@application.route("/showing-image/<image_name>", methods=["GET", "POST"])

def showing_image(image_name):
    if request.method == "POST":
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(os.path.join(application.config["IMAGE_UPLOADS"], image_name))
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        gunduMalli= round(prediction[0][0]*100,2)
        basmati= round(prediction[0][1]*100,2)
        
        """ image = cv2.imread(image_path)
        img = image.copy()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        image = image.astype("float32")
        image = image/255.0
        np_image = np.expand_dims(image, axis=0)

        predictions = model(np_image)
        max_class = np.argmax(predictions)
        min_class = np.argmin(predictions)
        max_probability = np.max(predictions)
        min_probability = np.min(predictions)
        predicted_max_class = rice_classes[max_class]
        predicted_min_class = rice_classes[min_class] """

        return render_template("prediction_result.html", image_name=image_name, gunduMalli=gunduMalli, basmati=basmati)

    return render_template("showing_image.html", value=image_name)



if __name__ == '__main__':
    application.run(port=5000)





