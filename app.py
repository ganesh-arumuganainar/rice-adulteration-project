from pyexpat import model
from flask import Flask, request, redirect, render_template
from werkzeug.utils import  secure_filename
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

rice_classes = ['Basmati', 'Gundu Malli', 'Raw Rice']
cereal_classes = ['Black Gram', 'Cow Peas', 'Green Moong Dal', 'Horse Gram']
application = Flask(__name__, static_url_path='/Users/ganesh/Downloads/Rice_Quality_Project_0.1/static')

application.config["IMAGE_UPLOADS"] = 'static'
application.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG"]

rice_model = tf.keras.models.load_model('models/rice_model')
cereal_model = tf.keras.models.load_model('models/cereal_model')

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

        category = request.form.get("subject")
        subCategory = request.form.get("topic")

        if request.files:

            image = request.files["image"]

            if image.filename == "":

                return redirect(request.url)

            if allowed_image(image.filename):

                filename = secure_filename(image.filename)
                
                image.save(os.path.join(application.config["IMAGE_UPLOADS"], filename))

                return redirect(f'/showing-image/{filename}/{category}/{subCategory}')

            else:

                return redirect(request.url)

    return render_template("upload_images.html")



@application.route("/showing-image/<image_name>/<category>/<subCategory>", methods=["GET", "POST"])

def showing_image(image_name, category, subCategory):
    if request.method == "POST":
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(os.path.join(application.config["IMAGE_UPLOADS"], image_name))
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        
        if category == "Rice":
            model = rice_model
            prediction = model.predict(data)
            gunduMalli= round(prediction[0][1]*100,2)
            basmati= round(prediction[0][0]*100,2)
            rawRice = round(prediction[0][2]*100,2)

            if rice_classes[np.argmax(prediction[0])]== subCategory and np.max(prediction[0]>=0.7):
                result = 'The variety is suitable for consumption'
            else:
                result = 'The variety is too much adulterated and not suitable for consumption'
            return render_template("prediction_result.html", image_name=image_name, category=category, subCategory=subCategory, gunduMalli=gunduMalli, basmati=basmati, rawRice=rawRice, result=result)

        
        else:
            model = cereal_model
            prediction = model.predict(data)
            blackGram= round(prediction[0][0]*100,2)
            cowPeas= round(prediction[0][1]*100,2)
            greenMoongDal = round(prediction[0][2]*100,2)
            horseGram = round(prediction[0][3]*100,2)
            if cereal_classes[np.argmax(prediction[0])]== subCategory and np.max(prediction[0]>=0.7):
                result = 'The variety is suitable for consumption'
            else:
                result = 'The variety is too much adulterated and not suitable for consumption'
            return render_template("prediction_result2.html", image_name=image_name, category=category, subCategory=subCategory, blackGram=blackGram, cowPeas=cowPeas, greenMoongDal=greenMoongDal, horseGram=horseGram, result=result)

        
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

        
    return render_template("showing_image.html", value=image_name)



if __name__ == '__main__':
    application.run(port=5000)
