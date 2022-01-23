from flask import Flask, request, redirect, render_template
from werkzeug.utils import  secure_filename
import os
import cv2
from riceQuality import getResults

application = Flask(__name__, static_url_path='/Users/ganesh/Downloads/Rice_Quality_Project_0.1/static')

application.config["IMAGE_UPLOADS"] = 'static'
application.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG"]


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
        
        image_path = os.path.join(application.config["IMAGE_UPLOADS"], image_name)
        Slender, Medium, Bold, Round, Total = getResults(image_path)

        return render_template("prediction_result.html", image_name=image_name, Slender=Slender, Medium=Medium, Bold=Bold, Round=Round, Total=Total)

    return render_template("showing_image.html", value=image_name)



if __name__ == '__main__':
    application.run()
    
