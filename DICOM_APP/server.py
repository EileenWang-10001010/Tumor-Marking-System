from flask import Flask, render_template, request, url_for, redirect, flash
import torch
# import torch.nn as nn
import logging
import time
from inference import *

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['nii', 'dcm', 'jpg', 'jpeg', 'png'])

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/fetchImage', methods = ['POST', 'GET'])
def FetchImage():

    CHECK = False
    outputImageName=[]
    inputImageName=[]

    if request.method == 'POST':
        
        # Read image files in list
        files = request.files.getlist("inputImage")
        
        # Check file part & file type
        for idx, file in enumerate(files):
            CHECK = False
            if file.filename == '':
                flash(f"No {idx+1}th file part")
                return redirect("home.html")
            
            if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
                flash("Not allowed extension")
        # Save nii files, the other files will be saved in other sections
            if 'nii' in file.filename:
                file.save("./static/" + file.filename)
        # Input single dcm,jpeg,png image OR a pack of niis 
            processedImg = ct_slices_generator(file.filename, file)
        # Initialize the model    
            model = GetModel()
        # Inference
            infer(file.filename,processedImg, model)  

            CHECK = True
            if 'nii' in file.filename:
                outputImageName.append("../static/{}_output/{}.jpg".format(file.filename, idx))
                inputImageName.append("../static/{}_input/{}.jpg".format(file.filename,idx))
                
            else:
                outputImageName.append("../static/others_output/{}.jpg".format( file.filename))
                inputImageName.append("../static/others_input/{}.jpg".format(file.filename))
            str = zip(outputImageName,inputImageName)
            
        return render_template("home.html", CHECK=CHECK, str=str)

    return render_template("home.html")
                               
def allowed_file(filename, ALLOWED_EXTENSIONS):
    extension = filename.split('.')[1]
    if extension in ALLOWED_EXTENSIONS:
        return True
    return False


def GetModel():
    model = torch.load('./model.pth')
    return model


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
    logging.basicConfig()

