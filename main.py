from wsgiref import simple_server
from flask import Flask, request, render_template,jsonify
from flask import Response
import os
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from feature_extraction import generate_data_set
# from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
# import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X,y)






os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")
print(pd.read_csv("./Prediction_Output_File/Predictions.csv"))

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


@app.route("/predicturl", methods=["GET", "POST"])
def predicturl():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("index.html", xx =-1)



# @app.route("/", methods=["GET", "POST"])
# def index():
#     return render_template("index.html")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # print(request.files)
            print("file name",type(request.files['file']))
            file1=request.files['file']
            
            path="./Prediction_Batch_Files/"
            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization
            print("prediction from model",pred)

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            print("path",path)


        except Exception as e:
            print(e)
    return render_template("index.html")

@app.route("/predictFolder", methods=["GET", "POST"])
def predictFolder():
    if request.method == "POST":
        try:
            # print(request.files)
            print("file name",type(request.form['folder']))
            path=request.form['folder']
            
            print(path)
            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization
            print("prediction from model",pred)

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            print("path",path)
            # prediction data
            predict_data=pd.read_csv("./Prediction_Output_File/Predictions.csv")
            predict_data_total=predict_data['Predictions'].value_counts()


        except Exception as e:
            print(e)
    return render_template("result.html", predict_data_total=predict_data_total)
        
@app.route("/retraining", methods=["GET", "POST"])
def retraining():
    try:
        path=request.form['retraing_path']
        train_valObj = train_validation(path) #object initialization

        train_valObj.train_validation()#calling the training_validation function


        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return render_template("index.html")





@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = prediction.form_response(dict_req)
                return render_template("index.html", response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html",xx= -1)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)







# webapp_root = "C:\Users\nitin\Downloads\phishingClassifierRecording+Code\phishingClassifier\webapp"

# static_dir = os.path.join(webapp_root, "static")
# template_dir = os.path.join(webapp_root, "templates")

# app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

# dashboard.bind(app)
# CORS(app)

# @app.route("/", methods=["GET", "POST"])
# def index1():

#     if request.method == "GET":
#         return render_template("index.html")

# @app.route("/predict", methods=['POST',"GET"])
# @cross_origin()
# def predictRouteClient():
#     try:
#         if request.json['folderPath'] is not None:
#             path = request.json['folderPath']

#             pred_val = pred_validation(path) #object initialization

#             pred_val.prediction_validation() #calling the prediction_validation function

#             pred = prediction(path) #object initialization

#             # predicting for dataset present in database
#             path = pred.predictionFromModel()
#             return Response("Prediction File created at %s!!!" % path)

#     except ValueError:
#         return Response("Error Occurred! %s" %ValueError)
#     except KeyError:
#         return Response("Error Occurred! %s" %KeyError)
#     except Exception as e:
#         return Response("Error Occurred! %s" %e)



# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():

#     try:
#         if request.json['folderPath'] is not None:
#             path = request.json['folderPath']
#             train_valObj = train_validation(path) #object initialization

#             train_valObj.train_validation()#calling the training_validation function


#             trainModelObj = trainModel() #object initialization
#             trainModelObj.trainingModel() #training the model for the files in the table


#     except ValueError:

#         return Response("Error Occurred! %s" % ValueError)

#     except KeyError:

#         return Response("Error Occurred! %s" % KeyError)

#     except Exception as e:

#         return Response("Error Occurred! %s" % e)
#     return Response("Training successfull!!")

# port = int(os.getenv("PORT",5000))
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)
    