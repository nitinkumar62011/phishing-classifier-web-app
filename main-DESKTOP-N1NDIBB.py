from wsgiref import simple_server
from flask import Flask, request, render_template,jsonify
from flask import Response
import os
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil

from sqlalchemy import column
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
#predict_data=pd.read_csv("./Prediction_Output_File/Predictions.csv")
#print("prediction",predict_data)

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


@app.route("/predicturl", methods=["GET", "POST"])
def predicturl():
    
    if request.method == "POST":

        url = request.form["url"]
        print(url)
        print(generate_data_set(url))
        x = np.array(generate_data_set(url)).reshape(1,30)
        print(x)
        x5=[x]
        print(x5)
    
        
    
        columnData=["having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol","double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State","Domain_registeration_length","Favicon","port","HTTPS_token","Request_URL","URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email","Abnormal_URL","Redirect","on_mouseover","RightClick","popUpWidnow","Iframe","age_of_domain","DNSRecord","web_traffic","Page_Rank","Google_Index","Links_pointing_to_page","Statistical_report"]
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        url_csv_file="phishing_"+str(date).replace('-','')+"_"+str(time)+".csv"
        data1=pd.DataFrame(x,columns=columnData)
        os.makedirs("./Prediction_Batch_Files",exist_ok=True)
        path_url="./Prediction_Batch_Files"
        print(path_url)
        
        data1.to_csv(os.path.join(path_url,url_csv_file), index=None, header=True)
        if os.path.isdir(path_url): #remove previously existing models for each clusters
            shutil.rmtree(path_url)
            os.makedirs(path_url,exist_ok=True)
        else:
            os.makedirs(path_url,exist_ok=True) #

        print(x)
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
            #print("file name",type(request.form['folder']))
            f = request.files['file1']
            urlFileFolder="C:\\Users\\nitin\\OneDrive\\Desktop\\phishing classifier\\code\\url_data_File_converted"
            f.save(os.path.join(urlFileFolder,secure_filename(f.filename)))
            fileFetch=pd.read_csv(os.path.join(urlFileFolder,secure_filename(f.filename)))
            print("fileFetch",fileFetch)
            allEncodedUrl=[]
            for i in range(len(fileFetch)):
                allEncodedUrl.append(generate_data_set(fileFetch.iloc[i,0]))
            print("allEncodedUrl",allEncodedUrl)
            columnData=["having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol","double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State","Domain_registeration_length","Favicon","port","HTTPS_token","Request_URL","URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email","Abnormal_URL","Redirect","on_mouseover","RightClick","popUpWidnow","Iframe","age_of_domain","DNSRecord","web_traffic","Page_Rank","Google_Index","Links_pointing_to_page","Statistical_report"]
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H%M%S")
            url_csv_file="phishing_"+str(date).replace('-','')+"_"+str(time)+".csv"
            data1=pd.DataFrame(allEncodedUrl,columns=columnData)
            os.makedirs("./Prediction_Batch_Files",exist_ok=True)
            path_url="./Prediction_Batch_Files"
            print(path_url)
        
            data1.to_csv(os.path.join(path_url,url_csv_file), index=None, header=True)






            path="./Prediction_Batch_Files/"
            
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
            print("predict from")
            print(predict_data)
            fileFetch["prediction_probabilities"]=predict_data["Predictions"]
        
            print("data1",fileFetch)
            finalListData=[]
            for i in range(len(fileFetch)):
                temp=[]
                temp.append(fileFetch.iloc[i,0])
                temp.append(fileFetch.iloc[i,1])
                finalListData.append(temp)
            print("finalListData",finalListData)

            path ="./Prediction_Batch_Files/"
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(path)
                os.makedirs(path,exist_ok=True)
            else:
                os.makedirs(path,exist_ok=True) #




            


        except Exception as e:
            print(e)
    return render_template("result.html", predict_data_total=finalListData)
        
@app.route("/retraining", methods=["GET", "POST"])
def retraining():
    try:
        path=request.form['retraing_path']
        train_valObj = train_validation(path) #object initialization

        train_valObj.train_validation()#calling the training_validation function


        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        #return Response("Error Occurred! %s" % ValueError)
        return render_template("index.html")

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        #return Response("Error Occurred! %s" % e)
        return render_template("index.html")

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

# port = int(os.getenv("PORT",8080))
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8080, debug=True)
    