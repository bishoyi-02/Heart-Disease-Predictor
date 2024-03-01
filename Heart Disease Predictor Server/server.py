import numpy as np
from flask import Flask, render_template,request,jsonify,url_for,json
import pickle
import json


model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
app = Flask(__name__, static_folder="static")
@app.route('/',methods=['POST','GET'])
def predict():
  
    if(request.method=="POST"):
        
        data = request.json
        print(data)
        data_ = json.loads(json.dumps(data))
        age=data_['age']
        sex=data_['sex']
        cp=data_['cp']
        trestbps=data_['trestbps']
        chol=data_['chol']
        fbs=data_['fbs']
        restecg=data_['restecg']
        thalach=data_['thalach']
        exang=data_['exang']
        oldpeak=data_['oldpeak']
        slope=data_['slope']
        ca=data_['ca']
        thal=data_['thal']
        getData = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        getData = np.reshape(getData,(1,13))
        getData = sc.transform(getData)
        prediction = model.predict(getData)
        if(prediction[0]):
            print("unhealthy")
            return "unhealthy"
        else:
            print("healthy")
            return "healthy"
    return "Hello from server"

if __name__=='__main__':
    app.run(port=5000,host='0.0.0.0')

