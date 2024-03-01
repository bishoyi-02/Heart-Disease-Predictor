import requests
import json
url = 'http://localhost:5000/'
resp = requests.post(url,json={"age":66,"sex":0,"cp":0,"trestbps":178,
                            "chol":228,"fbs":1,"restecg":1,"thalach":165,"exang":1,
                            "oldpeak":1,"slope":1,"ca":2,"thal":3})
print(resp.json())

