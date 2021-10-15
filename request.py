import requests
from model import *


url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'location':'1st Phase JP Nagar',sqft':1000, 'bath':2,'bhk':2})

print(r.json())
