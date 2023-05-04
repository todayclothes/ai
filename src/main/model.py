from flask import Blueprint, request, jsonify
import joblib
import torch
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from model.schedule.schedule import Tokenizer
from model.region.region import Tokenizer
from sklearn.ensemble import RandomForestClassifier

blue_model = Blueprint('model', __name__)

# parameter : 계절, 성별, 습도, 풍속, 강수, 기온, 스케줄
@blue_model.route('/clothes', methods=['GET'])
def get_clothes_recommend():
    try:
        season = float(request.args.get('season'))
        gender = float(request.args.get('gender'))
        humidity = float(request.args.get('humidity'))
        wind_speed = float(request.args.get('wind_speed'))
        rain = float(request.args.get('rain'))
        temp = float(request.args.get('temp'))
        schedule = request.args.get('schedule')
    except ValueError:
        return "Error: Invalid value for 'season' parameter. Must be a float."
    
    model_path = os.path.join(get_current_path(), '..', 'model')
    # top model path
    clothes_top_model_path = os.path.join(model_path, 'clothes', 'clothes_top_1.pkl')
    le_clothes_top_path = os.path.join(model_path, 'clothes', 'le_clothes_top_1.pkl')
    le_schedule_top_path = os.path.join(model_path, 'clothes', 'le_schedule_top_1.pkl')
    
    # bottom model path
    clothes_bottom_model_path = os.path.join(model_path, 'clothes', 'clothes_bottom_1.pkl')
    le_clothes_bottom_path = os.path.join(model_path, 'clothes', 'le_clothes_bottom_1.pkl')
    le_schedule_bottom_path = os.path.join(model_path, 'clothes', 'le_schedule_bottom_1.pkl')
    
    # top le
    le_clothes_top = LabelEncoder()
    le_schedule_top = LabelEncoder()
    
    # bottom le
    le_clothes_bottom = LabelEncoder()
    le_schedule_bottom = LabelEncoder()

    # top le load
    le_clothes_top = joblib.load(le_clothes_top_path)
    le_schedule_top = joblib.load(le_schedule_top_path)
    
    # bottom le load
    le_clothes_bottom = joblib.load(le_clothes_bottom_path)
    le_schedule_bottom = joblib.load(le_schedule_bottom_path)
    
    # top classification
    clf = joblib.load(clothes_top_model_path)
    sample = np.array([season, gender, humidity, wind_speed, rain, temp, schedule]).reshape(-1,1)    
    sample[6] = le_schedule_top.transform(sample[6])
    sample = torch.FloatTensor(sample.astype(float).reshape(1,-1))
    clothes_top = le_clothes_top.inverse_transform(torch.argmax(clf(sample)).cpu().numpy().ravel())
    
    # bottom classification
    clf = joblib.load(clothes_bottom_model_path)
    sample = np.array([season, gender, humidity, wind_speed, rain, temp, schedule]).reshape(-1,1)    
    sample[6] = le_schedule_bottom.transform(sample[6])
    sample = torch.FloatTensor(sample.astype(float).reshape(1,-1))
    clothes_bottom = le_clothes_bottom.inverse_transform(torch.argmax(clf(sample)).cpu().numpy().ravel())
    
    return jsonify({
        "top" : int(clothes_top[0]),
        "bottom" : int(clothes_bottom[0])})
 
@blue_model.route('/schedule', methods=['GET'])
def region_model():
    title = request.args.get('title')
    region = request.args.get('region')
    
    kiwi = Kiwi()
    tokenizer = Tokenizer()
    sample = [tokenizer(title)]
    
    vectorizer = TfidfVectorizer()
    
    model_path = os.path.join(get_current_path(), '..', 'model')
    vector_kiwi_3_path = os.path.join(model_path, 'schedule', 'vector_kiwi_3.pkl')
    clf_kiwi_3_path = os.path.join(model_path, 'schedule', 'clf_kiwi_3.pkl')
    region_vector_20_path = os.path.join(model_path, 'region', 'region_vector_20.pkl')
    region_clf_20_path = os.path.join(model_path, 'region', 'region_clf_20.pkl')
        
    vectorizer = joblib.load(vector_kiwi_3_path)
    sample = vectorizer.transform(sample)
    
    clf = RandomForestClassifier()
    clf = joblib.load(clf_kiwi_3_path)
    
    plan = clf.predict(sample)
    
    sample2 = [tokenizer(region)]
    
    vectorizer = joblib.load(region_vector_20_path)
    sample2 = vectorizer.transform(sample2)
    
    clf = SVC()
    clf = joblib.load(region_clf_20_path)
    
    region = clf.predict(sample2)
    
    return jsonify({"plan" : plan[0], "region" : region[0]})

def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))