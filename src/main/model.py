from flask import Blueprint, request, jsonify
import joblib
import torch
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from model.schedule.schedule import schedule_Tokenizer
from model.region.region import region_Tokenizer

blue_model = Blueprint('model', __name__)

# parameter : 성별, 습도, 풍속, 강수, 기온, 스케줄
@blue_model.route('/clothes', methods=['GET'])
def get_clothes_recommend():
    gender, humidity, wind_speed, rain, temp = map(float, [request.args.get('gender'),
                                                       request.args.get('humidity'),
                                                       request.args.get('wind_speed'),
                                                       request.args.get('rain'),
                                                       request.args.get('temp')])
    schedule = request.args.get('schedule')
    
    try:
        # path
        model_path = os.path.join(get_current_path(), '..', 'model', 'clothes')
        ## schedule le path
        le_schedule_path = os.path.join(model_path, 'le_schedule_3.pkl')
        ## top model path
        clothes_top_model_path = os.path.join(model_path, 'clothes_top_3.pkl')
        le_clothes_top_path = os.path.join(model_path, 'le_clothes_top_3.pkl')
        ## bottom model path
        clothes_bottom_model_path = os.path.join(model_path, 'clothes_bottom_3.pkl')
        le_clothes_bottom_path = os.path.join(model_path, 'le_clothes_bottom_3.pkl')
        
        # label encoder 선언 및 로드
        le_clothes_top, le_clothes_bottom, le_schedule = [LabelEncoder() for _ in range(3)]
        le_clothes_top, le_clothes_bottom = joblib.load(le_clothes_top_path), joblib.load(le_clothes_bottom_path)
        le_schedule = joblib.load(le_schedule_path) 
        
        # 데이터 전처리
        sample = np.array([gender, humidity, wind_speed, rain, temp, schedule]).reshape(-1,1)
        sample[5] = le_schedule.transform(sample[5])
        sample = torch.FloatTensor(sample.astype(float).reshape(1,-1))
        
        # clothes 예측
        ## clothes_top 예측
        clf = joblib.load(clothes_top_model_path)
        clothes_top = le_clothes_top.inverse_transform(torch.argmax(clf(sample)).cpu().numpy().ravel())
        ## clothes_bottom 예측
        clf = joblib.load(clothes_bottom_model_path)
        clothes_bottom = le_clothes_bottom.inverse_transform(torch.argmax(clf(sample)).cpu().numpy().ravel())
        
    except ValueError:
        return "Error: Invalid value for parameter. Must be a float."
    
    return jsonify({
        "top": int(clothes_top[0]),
        "bottom": int(clothes_bottom[0])})
 
@blue_model.route('/schedule', methods=['GET'])
def region_model():
    title = request.args.get('title')
    region = request.args.get('region')
    
    Token = schedule_Tokenizer()
    sample = [Token(title)]
    
    vectorizer = TfidfVectorizer()
    
    model_path = os.path.join(get_current_path(), '..', 'model')
    schedule_vector_path = os.path.join(model_path, 'schedule', 'schedule_vector_8.pkl')
    schedule_clf_path = os.path.join(model_path, 'schedule', 'schedule_clf_8.pkl')
    region_vector_path = os.path.join(model_path, 'region', 'region_vector_27.pkl')
    region_clf_path = os.path.join(model_path, 'region', 'region_clf_27.pkl')
        
    vectorizer = joblib.load(schedule_vector_path)
    sample = vectorizer.transform(sample)
    
    clf = SVC()
    clf = joblib.load(schedule_clf_path)
    
    plan = clf.predict(sample)
    
    Token = region_Tokenizer()
    sample2 = [Token(region)]
    
    vectorizer = joblib.load(region_vector_path)
    sample2 = vectorizer.transform(sample2)
    
    clf = SVC()
    clf = joblib.load(region_clf_path)
    
    region = clf.predict(sample2)
    
    return jsonify({"plan" : plan[0], "region" : region[0]})

def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))