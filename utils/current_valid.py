import sys
import time
import os
import glob
from time import sleep
import datetime
import pytz
import argparse

import numpy as np
import pandas as pd
import pickle
import joblib

import scipiy
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from Current_Feature_Extractor import Extract_Time_Features, Extract_Phase_Features, Extract_Freq_Features

    
def framework():
    print('########## 전류 유효 검증 시작 #############')
    print('시작시간(UCT+09:00)',datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))
    print('_Python.version:', sys.version)
    print('_Scipy.version:',scipy.__version__)
    print('_XGBoost.version:',xgb.__version__)
    print('_Sklearn.version:',sklearn.__version__)
    
def validation(filename):
        
    try:
        cur = pd.read_csv(filename, header=None, skiprows=9)
        cur = np.asarray(cur)[:,1:4].transpose()

        meta = pd.read_csv(filename, header=None, skiprows=4, nrows=1)
        rpm = int(meta[2])

        fs = pd.read_csv(filename, header=None, skiprows=6, nrows=1)
        Fs = int(fs[1])

        TimeFeatureExtractor = Extract_Time_Features(cur)
        features_time = TimeFeatureExtractor.Features().flatten()

        PhaseFeatureExtractor = Extract_Phase_Features(cur, Fs)
        features_phase = PhaseFeatureExtractor.Features().flatten()

        FreqFeatureExtractor = Extract_Freq_Features(cur, rpm, Fs)
        features_freq = FreqFeatureExtractor.Features().flatten()

        features = np.concatenate((features_time, features_phase, features_freq))
        data = pd.DataFrame(features).T

        data.columns = ['R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
               'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
               'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
               'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
               'R_Impulse', 'S_Impulse', 'T_Impulse',
               'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
               'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
               'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']

        data['WATT'] = meta[3]
        y = pd.read_csv(filename, header=None, skiprows=3, nrows=1)
        y = y[1]

    except (NameError, IndexError, TypeError, pd.errors.EmptyDataError):
        print('Source Data Error')
        
    else:
        df = data[['WATT', 'R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                   'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                   'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                   'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                   'R_Impulse', 'S_Impulse', 'T_Impulse',
                   'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                   'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                   'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']]

        #norm = np.load('current_norm.npy')#norm
        #X = (df - norm[0]) / norm[1]

        xgb_joblib = joblib.load('current_xgb.pkl') 
        y_pred = xgb_joblib.predict(df)
        name = os.path.basename(filename)
        print("Filename:%s, Prediction:%s, Y_Target:%d" % (name, y_pred[0], y))
        
    return y_pred[0], y[0]

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='current', type=str)
args = parser.parse_args()
datapath = glob.glob(args.path + '**/**/*.csv', recursive=True)

framework()
#random = random.sample(datapath, 100)

y_hat = []
y_target = []
for file in datapath:
    y_pred, y = validation(file)
    y_hat.append(y_pred)
    y_target.append(y)
    
target_names = ['0', '1', '2', '3', '4']
print('########## 전류데이터 유효검증 결과 ##########')
print(f"검증데이터수: {len(y_hat)} 개")

accuracy = accuracy_score(y_target, y_hat)
print('예측 정확도: {0:.4f}'.format(accuracy))
print(classification_report(y_target, y_hat, target_names=target_names, digits=4)) 
print('종료시간(UCT+09:00)',datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))

end = time.time()
elapsed_time = end-start
print(f"검증소요시간：{elapsed_time}초")

y_prediction = pd.DataFrame(y_hat)
y_prediction.to_csv('y_predict_current.csv', index=False)