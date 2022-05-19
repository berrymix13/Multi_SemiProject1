from flask import Flask, render_template, request
from sklearn.metrics import r2_score, mean_squared_error
from flask import current_app
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from bp_modue.Visualize import visual_bp
from bp_modue.model1 import time_bp
from xgboost import XGBRegressor
from datetime import datetime
import pickle
import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# render_template  : 페이지 이동
# request  : 주고받기
# url_for  :

app = Flask(__name__)
#app.register_blueprint(visual_bp, url_prefix='/Visualize')
#app.register_blueprint(time_bp, url_prefix='/model1')

@app.route('/')
def index():
    menu = {'ho': 1, 'm1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 0, 'm6' : 0}
    return render_template('index.html', menu = menu)

@app.route('/Taxi_move')
def Taxi_move():
    menu = {'ho': 0, 'm1': 1, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 0, 'm6': 0}
    return render_template('Visualize/Taxi_move.html', menu=menu)

@app.route('/Population')
def Population():
    menu = {'ho': 0, 'm1': 0, 'm2': 1, 'm3': 0, 'm4': 0, 'm5': 0, 'm6': 0}
    return render_template('Visualize/Population.html', menu=menu)

@app.route('/taxi/car', methods=['GET', 'POST'])
def classify():
    menu = {'ho': 0, 'm1': 0, 'm2': 0, 'm3': 1, 'm4': 0, 'm5': 0, 'm6' : 0}
    if request.method == 'GET':
        return render_template('module/m1_car.html', menu=menu)
    else:
        index = int(request.form['index'] or '0')
        df = pd.read_csv('static/data/car_test.csv')
        scaler = joblib.load('static/model/car_scaler.pkl')
        test_data = df.iloc[index, :-1].values.reshape(1, -1)
        test_scaled = scaler.transform(test_data)
        label = df.iloc[index, -1]

        x_test_sc = scaler.transform(np.array(df[['차량운행','접수건','탑승건']]))
        y_test = df[['평균대기시간']]
        lr = joblib.load('static/model/LR_car.pkl')
        lr_score = r2_score(y_test, lr.predict(x_test_sc))
        lr_mse = mean_squared_error(y_test, lr.predict(x_test_sc))

        svr = joblib.load('static/model/svr_car.pkl')
        svr_score = r2_score(y_test, svr.predict(x_test_sc))
        svr_mse = mean_squared_error(y_test, svr.predict(x_test_sc))

        rfr = joblib.load('static/model/rfr_car.pkl')
        rfr_score = r2_score(y_test, rfr.predict(x_test_sc))
        rfr_mse = mean_squared_error(y_test, rfr.predict(x_test_sc))

        xgr = joblib.load('static/model/xgr_car.pkl')
        xgr_score = r2_score(y_test, xgr.predict(x_test_sc))
        xgr_mse = mean_squared_error(y_test, xgr.predict(x_test_sc))

        pred_lr = lr.predict(test_scaled)
        pred_sv = svr.predict(test_scaled)
        pred_rf = rfr.predict(test_scaled)
        pred_xgr = xgr.predict(test_scaled)

        result = {'index':index, 'label':label,
                  'pred_lr':round(pred_lr[0],2), 'pred_sv': round(pred_sv[0],2), 'pred_rf':round(pred_rf[0],2), 'pred_xgr':round(pred_xgr[0],2),
                  'lr_score':round(lr_score,2), 'sv_score':round(svr_score,2), 'rf_score':round(rfr_score,2), 'xgr_score':round(xgr_score,2),
                  'lr_mse':round(lr_mse,2), 'sv_mse':round(svr_mse,2), 'rf_mse':round(rfr_mse,2), 'xgr_mse':round(xgr_mse,2)}

        return render_template('module/m1_car_res.html', menu=menu, res=result)
@app.route('/hist')
def hist():
    menu = {'ho': 0, 'm1': 0, 'm2': 0, 'm3': 0, 'm4': 1, 'm5': 0, 'm6' : 0}
    return render_template('model1/hist.html', menu=menu)

@app.route('/K_Means')
def K_Means():
    menu = {'ho': 0, 'm1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 1, 'm6' : 0}
    return render_template('model1/K_Means.html', menu=menu)

@app.route('/m2_time')
def m2_time():
    menu = {'ho': 0, 'm1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 0, 'm6' : 1}
    if request.method == 'GET':
        return render_template('model1/m2_time.html', menu=menu)
    else:
        index = int(request.form['index'] or '0')
        df = pd.read_csv('static/data/time_test.csv')

        test_data = df.iloc[index, :-1].values.reshape(1, -1)
        label = df.iloc[index, -1]

        lr = joblib.load('static/model/LR_time.pkl')
        svr = joblib.load('static/model/SVR_time.pkl')
        rfr = joblib.load('static/model/RFR_time.pkl')
        dtr = joblib.load('static/model/DTR_time.pkl')
        xgr = joblib.load('static/model/XGBR_time.pkl')

        pred_lr = lr.predict(test_data)
        pred_sv = svr.predict(test_data)
        pred_rf = rfr.predict(test_data)
        pred_dt = dtr.predict(test_data)
        pred_xgr = xgr.predict(test_data)
        result = {'index': index, 'label': label,
                  'pred_lr': pred_lr[0], 'pred_sv': pred_sv[0], 'pred_rf': pred_rf[0], 'pred_dt':pred_dt[0], 'pred_xgr':pred_xgr[0],
                  'lr_score':(0.22), 'sv_score':(0.34), 'dt_score':(0.14),'rf_score':(0.43), 'xgr_score':(0.38)}

        org = dict(zip(df.columns[:-1], test_data))
    return render_template('model1/m2_time_res.html', menu=menu, org = org, res = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
if __name__ == '__main__':
    app.run(debug=True)
