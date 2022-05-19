from flask import Flask, render_template, request
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from bp_modue.Visualize import visual_bp
from bp_modue.model1 import time_bp
from xgboost import XGBRegressor
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

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
def car():
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

@app.route('/time', methods=['GET', 'POST'])
def m2_time():
    menu = {'ho': 0, 'm1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'm5': 0, 'm6' : 1}
    if request.method == 'GET':
        return render_template('model1/m2_time.html', menu=menu)

    else:
        index2 = int(request.form['index2'] or '0')
        df2 = pd.read_csv('static/data/time_test.csv')

        test_data2 = df2.iloc[index2, :-1].values.reshape(1, -1)
        label2 = df2.iloc[index2, -1]

        lr2 = joblib.load('static/model/LR_time.pkl')
        svr2 = joblib.load('static/model/SVR_time.pkl')
        rfr2 = joblib.load('static/model/RFR_time_local.pkl')
        dtr2 = joblib.load('static/model/DTR_time.pkl')
        xgr2 = XGBRegressor()  # 모델 초기화
        xgr2.load_model('./static/model/XGBR_time.model')

        pred_lr2 = lr2.predict(test_data2)
        pred_sv2 = svr2.predict(test_data2)
        pred_rf2 = rfr2.predict(test_data2)
        pred_dt2 = dtr2.predict(test_data2)
        pred_xgr2 = xgr2.predict(test_data2)
        result2 = {'index2': index2, 'label2': label2,
                  'pred_lr2': pred_lr2[0], 'pred_sv2': pred_sv2[0], 'pred_rf2': pred_rf2[0], 'pred_dt2':pred_dt2[0], 'pred_xgr2':pred_xgr2[0],
                  'lr_score2':(0.22), 'sv_score2':(0.34), 'dt_score2':(0.14),'rf_score2':(0.43), 'xgr_score2':(0.38)}

        return render_template('model1/m2_time_res.html', menu= menu, res = result2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
if __name__ == '__main__':
    app.run(debug=True)
