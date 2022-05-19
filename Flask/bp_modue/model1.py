from flask import Blueprint, render_template, request, redirect
import joblib
import pandas as pd
from xgboost import XGBRegressor

time_bp = Blueprint('time_bp', __name__)
menu = {'ho':0, 'm1':0, 'm2':0, 'm3':1}

@time_bp.route('/hist')
def hist():
    return render_template('model1/hist.html', menu=menu)

@time_bp.route('/K_Means')
def K_Means():
    return render_template('model1/K_Means.html', menu=menu)

@time_bp.route('/m2_time')
def m2_time():
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
    return render_template('model1/m2_time_res.html', menu=menu, org = org)
