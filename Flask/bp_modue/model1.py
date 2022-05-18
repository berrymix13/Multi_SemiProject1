from flask import Blueprint, render_template, request, redirect

m1_bp = Blueprint('m1_bp', __name__)
menu = {'ho':0, 'm1':0, 'm2':1, 'm3':0}

@m1_bp.route('/Taxi_move')
def m1_ttest():
    if request.method == 'GET':
        return render_template('classify.html', menu=menu)
    else:
        index = int(request.form['index'] or '0')
        df = pd.read_csv('static/data/titanic_test.csv')
        scaler = joblib.load('static/model/titanic_scaler.pkl')
        test_data = df.iloc[index, :-1].values.reshape(1, -1)
        test_scaled = scaler.transform(test_data)
        label = df.iloc[index, 0]
        lrc = joblib.load('static/model/titanic_lr.pkl')
        svc = joblib.load('static/model/titanic_sv.pkl')
        rfc = joblib.load('static/model/titanic_rf.pkl')
        pred_lr = lrc.predict(test_scaled)
        pred_sv = svc.predict(test_scaled)
        pred_rf = rfc.predict(test_scaled)
        result = {'index': index, 'label': label,
                  'pred_lr': pred_lr[0], 'pred_sv': pred_sv[0], 'pred_rf': pred_rf[0]}

        tmp = df.iloc[index, 1:].values
        value_list = []
        int_index_list = [0, 1, 3, 4, 6, 7]
        for i in range(8):
            if i in int_index_list:
                value_list.append(int(tmp[i]))
            else:
                value_list.append(tmp[i])
        org = dict(zip(df.columns[1:], value_list))
        return render_template('classify_res.html', menu=menu, res=result, org=org)

@m1_bp.route('/m1_car')
def Population():
    return render_template('module/m1_car.html', menu=menu)