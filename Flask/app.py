from flask import Flask, render_template, request
from flask import current_app
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from bp_modue.module import module_bp
from datetime import datetime
import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# render_template  : 페이지 이동
# request  : 주고받기
# url_for  :

app = Flask(__name__)
app.register_blueprint(module_bp, url_prefix='/module')

@app.route('/')
def index():
    menu = {'ho': 1, 'm1': 0, 'm2': 0, 'm3': 0, 'cf': 0, 'cu': 0}
    return render_template('index.html', menu = menu)

@app.route('/menu1', methods=['GET', 'POST'])
def menu1():
    menu = {'ho':0, 'm1':1, 'm2':0, 'm3':0, 'cf':0, 'cu':0}
    if request.method == 'GET':
        return render_template('menu1.html', menu=menu)
    else:
        text = request.form['text']
        review = request.form['review'].replace('\n','<br>')
        lang = request.form['lang']
        return render_template('menu1_res.html', menu=menu,
                               text=text, review=review, lang=lang)

@app.route('/menu2')
def menu2():
    menu = {'ho':0, 'm1':0, 'm2':1, 'm3':0, 'cf':0, 'cu':0}
    items = [
        {'id':1001, 'title':'HTML', 'content':'HTML is HyperText ...'},
        {'id':1002, 'title':'CSS', 'content':'CSS is Cascading ...'},
        {'id':1003, 'title':'JS', 'content':'JS is Javascript ...'},
    ]
    now = datetime.now()
    np.random.seed(now.microsecond)
    X = np.random.rand(100)
    Y = np.random.rand(100)
    plt.figure()
    plt.scatter(X, Y)
    img_file = os.path.join(current_app.root_path, 'static/img/menu2.png')
    plt.savefig(img_file)
    mtime = int(os.stat(img_file).st_mtime)

    return render_template('menu2.html', menu=menu, mtime=mtime,
                            now=now.strftime('%Y-%m-%d %H:%M:%S.%f'), items=items)


if __name__ == '__main__':
    app.run(debug=True)
