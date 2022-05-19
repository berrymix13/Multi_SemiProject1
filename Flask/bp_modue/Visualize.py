from flask import Blueprint, render_template, request, redirect

visual_bp = Blueprint('visual_bp', __name__)
menu = {'ho':0, 'm1':1, 'm2':0, 'm3':0}

@visual_bp.route('/Taxi_move')
def Taxi_move():
    return render_template('Visualize/Taxi_move.html', menu=menu)

@visual_bp.route('/Population')
def Population():
    return render_template('Visualize/Population.html', menu=menu)