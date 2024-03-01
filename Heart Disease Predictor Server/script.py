
from flask import Flask,request,jsonify, render_template
app =Flask(__name__)
@app.route('/')
def index():
    return render_template('heart_disease_analysis.html')

if __name__=='__main__':
    app.run()