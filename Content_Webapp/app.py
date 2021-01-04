import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from reco import get_reco

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=[ 'POST','GET'])
def predict():
    title = request.form['movie']
    print(title)

# @app.route('/predict_api', methods=[ 'POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)