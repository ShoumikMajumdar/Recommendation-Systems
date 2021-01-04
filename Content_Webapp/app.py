import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from ContentPlotEngine import Reco_plot
#from ContentPlotEngine import get_director get_list, get_reco, get_reco_genre_credits,get_reco_plot,get_screenplay,clean_data

app = Flask(__name__)
model = Reco_content()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=[ 'POST','GET'])
def predict():
    title = request.form['movie']
    res = model.get_reco_genre_credits(title)
    return jsonify(res)
    

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