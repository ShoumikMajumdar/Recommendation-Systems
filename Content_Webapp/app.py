import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from ContentPlotEngine import Reco_content
from CollaborativePlotEngine import Reco_colab

app = Flask(__name__)
content = Reco_content()
colab = Reco_colab()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/plot',methods=[ 'POST','GET'])
def plot():
    #title = request.form['movie']
    title = request.args.get('movie')
    res = content.get_reco_plot(title)
    return jsonify(res)


@app.route('/crew',methods=[ 'POST','GET'])
def crew():
    #title = request.form['movie']
    title = request.args.get('movie')
    res = content.get_reco_crew(title)
    return jsonify(res)

@app.route('/item',methods=[ 'POST','GET'])
def item():
    #title = request.form['movie']
    title = request.args.get('movie')
    title+=' '
    res = colab.get_recommended(title)
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