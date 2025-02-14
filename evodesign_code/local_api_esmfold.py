"""
gunicorn -w 1 -b 127.0.0.1:8000 local_api_esmfold_only:app
"""
from flask import Flask, request, jsonify
from evodesign.Prediction.ESMFold import ESMFold



esmfold_model = ESMFold(gpu_device="cuda:0")
esmfold_model.predict_pdb_str('GREETINGS')





app = Flask(__name__)

@app.route("/esmfold", methods=[ "POST" ])
def esmfold():
    data = request.get_json()
    sequence = data["sequence"]
    prediction = esmfold_model.predict_pdb_str(sequence)
    return jsonify({ "pdb": prediction })



if __name__ == '__main__':
    app.run(debug=False)
