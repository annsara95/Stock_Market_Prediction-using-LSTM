from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask_cors import CORS
from flask import jsonify
import os
import numpy as np
import pandas as pd
from flask_restful import Resource, Api
import json
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
app = Flask(__name__)
CORS(app)


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def denormalize(normalized_value): 

    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = MinMaxScaler()
    
    old = min_max_scaler.fit(normalized_value)
    
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

@app.route('/csv/')  
def download_csv():
  
    response = make_response(predictions)
    #d = 'attachment; filename=mycsv.csv'
    #esponse.headers['Content-Disposition'] = cd 
    #esponse.mimetype='text/csv'

    return response

@app.route('/upload')
def upload_filepage():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
      if request.method == 'POST':
          f = request.files['file']
          f.save(os.path.join(secure_filename(f.filename)))
          #downloads3_file()
          df = pd.read_csv(os.path.join(secure_filename(f.filename)))
          df = df.set_index(['date'])
          x_train, y_train, x_test, y_test = load_data(df, 20)
          
          model = load_model(os.path.join('my_model.h5'))
            
          predictions = model.predict(x_test)
        
          #redictions = denormalize(predictions)
            
          predictions = pd.DataFrame(predictions)
          
          predictions.to_csv(os.path.join('Predictions.csv'),index=False)
          print("Prediction Done")
          
          #jsonfile = jsonify(**modeldict)
          return 'file uploaded successfully and prediction will be given in csv'

@app.route('/',methods=['GET'])
def predictform():
        return render_template('predictform.html')

@app.route('/predict',methods=['POST'])
def predict():
        content = request.json      
        lstm  = doPrediction(content)
        return jsonify(lstm)

def doPrediction(content):  

        filename = 'my_model.h5'
        
        print("Loading Models")
       
        model = load_model(os.path.join('my_model.h5'))

        # Get the data

        print("Data Fetched")

 
        close = content['param1']
        high = content['param2']
        low = content['param3']
        volume = content['param4']
        prevday_open = content['param5']

        print("Data putting to array")

        x=np.array([close,volume,prevday_open,high,low]).reshape(1,-1)

        print("Data ready for prediction") 

        lstm=model.predict(x)

        print("lstm done")
        
        print("Prediction Done")
                
        return lstm
        
if __name__ == '__main__':
   app.run(debug = True)