#import
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import dump, load
import csv
import codecs
from io import StringIO
from fastapi import FastAPI, File, UploadFile
# from parse_csv import convert
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

#%matplotlib inline


from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response
from fastapi.responses import ORJSONResponse


import datetime
import janitor
from janitor import groupby_agg

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

app = FastAPI(title="anomaly detection Project",description="This is a very fancy project")


# load the pre-trained Keras model
def define_model():
    global model
    model = load_model('/Users/mohammad/Desktop/source/anomaly_detection_model.h5')
    return print("Model Loaded")



@app.get("/")
async def root():
    return {"message": "anomaly detection project"}




@app.post("/files/" )#, response_class=ORJSONResponse)
async def create_file(file: UploadFile = File(...)):
    data_out = {}

    if not file:
        return "No file submitted"

    # read data
    data = await file.read()
    data = data.decode('utf8')

    #replace & split
    data = data.replace("]", "[")
    data = StringIO(data)
    data = pd.read_csv(data, delimiter='[', error_bad_lines=False, )
    data.columns = ['IP', 'time', 'a', 'methodpath', 'statuscode&responselength', 'b', 'useragent', 'c', 'responsetime']
    new = data['statuscode&responselength'].str.split(" ", n=2, expand=True)
    data["statuscode"] = new[1]
    data["responselength"] = new[2]
    data.drop(columns=["statuscode&responselength"], inplace=True)

    new = data["methodpath"].str.split(" ", n=2, expand=True)

    data["method"] = new[0]
    data["path"] = new[1]
    data.drop(columns=["methodpath"], inplace=True)
    data.drop(columns=["a", "b", "c"], inplace=True)

     #feature extraction "hour"
    data["time"] = pd.to_datetime(data["time"])
    data["hour"] = data["time"].dt.hour - data["time"].dt.hour.max() + 10

    #feature extraction "request"
    data['request1'] = 1
    filtered_df = data.groupby_agg(by=['hour', 'IP', 'useragent'],
                                   agg='size',
                                   agg_column_name='request1',
                                   new_column_name='request'
                                   )
    rslt_df = filtered_df[filtered_df['request'] > 11]
    rslt_df = rslt_df.drop(['request1'], axis=1)

    # encode statuscode
    rslt_df['statuscode'] = rslt_df['statuscode'].astype(int) // 100

    #missing value
    rslt_df = rslt_df[rslt_df['IP'] != '- ']
    rslt_df['responsetime'].loc[(rslt_df['responsetime'] == ' -')] = 0

    #feature extract
    rslt_df['depth'] = rslt_df['path'].str.count('/')
    rslt_df['robot'] = rslt_df['path'].str.contains("robots.txt")
    rslt_df['robot'] = rslt_df['robot'] * 1
    rslt_df['robot'].value_counts()

    rslt_df_3 = rslt_df.copy()

    rslt_df_3['time'] = pd.to_datetime(rslt_df_3['time'])

    #scaling
    scaler = load("/Users/mohammad/Desktop/source/scaler_data")
    qt = load("/Users/mohammad/Desktop/source/qt_data")
    qt1 = load("/Users/mohammad/Desktop/source/qt1_data")
    ohe = load("/Users/mohammad/Desktop/source/ohe_data")
    ohe1 = load("/Users/mohammad/Desktop/source/ohe1_data")
    result = ohe.transform(rslt_df_3[['statuscode']].astype(np.float))
    result2 = ohe1.transform(rslt_df_3[['method']])

    a1 = pd.DataFrame(result, columns=ohe.categories_, index=rslt_df.index)
    a2 = pd.DataFrame(result2, columns=ohe.categories_, index=rslt_df.index)
    rslt_df_3 = pd.concat((rslt_df_3, a2), axis=1)
    rslt_df_3 = pd.concat((rslt_df_3, a1), axis=1)
    rslt_df_3['depth'] = scaler.transform(rslt_df_3[['depth']])
    rslt_df_3['responsetime'] = qt.transform(rslt_df_3[['responsetime']])
    rslt_df_3['responselength'] = qt1.transform(rslt_df_3[['responselength']])


    #drop columns
    rslt_df_3.drop(columns=["statuscode", "method", "path", "request"], inplace=True)

    # feature extraction "ip_agent"
    rslt_df_3['ip_agent'] = rslt_df_3.IP + ':' + rslt_df_3.useragent
    df_train = rslt_df_3.set_index(['ip_agent', 'time']).sort_index()
    df_train.drop(columns=["IP", "useragent", "hour"], inplace=True)
    #structure
    ds_np_train = df_train.groupby(level=0).apply(lambda x: x.values).values
    ds_np_train = [x[-4:][np.newaxis, :, :] for x in ds_np_train]
    ds_np_train = np.concatenate(ds_np_train, axis=0)

    model = load_model('/Users/mohammad/Desktop/source/anomaly_detection_model.h5')
    #prediction
    ds_np_train_pred = model.predict(ds_np_train, verbose=0)
    train_mae_loss = np.mean(np.abs(ds_np_train_pred - ds_np_train), axis=(1, 2))
    print(train_mae_loss)
    anomalies1 = train_mae_loss > 0.18
    anomalies1.shape
    unique_elements, counts_elements = np.unique(anomalies1, return_counts=True)
    print (unique_elements, counts_elements)
    result1 = np.where(anomalies1)
    print(result1)



    if len(result1) > 0:
        out=[]
        for j in result1[0]:
            out.append(df_train.index.levels[0][j])

        out = json.dumps(out)
        return out

    else:
        return{"Anomaly": "No Anomalies Detected"}









if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000, debug=True)
