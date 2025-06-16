#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import argparse


# ---------------------------
# CLI argument parser
# ---------------------------
parser = argparse.ArgumentParser(description="NYC Taxi Duration Prediction")
parser.add_argument("--year", type=int, required=True, help="Year of the trip data")
parser.add_argument("--month", type=int, required=True, help="Month of the trip data")
args = parser.parse_args()
year = args.year
month = args.month

# ---------------------------
# Load model
# ---------------------------
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# ---------------------------
# Read and preprocess data
# ---------------------------
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(url)

# ---------------------------
# Run inference
# ---------------------------
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# ---------------------------
# Prepare results
# ---------------------------
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

df_result.to_parquet(
    'predictions.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)

print("Mean predicted duration:", np.mean(y_pred))
