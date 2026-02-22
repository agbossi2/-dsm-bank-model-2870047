from flask import Flask, request, Response, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "bankData" / "bank.csv"

# load data
df = pd.read_csv(DATA_PATH, header=None)

# remove campaign columns. No header, so index
df.drop(df.iloc[:, 8:16], inplace=True, axis=1)

X = df.iloc[:, :-1].values # features
y = df.iloc[:, -1].values # target

numeric_features = df.iloc[:, [0, 5]].values

# scale numeric features
scaler = StandardScaler()
scaled_numeric_features = scaler.fit_transform(numeric_features)
numeric_df = pd.DataFrame(scaled_numeric_features, dtype=object, columns=['age', 'balance'])

categoric_features = df.iloc[:, [1, 2, 3, 4, 6, 7]].values

ohe = OneHotEncoder()
enc_categoric_features = ohe.fit_transform(categoric_features).toarray()
categoric_df = pd.DataFrame(enc_categoric_features)

X_final = pd.concat([numeric_df, categoric_df], axis=1)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_final, y)

# Flask instance
app = Flask(__name__)

# create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    # read data from req
    data = request.get_json(force=True)
    data_categoric = np.array([
        data["job"], data["marital"],
        data["education"], data["default"],
        data["housing"], data["loan"]
    ])
    data_categoric = np.reshape(data_categoric, (1, -1))
    
    data_numeric = np.array([data["age"], data["balance"]])
    data_numeric = np.reshape(data_numeric, (1, -1))

    # pre-processing
    data_categoric = ohe.transform(data_categoric).toarray()
    data_numeric = scaler.transform(data_numeric)

    data_final = np.column_stack((data_numeric, data_categoric))
    data_final = pd.DataFrame(data_final, dtype=object)


    predict = model.predict(data_final)
    return Response(json.dumps(predict[0]))

# TODO: VER COMO CORRER ESTO