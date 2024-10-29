import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys

from sklearn import set_config
set_config(transform_output='pandas')


## Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)

# Define the load raw eda data function with caching
@st.cache_data
def load_data():
    fpath = FPATHS['data']['raw']['eda']
    df = pd.read_csv(fpath)
    df = df.set_index("PID")
    return df

# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(FPATHS, data_type='test'):
    filepath = FPATHS['data']['ml'][data_type]
    return joblib.load(filepath)

@st.cache_resource
def load_model_ml(FPATHS,type='linear_regression'):
    filepath = FPATHS['models'][type]
    return joblib.load(filepath)

### Start of App
st.title('House Prices in Ames, Iowa')

# Include the banner image
st.image(FPATHS['images']['banner'])

# Load & cache dataframe
df = load_data()

# Load training data
X_train, y_train = load_Xy_data(FPATHS, data_type='train')
# Load testing data
X_test, y_test = load_Xy_data(FPATHS, data_type='test')

# Load models
linreg = load_model_ml(FPATHS, type='linear_regression')


# Add text for entering features
st.subheader("Select values using the sidebar on the left.\n Then check the box below to predict the price.")
st.sidebar.subheader("Enter House Features For Prediction")

# # Create widgets for each feature
selected_sqft = st.sidebar.number_input('Living Area Sqft (100-6000 sqft)', min_value=100, max_value=6000, step = 100)
selected_lot_front = st.sidebar.number_input('Lot Frontage (20-350 ft)',min_value=20, max_value=350, step = 10)
selected_bldg_type = st.sidebar.selectbox('Bldg Type', options=(df['Bldg Type'].unique()))
selected_bedrooms = st.sidebar.slider('Bedroom', min_value=0, max_value=df['Bedroom'].max())
selected_baths = st.sidebar.slider("Total Full Baths", min_value=0, max_value=6, step = 1)
selected_ms_zoning = st.sidebar.selectbox("MS Zoning", options=df['MS Zoning'].unique())
selected_street = st.sidebar.selectbox("Street", options=df['Street'].unique())
selected_alley = st.sidebar.selectbox("Alley", options=df['Alley'].unique())
selected_utilities = st.sidebar.selectbox("Utilities", options=df['Utilities'].unique())

# Define function to convert widget values to dataframe
def get_X_to_predict():
    X_to_predict = pd.DataFrame({'Living Area Sqft': selected_sqft,
                             'Lot Frontage': selected_lot_front, 
                             'Bldg Type': selected_bldg_type,
                             'Bedroom': selected_bedrooms,
                             'Total Full Baths':selected_baths,
                             'MS Zoning':selected_ms_zoning,
                             'Street':selected_street,
                             "Alley":selected_alley,
                             "Utilities":selected_utilities},
                             index=['New'])
    return X_to_predict

def get_prediction(model,X_to_predict):
    return  model.predict(X_to_predict)[0]


from lime.lime_tabular import LimeTabularExplainer

# Don't hash by adding _ in front of model_pipe
@st.cache_resource
def get_explainer(_model_pipe, X_train):
    X_train_tf = pd.DataFrame(_model_pipe[0].transform(X_train))
    X_train_tf.columns = _model_pipe[0].get_feature_names_out()

    explainer = LimeTabularExplainer(X_train_tf.values,
                                 feature_names=X_train_tf.columns,                                    
                                    random_state=42,
                                     mode='regression')

    return explainer

# Define explainer object with function
explainer = get_explainer(linreg, X_train)

def explain_instance(explainer, _model_pipe, X_to_explain):
    X_to_explain_tf = pd.DataFrame(_model_pipe[0].transform(X_to_explain))
    X_to_explain_tf.columns = _model_pipe[0].get_feature_names_out()

    explanation = explainer.explain_instance(X_to_explain_tf.iloc[0],
                                             _model_pipe[-1].predict)
    return explanation

import streamlit.components.v1 as components

# This is modified to include displaying the LIME explanation
if st.checkbox("Predict"):
    
    X_to_pred = get_X_to_predict()
    new_pred = get_prediction(linreg, X_to_pred)
    
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
    
    explanation = explain_instance(explainer, linreg, X_to_pred)
    components.html(explanation.as_html(show_predicted_value=False))
else:
    st.empty()

components.html(explanation.as_html(show_predicted_value=False), height = 800)

