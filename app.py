import streamlit as st
import joblib
import pandas as pd

# Load models
model_v1 = joblib.load('/content/iris_mlp_model.joblib')
model_v2 = joblib.load('/content/iris_mlp_model_v2.joblib')

st.title('Clasificador de Flores Iris')
st.write('Ingrese las dimensiones de la flor para obtener la predicción.')

# Editable input fields
sepal_l = st.number_input('Sepal Length', value=5.1)
sepal_w = st.number_input('Sepal Width', value=3.5)
petal_l = st.number_input('Petal Length', value=1.4)
petal_w = st.number_input('Petal Width', value=0.2)

if st.button('Predecir'):
    # Prepare input data
    input_df = pd.DataFrame([[sepal_l, sepal_w, petal_l, petal_w]], 
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Predictions
    pred1 = model_v1.predict(input_df)[0]
    pred2 = model_v2.predict(input_df)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Modelo V1')
        st.success(f'Especie: {pred1}')
    with col2:
        st.subheader('Modelo V2')
        st.success(f'Especie: {pred2}')
