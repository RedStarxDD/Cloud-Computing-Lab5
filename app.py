import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# Variables de conexión
USER = "postgres.yhvmgdxeegmwyyalfpax"
PASSWORD = "b53n98nv983"
HOST = "aws-1-sa-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Conexión cacheada
@st.cache_resource
def get_connection():
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    conn.autocommit = True
    return conn

# Configuración de página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Obtener conexión UNA SOLA VEZ
connection = get_connection()

# Cargar modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo")
        return None, None, None

# Insertar datos
def insert_prediction(connection, data):
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO ml.tb_iris
            (longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo, prediccion)
            VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(query, data)
        cursor.close()
    except Exception as e:
        st.error(f"Error al insertar: {e}")

# Obtener historial
def get_history(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT 
                longitud_sepalo,
                ancho_sepalo,
                longitud_petalo,
                ancho_petalo,
                prediccion,
                created_at
            FROM ml.tb_iris
            ORDER BY created_at DESC;
        """)
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except Exception as e:
        st.error(str(e))
        return []

# UI
st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", 0.0, 10.0, 3.0, 0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", 0.0, 10.0, 1.0, 0.1)

    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        confidence = float(max(probabilities))

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")

        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # Insertar en BD (CORRECTO: dentro del botón)
        insert_prediction(connection, (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species
        ))

    # Historial
    st.header("📊 Histórico de predicciones")

    history = get_history(connection)

    if history:
        df = pd.DataFrame(history, columns=[
            "Longitud del Sépalo",
            "Ancho del Sépalo",
            "Longitud del Pétalo",
            "Ancho del Pétalo",
            "Especie",
            "Fecha"
        ])
        st.dataframe(df)
    else:
        st.info("No hay registros aún.")