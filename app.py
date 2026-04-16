import streamlit as st
import joblib
import pickle
import numpy as np

import psycopg2
# Fetch variables
USER = "postgres.yhvmgdxeegmwyyalfpax" #os.getenv("user")
PASSWORD = "b53n98nv983"# os.getenv("password")
HOST = "aws-1-sa-east-1.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
# Connect to the database
try:
    connection = get_connection()
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)
    # Close the cursor and connection
    cursor.close()

except Exception as e:
    st.write(str(e))

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Función para insertar datos
def insert_prediction(connection, data):
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO tb_iris 
            (longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo, prediccion)
            VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(query, data)
        connection.commit()
        cursor.close()
    except Exception as e:
        st.error(f"Error al insertar: {e}")

# Función para mostrar el historial
def get_history(connection):
    cursor = connection.cursor()

    query = """
        SELECT 
            longitud_sepalo,
            ancho_sepalo,
            longitud_petalo,
            ancho_petalo,
            prediccion,
            created_at
        FROM tb_iris
        ORDER BY created_at DESC;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()

    return rows
        
# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    st.write(result)
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")
            
    try:
        connection = get_connection()

        insert_prediction(connection, (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species
        ))

    except Exception as e:
        st.error(str(e))
        
    st.header("📊 Histórico de predicciones")

    history = get_history(connection)

    if history:
        import pandas as pd

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