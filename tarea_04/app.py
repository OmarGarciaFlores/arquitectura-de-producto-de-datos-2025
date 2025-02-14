import streamlit as st

# Título de la app
st.title("¡Mi primera app con Streamlit! 🚀")

# Texto introductorio
st.write("Esta es una aplicación sencilla en Streamlit.")

# Entrada de usuario
nombre = st.text_input("¿Cuál es tu nombre?")
if nombre:
    st.write(f"¡Hola, {nombre}! Bienvenido a Streamlit. 😊")

# Botón de acción
if st.button("Presiona aquí"):
    st.write("¡Has presionado el botón!")

# Mostrar número aleatorio
import random
numero = random.randint(1, 100)
st.write(f"Tu número de la suerte es: {numero}")
