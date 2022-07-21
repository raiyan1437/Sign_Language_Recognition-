from model import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras.preprocessing
import home, pred, about
pages = {
    "Home": home,
    "Prediction": pred,
    "About": about
}



st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", list(pages.keys()))
if page == None:
    home.app()
else:
    pages[page].app()