import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('convlstm_model___Date_Time_2022_06_24__08_08_47___Loss_0.8155478835105896___Accuracy_0.6747967600822449 (1).h5')
  #model = tf.keras.models.load_model('convlstm_model___Date_Time_2022_07_23__02_40_22___Loss_0.18381743133068085___Accuracy_0.8834951519966125.h5')
  return model
