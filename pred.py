from model import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras.preprocessing
import cv2
import math
import random

def predict_single_actionlstm(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''  
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities =  convlstm_model.predict(video_bytes)[0]
    print(predicted_labels_probabilities)
    
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    print(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    video_reader.release()

def app():
    with st.spinner('Model is being loaded..'):
        model = load_model()
    st.subheader("""
    **NOTE:** This app work best only when you uplode image of a Cat or Dog.""")
    file = st.file_uploader("Please upload a Video of ASL Sign which You want to Translate")
    st.set_option('deprecation.showfileUploaderEncoding', True)

    if file is None:
        st.write("""Please upload an Video file""")
    else:
        if st.button("Predict"):
            try:
                video_file = open('file', 'rb')            
                video_bytes = video_file.read()
                
                st.video(video_bytes)
                #vid = Video.open(file)            
                #st.video(vid, use_column_width=True)
                #test_vid = tf.image.resize(pic, [64, 64])
                #img = keras.preprocessing.image.img_to_array(test_img)
                #img = np.expand_dims(img, axis=0)/*
                cnn = model.predict(video_bytes)
                if cnn[0][0] == 1:
                    pred = "It's a DOG"
                else:
                    pred = "It's a CAT"
                st.success(pred)
            except:
                st.error("Invalid Video Type For This Model")
