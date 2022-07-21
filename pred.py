from model import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras.preprocessing
import math
import random

def predict_single_actionlstm(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''  
    CLASSES_LIST = [ "who", "what", "wait", "help", "drink"]
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities =  model.predict(video_file_path)[0]
    #st.write(predicted_labels_probabilities)
    
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    #st.write(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    st.wrtie(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
       
def app():
    with st.spinner('Model is being loaded..'):
        model = load_model()
    st.subheader("""**NOTE:** This app work best only when you uplode Video of ASL.""")
    file = st.file_uploader("Please upload a Video of ASL Sign which You want to Translate")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if file is None:
        st.write("""Please upload an Video file""")
    else:
        if st.button("Predict"):
            try:
                #CONVO+LSTM MODEL
                
                # Specify the height and width to which each video frame will be resized in our dataset.
                IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
                SEQUENCE_LENGTH = 25
                
                # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
                CLASSES_LIST = [ "who", "what", "wait", "help", "drink"]
                
                #read video & frames from upload
                video_file = open('Copy of Copy of 62113.mp4', 'rb')            
                video_bytes = video_file.read()
                st.video(video_file)
                # Perform Single Prediction on the Test Video.
                predict_single_actionlstm(video_bytes, SEQUENCE_LENGTH)
                st.video(video_bytes)
                #vid = Video.open(file)            
                #st.video(vid, use_column_width=True)
                #test_vid = tf.image.resize(pic, [64, 64])
                #img = keras.preprocessing.image.img_to_array(test_img)
                #img = np.expand_dims(img, axis=0)/*
                st.success("Successfull")
            except:
                #st.video(video_file)
                st.error("Invalid Video Type For This Model")
