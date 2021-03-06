from model import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras.preprocessing
import math
import random
import cv2
import tempfile

def predict_single_actionlstm(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
    CLASSES_LIST = [ "who", "what", "wait", "help", "drink"]
    #CLASSES_LIST = [ "BEAUTIFUL", "BOOK", "COMPUTER", "DRINK", "HAPPY", "HEALTHY", "HELP"]
    # Specify the height and width to which each video frame will be resized in our dataset.
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
    model = load_model()
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Declare a list to store video frames we will extract.
    frames_list = []
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        # Read a frame.
        success, frame = video_reader.read() 
        # Check if frame is not read properly then break the loop.
        if not success:
            break
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities =  model.predict(np.expand_dims(frames_list, axis = 0))[0]   
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    # Display the predicted action along with the prediction confidence.
    st.write(f'Action Predicted : {predicted_class_name}')
    st.write(f'Confidence : {predicted_labels_probabilities[predicted_label]}')
    # Release the VideoCapture object. 
    video_reader.release()
        
       
def app():
    with st.spinner('Model is being loaded..'):
        model = load_model()
    st.subheader("""**NOTE:** This app work best only when you uplode Video of ASL.""")
    f = st.file_uploader("Please upload a Video of ASL Sign which You want to Translate")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if f is None:
        st.write("""Please upload an Video file""")
    else:
        if st.button("Predict"):
            try:
                #CONVO+LSTM MODEL
                # Specify the height and width to which each video frame will be resized in our dataset.
                SEQUENCE_LENGTH = 25
                #st.write('working')
                # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
                CLASSES_LIST = [ "who", "what", "wait", "help", "drink"]
                #CLASSES_LIST = [ "BEAUTIFUL", "BOOK", "COMPUTER", "DRINK", "HAPPY", "HEALTHY", "HELP"]
                #read video & frames from upload
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(f.read())

                video_file = open(tfile.name, 'rb')            
                video_bytes = video_file.read()
                #st.write("file download")
                # Perform Single Prediction on the Test Video.
                predict_single_actionlstm(tfile.name, SEQUENCE_LENGTH)
                st.success("Successfully Predicted")
                st.video(video_bytes)
            except:
                st.error("Invalid Video Type For This Model Or The Uploaded video does not belong to ASL Category")
                st.video(video_bytes)
