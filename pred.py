from model import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras.preprocessing


def app():
	
    	with st.spinner('Model is being loaded..'):
        	model = load_model()
    		st.subheader("""**NOTE:** This app work best only when you uplode image of a Cat or Dog.""")
    		file = st.file_uploader("Please upload a Video of ASL Sign which You want to Translate", type=["MP4", "MOV", "MKV", "WMV", "MPEG-2"])
    		st.set_option('deprecation.showfileUploaderEncoding', False)

    		if file is None:
        		st.write("""Please upload an Video file""")
    		else:
        		if st.button("Predict"):
				try:
					video_file = open(file, 'rb')
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
