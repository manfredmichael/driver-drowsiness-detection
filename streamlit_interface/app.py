import base64
import json
import os
import re
import time
import uuid
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from utils import inference

def from_picture():
    st.markdown("### STEP 1: Upload your virtual background image")    
    virtual_background_file = st.file_uploader('Upload virtual background image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if virtual_background_file: 
        virtual_background = Image.open(virtual_background_file).convert('RGB')
        st.image(virtual_background)

    st.markdown("### STEP 2: Upload your empty background image (tip: hide under the table!)")    
    background_file = st.file_uploader('Upload background image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if background_file: 
        background = Image.open(background_file).convert('RGB')
        st.image(background)

    st.markdown("### STEP 3: Upload YOU!")   
    webcam_file = st.file_uploader('Upload webcam image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if webcam_file: 
        webcam = Image.open(webcam_file).convert('RGB')
        st.image(webcam)

    if webcam_file and background_file and virtual_background_file:
        st.markdown("---")
        st.markdown("# Virtual Background Result")

        steps, result = inference(webcam, background, virtual_background)

        st.markdown(f"### The steps")
        for step_name, step_image in steps.items():
            st.markdown(f"##### {step_name}")
            st.image(step_image)

        st.markdown(f"### The result")
        st.image(result)

def from_camera():
    st.markdown("### STEP 1: Upload your virtual background image")    
    virtual_background_file = st.file_uploader('Upload virtual background image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if virtual_background_file: 
        virtual_background = Image.open(virtual_background_file).convert('RGB')
        st.image(virtual_background)

    st.markdown("### STEP 2: Take your empty background image (tip: hide under the table!)")    
    background_file = st.camera_input("Take a picture of the background")
    if background_file: 
        background = Image.open(background_file).convert('RGB')
        st.image(background)

    st.markdown("### STEP 3: Take a picture of YOU!")   
    webcam_file = st.camera_input("Take a picture of yourself") 
    if webcam_file: 
        webcam = Image.open(webcam_file).convert('RGB')
        st.image(webcam)

    if webcam_file and background_file and virtual_background_file:
        st.markdown("---")
        st.markdown("# Virtual Background Result")

        steps, result = inference(webcam, background, virtual_background)

        st.markdown(f"### The steps")
        for step_name, step_image in steps.items():
            st.markdown(f"##### {step_name}")
            st.image(step_image)

        st.markdown(f"### The result")
        st.image(result)

def main():
    page_names_to_funcs = {
        "From Uploaded Picture": from_picture,
        "From Camera": from_camera,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Virtual Background with Image Processing", page_icon=":pencil2:"
    )
    st.title("Virtual Background with Image Processing")
    # st.sidebar.subheader("Configuration")
    main()
