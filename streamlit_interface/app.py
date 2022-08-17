import base64
import json
import os
import re
import time
import uuid
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from utils import inference

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}


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

        st.image(inference(webcam, background, virtual_background))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Virtual Background with Image Processing", page_icon=":pencil2:"
    )
    st.title("Virtual Background with Image Processing")
    # st.sidebar.subheader("Configuration")
    main()