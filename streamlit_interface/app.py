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

    webcam_file = st.file_uploader('Upload webcam image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if webcam_file: 
        webcam = Image.open(webcam_file).convert('RGB')
        st.image(webcam)
    background_file = st.file_uploader('Upload background image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if background_file: 
        background = Image.open(background_file).convert('RGB')
        st.image(background)
    virtual_background_file = st.file_uploader('Upload virtual background image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if virtual_background_file: 
        virtual_background = Image.open(virtual_background_file).convert('RGB')
        st.image(virtual_background)

    if webcam_file and background_file and virtual_background_file: 
        # result = inference(image)
        # def serve_pil_image(pil_img):
        #     img_io = BytesIO()
        #     pil_img.save(img_io, 'JPEG', quality=70)
        #     img_io.seek(0)
        #     return img_io
        # r = Image.open(BytesIO(serve_pil_image(image).read()))
        # st.write(type(r))
        # st.write(type(image))

        # st.write(result)
        st.image(inference(webcam, background, virtual_background))

    # color_annotation_app()

    # with st.sidebar:
    #     # st.markdown("---")
    #     st.markdown(
    #         '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/manfredmichael">Manfred Michael</a></h6>',
    #         unsafe_allow_html=True,
    #     )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Driver Drowsiness Detector", page_icon=":pencil2:"
    )
    st.title("Driver Drowsiness Detector")
    # st.sidebar.subheader("Configuration")
    main()