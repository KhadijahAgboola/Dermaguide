import streamlit as st
import cv2
import easyocr
import pandas as pd
import numpy as np
from PIL import Image
from docx import Document
from pdf2image import convert_from_bytes

# Function to display the versions of the installed libraries
def check_versions():
    st.write("### Installed Library Versions")
    st.write(f"**Streamlit:** {st.__version__}")
    st.write(f"**OpenCV:** {cv2.__version__}")
    st.write(f"**EasyOCR:** {easyocr.__version__}")
    st.write(f"**Pandas:** {pd.__version__}")
    st.write(f"**Numpy:** {np.__version__}")
    st.write(f"**Pillow (PIL):** {Image.__version__ if hasattr(Image, '__version__') else 'N/A'}")
    st.write(f"**python-docx:** {Document.__version__ if hasattr(Document, '__version__') else 'N/A'}")
    st.write(f"**pdf2image:** {convert_from_bytes.__module__}")

# Call the function to check versions
check_versions()
