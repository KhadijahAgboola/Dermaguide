
import streamlit as st
import cv2
import easyocr
import pandas as pd
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image
from docx import Document

# CSS for background color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ADD8E6;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Problem Statement
st.title("Image Text Extraction and Management Program")
st.subheader("Problem Statement")
st.write("""
We need a developer to develop a tool that extracts text (OCR) from uploaded images, organizes and stores the extracted data, and provides a searchable management system. 

**Target Users**: General users and small businesses.
""")

import streamlit as st

st.sidebar.markdown(
    """
    <style>
        .developer-box {
            background-color: #E8BCB9; /* Light pinkish-grey */
            color: black;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image(
    "Headshot.jpg", 
    caption="Khadijat Agboola",
    width=200
)


st.sidebar.markdown(
    """
    <style>
        .developer-box {
            background-color: #E8BCB9; /* Light pinkish-grey */
            color: black;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    <div class="developer-box">
        <h3>About the Developer</h3>
        <p>
            I am <strong>Khadijat Agboola</strong>, a dedicated developer with expertise in data science, 
            machine learning, and artificial intelligence. 
        </p>
        <p>
            I created this program to help simplify text extraction from images and manage the data effectively. 
            This tool extracts text (especially license plate numbers), organizes it into structured formats like Excel, 
            and saves the data locally.
        </p>
        <p>
            This solution is still under construction, and improvements are ongoing to ensure higher accuracy 
            and additional features.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Choose Your Task
st.subheader("Select an action you'd like to perform with the app")
st.write("""
This tool allows you to:
1. Extract text from images (e.g car license plate numbers from images).
2. Extract text from scanned documents.
""")

# User Choice
task_choice = st.radio(
    "What would you like to do?",
    ("Scan Car License Number or any other image", "Upload Scanned Document")
)

if task_choice == "Scan Car License Number or any other image":
    st.write("You can find car license number images from this [link](https://www.google.com/search?q=car+license+number&sca_esv=c640f05120339334&udm=2&biw=1280&bih=593&sxsrf=ADLYWIKRS_2-X9ucKHC0NIp743salsE3KQ%3A1732294667011&ei=C7hAZ7gvxZSFsg_-ktDBAQ&ved=0ahUKEwj44JfCtPCJAxVFSkEAHX4JNBgQ4dUDCBA&uact=5&oq=car+license+number&gs_lp=EgNpbWciEmNhciBsaWNlbnNlIG51bWJlcioCCAAyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBBAAGB4yBBAAGB4yBBAAGB4yBBAAGB5IAFAAWABwAHgAkAEAmAHWAaAB1gGqAQMyLTG4AQHIAQCYAgGgAtsBmAMAkgcDMi0xoAfHBQ&sclient=img).")
else:
    st.write("Upload a scanned document to extract text.")

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return enhanced_image

# Function to extract text
def extract_text(image, reader):
    preprocessed_image = preprocess_image(image)
    results = reader.readtext(preprocessed_image, detail=0, paragraph=False)
    return results

# Function to annotate image with bounding boxes
def annotate_image(image, reader):
    results = reader.readtext(image, detail=1, paragraph=False)
    for (bbox, text, confidence) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(
            image, text, (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )
    return image

# Save data to Excel
def save_license_data(data):
    output = BytesIO()
    df = pd.DataFrame(data)
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output

# Save text to Word
def save_to_word(text):
    output = BytesIO()
    doc = Document()
    for line in text:
        doc.add_paragraph(line)
    doc.save(output)
    output.seek(0)
    return output

# Process the chosen task
if task_choice == "Scan Car License Number":
    uploaded_file = st.file_uploader(
        "Upload an image file of a car license plate (JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(["en"], gpu=False)

        # Extract text
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text(image, reader)
        
        if extracted_text:
            license_number = extracted_text[0]
            date_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = [license_number]
            
            st.write("**Extracted License Plate Text:**")
            st.write(license_number)

            # Annotate the image
            annotated_image = annotate_image(image.copy(), reader)
            st.image(annotated_image, caption="Annotated Image with Detected Text", use_column_width=True)

            if st.button("Save Data as Excel"):
                output = save_license_data(data)
                st.download_button(
                    label="Download Excel File",
                    data=output,
                    file_name="license_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.write("No text detected in the uploaded image.")

elif task_choice == "Upload Scanned Document":
    uploaded_file = st.file_uploader(
        "Upload a scanned document (JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(["en"], gpu=False)

        # Extract text
        st.image(image, caption="Uploaded Document", use_column_width=True)
        extracted_text = extract_text(image, reader)
        
        if extracted_text:
            st.write("**Extracted Text:**")
            st.write("\n".join(extracted_text))

            if st.button("Save Data as Word"):
                output = save_to_word(extracted_text)
                st.download_button(
                    label="Download Word File",
                    data=output,
                    file_name="extracted_text.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
        else:
            st.write("No text detected in the uploaded document.")

