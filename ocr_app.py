import streamlit as st
import cv2
import easyocr
import pandas as pd
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image
from docx import Document
from pdf2image import convert_from_bytes

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
st.title("Image Text Extraction and Management System")
st.subheader("Problem Statement")
st.write("""
We need a developer to develop a tool that extracts text (OCR) from uploaded images, organizes and stores the extracted data, and provides a searchable management system. 

**Target Users**: General users and small businesses.
""")

# Sidebar Information
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
    <div class="developer-box">
        <h3>About the Developer</h3>
        <p>
            I am <strong>Khadijat Agboola</strong>, a dedicated developer with expertise in data science, 
            machine learning, and artificial intelligence. 
        </p>
        <p>
            I created this program to help simplify text extraction from images and manage the data effectively. 
            This tool extracts text from images, organizes it into structured formats like Excel, word
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

# Specify the path to the poppler binary directory
poppler_path = r"C:\\Users\\Khadijat Agboola\\Desktop\\poppler-24.08.0\\Library\\bin"

# Choose Your Task
st.subheader("Select an action you'd like to perform with the app")
st.write("""
This tool allows you to:
1. Extract text from images (e.g., spreadsheet image) and save the result as an Excel file.
2. Extract text from scanned documents and save the result as a Word document.
""")

# User Choice
task_choice = st.radio(
    "What would you like to do?",
    ("Upload spreadsheet image or any other image", "Upload Scanned word Document")
)

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return enhanced_image

# Function to extract text with improved accuracy
def extract_text(image, reader):
    preprocessed_image = preprocess_image(image)
    results = reader.readtext(preprocessed_image, detail=1, paragraph=False)
    extracted_text = [text for _, text, _ in results]
    return extracted_text


# Save data to Excel
def save_license_data(data):
    output = BytesIO()
    df = pd.DataFrame({"Extracted Text": data})
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

# Function to extract text with spatial arrangement
def extract_text_with_columns(image, reader):
    """
    Extracts text from the image along with their bounding box positions and organizes it into columns.

    Parameters:
    - image: The image to process.
    - reader: EasyOCR Reader object.

    Returns:
    - A DataFrame with the extracted text organized into columns.
    """
    results = reader.readtext(image, detail=1, paragraph=False)
    extracted_data = []

    for (bbox, text, confidence) in results:
        top_left = bbox[0]
        bottom_right = bbox[2]
        x_center = (top_left[0] + bottom_right[0]) / 2  # Calculate x-center for column sorting
        y_center = (top_left[1] + bottom_right[1]) / 2  # Calculate y-center for row sorting
        extracted_data.append((text, x_center, y_center))

    # Sort by x-coordinate to identify columns
    extracted_data = sorted(extracted_data, key=lambda x: x[1])

    # Group text into columns based on x-coordinate proximity
    columns = []
    current_column = []
    column_threshold = 50  # Adjust as needed based on image dimensions

    for i, data in enumerate(extracted_data):
        if i == 0:
            current_column.append(data)
        else:
            # Check if the x-coordinate difference indicates a new column
            if abs(data[1] - extracted_data[i - 1][1]) > column_threshold:
                # Save the current column and start a new one
                columns.append(current_column)
                current_column = [data]
            else:
                current_column.append(data)

    # Append the last column
    if current_column:
        columns.append(current_column)

    # Sort text within each column by y-coordinate
    sorted_columns = [sorted(col, key=lambda x: x[2]) for col in columns]

    # Prepare DataFrame
    max_rows = max(len(col) for col in sorted_columns)
    data_dict = {f"Column {i + 1}": [col[j][0] if j < len(col) else "" for j in range(max_rows)] for i, col in enumerate(sorted_columns)}

    df = pd.DataFrame(data_dict)
    return df


# Updated logic for "Scan Car License Number or any other image"
if task_choice == "Upload spreadsheet image or any other image":
    uploaded_file = st.file_uploader(
        "Upload an image file of a spreadsheet (JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(["en"], gpu=False)

        # Extract text organized into columns
        st.image(image, caption="Uploaded Image", use_column_width=True)
        df = extract_text_with_columns(image, reader)

        if not df.empty:
            # Save the DataFrame to an Excel file
            output_file = "columnar_extracted_text.xlsx"
            df.to_excel(output_file, index=False)

            # Display the extracted text in columns
            st.write("**Extracted Text from Scanned Image (Organized into Columns):**")
            st.write(df)

            # Provide download link for the Excel file
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Extracted Text as Excel",
                    data=file,
                    file_name="columnar_extracted_text.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.write("No text detected in the uploaded image.")


# Process the chosen task
elif task_choice == "Upload Scanned word Document":
    uploaded_file = st.file_uploader(
        "Upload a scanned document (JPEG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"]
    )
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            # Convert PDF to images using the specified poppler path
            pdf_pages = convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
            st.write(f"PDF contains {len(pdf_pages)} page(s). Processing pages...")

            all_text = []
            for page_number, page in enumerate(pdf_pages, start=1):
                st.write(f"Processing page {page_number}...")
                image = np.array(page)
                st.image(image, caption=f"Uploaded Page {page_number}", use_column_width=True)
                reader = easyocr.Reader(["en"], gpu=False)
                extracted_text = extract_text(image, reader)
                all_text.extend(extracted_text)

        else:
            # Process image files
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            reader = easyocr.Reader(["en"], gpu=False)
            extracted_text = extract_text(image, reader)
            all_text = extracted_text

        if all_text:
            st.write("**Extracted Text:**")
            st.write("\n".join(all_text))

            # Save to Word file
            output = save_to_word(all_text)
            st.download_button(
                label="Download Word File",
                data=output,
                file_name="extracted_text.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            st.write("No text detected in the uploaded document.")

