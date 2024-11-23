import cv2
import easyocr
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


# Function to save data to a CSV file
def save_to_csv(data, csv_path='extracted_text.csv'):
    headers = ['Text', 'Score', 'BoundingBox']
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write headers if the file is new
        writer.writerows(data)

# Function to search for text in the CSV file
def search_text(query, csv_path='extracted_text.csv'):
    results = []
    if not os.path.isfile(csv_path):
        print(f"No data found. {csv_path} does not exist.")
        return results

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if query.lower() in row['Text'].lower():
                results.append(row)
    return results

# Code to read image
image = cv2.imread('images (1).jfif')


# Create an EasyOCR text detector instance to process the image and extract text data
reader = easyocr.Reader(['en'], gpu=False)

#Perform Optical Character Recognition (OCR) on the image to detect text, including its content, 
#bounding box coordinates, and confidence score
text_ = reader.readtext(image, min_size=3, text_threshold=0.5, low_text=0.3)
threshold = 0.3


# Organize data for storage
extracted_data = []
for t_, t in enumerate(text_):
    bbox, text, score = t
    if score > threshold:
        cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
        cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
        extracted_data.append([text, score, bbox])  # Store in list format for saving

# Save data to CSV
save_to_csv(extracted_data)

# Display annotated image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Search functionality
query = input("Enter text to search: ")
search_results = search_text(query)

if search_results:
    print("Search Results:")
    for result in search_results:
        print(f"Text: {result['Text']}, Score: {result['Score']}, BoundingBox: {result['BoundingBox']}")
else:
    print("No matches found.")
