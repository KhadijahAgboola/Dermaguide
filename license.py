import cv2
import easyocr
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt  


# Function to preprocess the image for better OCR performance
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return enhanced_image


# Function to extract license plate text from the image
def extract_license_number(image_path, reader):
    try:
        preprocessed_image = preprocess_image(image_path)
        results = reader.readtext(preprocessed_image, detail=0, paragraph=False)
        return results
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return []


# Function to save extracted license plate data to an Excel file
def save_to_excel(data, output_path="license_plate_data.xlsx"):
    try:
        # If the file exists, append data. Otherwise, create a new DataFrame.
        if os.path.isfile(output_path):
            df = pd.read_excel(output_path)
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        else:
            df = pd.DataFrame(data)
        
        # Save to Excel
        df.to_excel(output_path, index=False)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to Excel: {e}")

# Function to display annotated image
def display_annotated_image(image_path, reader):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Detect text with bounding boxes
        results = reader.readtext(image, detail=1, paragraph=False)
        for (bbox, text, confidence) in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Draw rectangle around detected text
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # Overlay the text
            cv2.putText(
                image, text, (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        # Convert BGR (OpenCV format) to RGB (Matplotlib format) for displaying
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Annotated Image with Detected Text")
        plt.show()

    except Exception as e:
        print(f"Error displaying annotated image: {e}")


# Function to process a single image
def process_license_plate(image_path, reader, output_path="license_plate_data.xlsx"):
    extracted_text = extract_license_number(image_path, reader)
    if extracted_text:
        # Assume the license plate is the first meaningful text detected
        license_number = extracted_text[0]
        date_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare data to save
        data = [{"License Number": license_number, "Date Processed": date_processed}]
        save_to_excel(data, output_path)

        print(f"License plate extracted and saved for {image_path}: {license_number}")

        # Display the annotated image
        display_annotated_image(image_path, reader)

    else:
        print(f"No text found in {image_path}")


# Main function
def main(image_directory="images/", output_excel="license_plate_data.xlsx"):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(["en"], gpu=False)

    # Create directory for input files if it doesn't exist
    os.makedirs(image_directory, exist_ok=True)

    for filename in sorted(os.listdir(image_directory)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_directory, filename)
            print(f"Processing {filename}...")
            process_license_plate(image_path, reader, output_excel)


# Run the program
if __name__ == "__main__":
    main()
