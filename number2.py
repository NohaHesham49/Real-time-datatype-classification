from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import pytesseract  # Import pytesseract for OCR text detection
import cv2
import numpy as np
import re

# Initialize models for text classification (BERT) and image classification (ResNet)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load pretrained ResNet50 model
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Preprocessing for the image model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize webcam for real-time video capture
cap = cv2.VideoCapture(0)

# Function to classify input (text, numeric, or image)
def classify_input(input_data):
    """ Classify whether the input is text, numeric, or an image """
    if isinstance(input_data, str):
        # Check if the input contains primarily numeric content
        numeric_pattern = r'^\d+(\.\d+)?$'  # Matches integers or decimals
        if re.match(numeric_pattern, input_data):
            return "Numeric"
        return "Text"
    elif isinstance(input_data, np.ndarray):
        return "Image"

def detect_text_in_image(image_frame):
    """ Use Tesseract OCR to detect text in the image frame """
    text = pytesseract.image_to_string(image_frame)
    return text.strip()

def detect_numeric_in_text(detected_text):
    """ Enhanced function to check if the detected text contains numeric values """
    # Regex to match integers, decimals, and scientific notation
    numeric_pattern = r'\b(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b'
    numbers = re.findall(numeric_pattern, detected_text)
    return numbers

def get_input_from_camera():
    """ Capture frame from the webcam and classify it """
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        return None
    return frame

# Real-time classification loop
detected_numbers = []  # List to store detected numbers
while True:
    frame = get_input_from_camera()
    
    if frame is None:
        break
    
    # Detect text in the frame using OCR
    detected_text = detect_text_in_image(frame)
    
    if detected_text:
        # Check if detected text contains numbers
        numbers_in_text = detect_numeric_in_text(detected_text)
        
        if numbers_in_text:
            detected_numbers = numbers_in_text  # Store numbers in the list to display them continuously
            result = "Numeric"
        else:
            result = "Text"
            print(f"Detected Text: {detected_text}")
    else:
        result = classify_input(frame)  # Classify as Image
    
    # Display the detected numbers continuously
    if detected_numbers:
        print(f"Detected Numbers: {', '.join(detected_numbers)}")
    
    print(f"Classification Result: {result}")
    
    # Display the live webcam feed
    cv2.imshow("Real-Time Classification (Press 'q' to quit)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
