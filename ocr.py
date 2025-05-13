import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import pytesseract  # Import pytesseract for OCR text detection
import cv2
import numpy as np

# Initialize models for text classification (BERT) and image classification (ResNet)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load pretrained ResNet50 model
image_model = models.resnet50(pretrained=True)
image_model.eval()  # Set the model to evaluation mode

# Preprocessing for the image model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize webcam for real-time video capture
cap = cv2.VideoCapture(0)  # '0' is the default webcam

# Function to classify input (text or image)
def classify_input(input_data):
    """ Classify whether the input is text or an image """
    if isinstance(input_data, str):  # Text classification
        return "Text"  # Classify as "Text"
    
    elif isinstance(input_data, np.ndarray):  # Image classification
        return "Image"  # Classify as "Image"

def detect_text_in_image(image_frame):
    """ Use Tesseract OCR to detect text in the image frame """
    text = pytesseract.image_to_string(image_frame)
    return text.strip()  # Return the detected text, if any

def get_input_from_camera():
    """ Capture frame from the webcam and classify it """
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Failed to grab frame from webcam")
        return None
    return frame

# Real-time classification loop
while True:
    frame = get_input_from_camera()  # Get input from the camera
    
    if frame is None:
        break
    
    # Detect text in the frame using OCR
    detected_text = detect_text_in_image(frame)
    
    if detected_text:  # If text is detected, classify as text
        result = classify_input(detected_text)  # Classify the detected text
    else:  # If no text is detected, classify as image
        result = classify_input(frame)  # Classify the frame as an image
    
    print(f"Classification Result: {result}")  # Print the classification result
    
    # Display the live webcam feed
    cv2.imshow("Real-Time Classification (Press 'q' to quit)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
