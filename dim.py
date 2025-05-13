import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import pytesseract
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

# Define a lightweight CNN for digit recognition
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Reduced filters for efficiency
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Reduced size for efficiency
        self.fc2 = nn.Linear(64, 10)  # 10 classes for digits 0-9
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load or train the digit CNN model
def load_digit_cnn():
    model = DigitCNN()
    try:
        model.load_state_dict(torch.load('digit_cnn.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Pretrained digit CNN model not found. Please train the model on MNIST.")
    model.eval()
    return model

digit_cnn = load_digit_cnn()

# Preprocessing for digit CNN
digit_preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
])

# Initialize webcam for real-time video capture
cap = cv2.VideoCapture(0)

# Function to preprocess image for both OCR and digit detection
def preprocess_image(image):
    """ Preprocess image for OCR and digit detection """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return gray, thresh

# Function to detect text in image using Tesseract OCR
def detect_text_in_image(image_frame):
    """ Use Tesseract OCR to detect text in the image frame """
    _, preprocessed = preprocess_image(image_frame)
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text.strip()

# Function to detect individual digits using CNN
def detect_digits_in_image(image_frame):
    """ Detect individual digits in the image using the CNN model """
    gray, thresh = preprocess_image(image_frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_digits = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Filter small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if 0.5 < w / h < 2 and w > 10 and h > 10:  # Ensure roughly square region
            roi = gray[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_tensor = digit_preprocess(roi_pil).unsqueeze(0)
            with torch.no_grad():
                output = digit_cnn(roi_tensor)
                _, predicted = torch.max(output, 1)
                detected_digits.append(predicted.item())
    
    return detected_digits

# Function to classify input (text, numeric, or image)
def classify_input(input_data):
    """ Classify whether the input is text, numeric, or an image """
    if isinstance(input_data, str):
        return "Text"
    elif isinstance(input_data, np.ndarray):
        return "Image"
    elif isinstance(input_data, (float, int)):
        return "Numeric"

# Function to get input from camera
def get_input_from_camera():
    """ Capture frame from the webcam and classify it """
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        return None
    return frame

# Real-time classification loop
while True:
    frame = get_input_from_camera()
    
    if frame is None:
        break
    
    # Detect text in the frame using OCR
    detected_text = detect_text_in_image(frame)
    
    if detected_text:
        numbers = re.findall(r'\d+', detected_text)
        if numbers:
            result = f"Numeric (Detected numbers: {', '.join(numbers)})"
        else:
            result = f"Text (Detected: {detected_text})"
    else:
        digits = detect_digits_in_image(frame)
        if digits:
            result = f"Numeric (Detected numbers: {', '.join(map(str, digits))})"
        else:
            result = "Image"
    
    print(f"Classification Result: {result}")
    
    # Display the live webcam feed
    cv2.imshow("Real-Time Classification (Press 'q' to quit)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()