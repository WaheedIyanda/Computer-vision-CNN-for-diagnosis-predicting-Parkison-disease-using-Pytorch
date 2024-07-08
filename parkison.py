import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class ParkinsonCNN(nn.Module):
    def __init__(self):
        super(ParkinsonCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonCNN().to(device)
model.load_state_dict(torch.load('parkinson_spiral_model.pth', map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title('Parkinson\'s Disease Prediction from Spiral Drawing')

uploaded_file = st.file_uploader("Choose a spiral drawing image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Display result
    st.write('Prediction:')
    if prediction.item() > 0.5:
        st.write('The spiral drawing indicates a high likelihood of Parkinson\'s disease.')
    else:
        st.write('The spiral drawing indicates a low likelihood of Parkinson\'s disease.')
    
    st.write(f'Confidence: {prediction.item():.2f}')

st.write('Note: This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.')