import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# CPU forcé pour Streamlit Cloud
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False, weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load("densenetaidetection.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Tes transforms EXACTES
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['real', 'fake']  # ← ADAPTÉ après vérif classes
model = load_model()

st.set_page_config(page_title="Fake Detector", page_icon="🖼️")
st.title("🖼️ Fake/Real Detector - DenseNet121")
st.info(f"**0={CLASS_NAMES[0].upper()} | 1={CLASS_NAMES[1].upper()}**")

uploaded_file = st.file_uploader("📁 Image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Prêt pour analyse", use_column_width=True)
    
    if st.button("🔍 **CLASSER**", type="primary"):
        with st.spinner("DenseNet121 en cours..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                pred_class = int(predicted)
            
            # UI améliorée
            col1, col2 = st.columns(2)
            with col1:
                class_name = CLASS_NAMES[pred_class]
                if pred_class == 1:  # Fake ?
                    st.error(f"🚨 **FAKE**")
                else:
                    st.success(f"✅ **REAL**")
                st.metric("Confiance", f"{confidence:.1%}")
            
            with col2:
                st.metric("Real", f"{probabilities[0]:.1%}")
                st.metric("Fake", f"{probabilities[1]:.1%}")
            
            st.balloons()
