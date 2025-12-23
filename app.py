import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # 0=premier dossier, 1=deuxième dossier
    )
    model.load_state_dict(torch.load("densenetaidetection.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ADAPTATION AUTOMATIQUE - Remplace par tes vraies classes
CLASS_NAMES = ['real', 'fake']  # ← CHANGE ICI selon print(classes) !

model = load_model()

st.set_page_config(page_title="Fake Detector", page_icon="🖼️")
st.title("🖼️ Fake/Real Detector - DenseNet121")
st.info(f"**Classes :** 0={CLASS_NAMES[0].upper()} | 1={CLASS_NAMES[1].upper()}")

uploaded_file = st.file_uploader("📁 Image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Analyse en cours...", use_column_width=True)
    
    if st.button("🔍 **CLASSER**", type="primary"):
        with st.spinner("DenseNet121..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                pred_class = predicted.item()
            
            # Affichage selon prédiction 0/1
            col1, col2 = st.columns([2,1])
            with col1:
                if pred_class == 1:  # Deuxième classe = Fake ?
                    st.error(f"🚨 **FAKE** ({CLASS_NAMES[1]})")
                else:
                    st.success(f"✅ **REAL** ({CLASS_NAMES[0]})")
                st.metric("Confiance", f"{confidence.item():.1%}")
            
            with col2:
                st.metric("Real", f"{probabilities[0].item():.1%}")
                st.metric("Fake", f"{probabilities[1].item():.1%}")
