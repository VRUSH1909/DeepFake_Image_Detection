import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import base64

# ✅ FIRST Streamlit command
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# Encode and set background
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

encoded_bg = get_base64_image("background.jpg")  # Use the correct file name

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Model and processor type mapping
MODEL_MAPPING = {
    "prithivMLmods/Deep-Fake-Detector-Model": ("ViTImageProcessor", ViTForImageClassification),
    "ashish-001/deepfake-detection-using-ViT": ("AutoImageProcessor", AutoModelForImageClassification),
    "prithivMLmods/AI-vs-Deepfake-vs-Real": ("ViTImageProcessor", ViTForImageClassification),
    "openai/clip-vit-base-patch32": ("CLIPProcessor", CLIPModel),
    "dima806/ai_vs_real_image_detection": ("ViTImageProcessor", ViTForImageClassification),
    "dima806/deepfake_vs_real_image_detection": ("ViTImageProcessor", ViTForImageClassification),
    "prithivMLmods/Deep-Fake-Detector-v2-Model": ("ViTImageProcessor", ViTForImageClassification),
}

# Optional: explain model outputs
MODEL_LABEL_INFO = {
    "prithivMLmods/Deep-Fake-Detector-Model": "Predicts: Real or Fake",
    "ashish-001/deepfake-detection-using-ViT": "Predicts: Real or Fake",
    "prithivMLmods/AI-vs-Deepfake-vs-Real": "Predicts: Real, Deepfake, or Artificial",
    "openai/clip-vit-base-patch32": "Predicts: Real or Fake (via zero-shot image + text matching)",
    "dima806/ai_vs_real_image_detection": "Predicts: Real or Fake",
    "dima806/deepfake_vs_real_image_detection": "Predicts: Real or Fake",
    "prithivMLmods/Deep-Fake-Detector-v2-Model": "Predicts: Real or Fake",
}

@st.cache_resource
def load_model(model_name):
    processor_type, model_class = MODEL_MAPPING[model_name]
    if processor_type == "CLIPProcessor":
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    elif processor_type == "AutoImageProcessor":
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
    else:  # ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
    return processor, model

def predict_image(model_name, image):
    processor, model = load_model(model_name)
    image = image.convert("RGB")

    if model_name == "openai/clip-vit-base-patch32":
        labels = ["a real human face photo", "an AI-generated deepfake image"]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()
        predicted_label = "Real" if real_prob > fake_prob else "Fake"
        confidence = max(real_prob, fake_prob)
        class_probs = {"Real": real_prob, "Fake": fake_prob}
    else:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        predicted_class = probs.argmax().item()
        label_map = model.config.id2label

        if model_name == "ashish-001/deepfake-detection-using-ViT":
            predicted_label = "Real" if predicted_class == 1 else "Fake"
            class_probs = {
                "Fake": probs[0][0].item(),
                "Real": probs[0][1].item()
            }
        else:
            predicted_label = label_map[predicted_class]
            class_probs = {label_map[i]: probs[0][i].item() for i in range(len(label_map))}
        confidence = probs[0][predicted_class].item()

    return predicted_label, confidence, class_probs

# ---------------- STREAMLIT UI -------------------

st.markdown(
    """
    <h1 style='text-align: center; color: #b50480;'>🧠 Deepfake Image Detector</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an image and let AI detect if it's <b>Real, Fake, or AI-generated</b>.</p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("🧪 Model Selection")
    model_choice = st.selectbox("Choose a model:", list(MODEL_MAPPING.keys()))
    if model_choice:
        st.info(MODEL_LABEL_INFO.get(model_choice, "Model predicts Real, Fake or AI."))

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file and model_choice:
    image = Image.open(uploaded_file)
    st.divider()

    with st.spinner("🔍 Analyzing the image..."):
        label, confidence, class_probs = predict_image(model_choice, image)

    # Layout: Columns for image and result
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown(
            f"<h3 style='color:#1F77B4;'>🧾 Prediction: <span style='color:#E76F51'>{label}</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

        st.subheader("📊 Class Probabilities")
        for cls, prob in class_probs.items():
            bar_color = "green" if "real" in cls.lower() else ("red" if "fake" in cls.lower() else "orange")
            st.progress(prob, text=f"{cls}: {prob * 100:.2f}%")

    st.divider()

    # Show a final confidence overlay (optional)
    st.subheader("🖼️ Prediction Overlay")
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{label} ({confidence*100:.2f}%)",
                 fontsize=14,
                 color='green' if "real" in label.lower() else 'red')
    st.pyplot(fig)
