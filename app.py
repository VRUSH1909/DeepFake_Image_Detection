import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from collections import Counter, defaultdict

# ✅ Streamlit config
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# ---------------- THEME STYLING ---------------------
def apply_theme(theme_choice):
    if theme_choice == "Dark":
        background_color = "#1c1c1c"
        text_color = "#ffffff"
        upload_text_color = "#ffffff"
        analyzing_text_color = "#ffffff"
        file_button_color = "#444444"
        file_button_hover_color = "#666666"
        deploy_button_color = "#DA0037"
        deploy_button_hover_color = "#9c002c"
    else:
        background_color = "#f0f2f6"
        text_color = "#000000"
        upload_text_color = "#000000"
        analyzing_text_color = "#000000"
        file_button_color = "#ffffff"
        file_button_hover_color = "#f1f1f1"
        deploy_button_color = "#DA0037"
        deploy_button_hover_color = "#9c002c"

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: #333333;
        }}
        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# ------------------- MODEL SETUP ----------------------
MODEL_MAPPING = {
    "prithivMLmods/Deep-Fake-Detector-Model": ("AutoImageProcessor", AutoModelForImageClassification),
    "ashish-001/deepfake-detection-using-ViT": ("AutoImageProcessor", AutoModelForImageClassification),
    "prithivMLmods/AI-vs-Deepfake-vs-Real": ("ViTImageProcessor", ViTForImageClassification),
    "openai/clip-vit-base-patch32": ("CLIPProcessor", CLIPModel),
    "dima806/ai_vs_real_image_detection": ("ViTImageProcessor", ViTForImageClassification),
    "dima806/deepfake_vs_real_image_detection": ("ViTImageProcessor", ViTForImageClassification),
    "prithivMLmods/Deep-Fake-Detector-v2-Model": ("ViTImageProcessor", ViTForImageClassification),
}

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
    else:
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

# ------------------- UI -----------------------------

with st.sidebar:
    theme_choice = st.radio("Choose Theme", ["Light", "Dark"])
    apply_theme(theme_choice)

    st.header("🧪 Model Selection")
    model_choice = st.selectbox("Choose a model:", list(MODEL_MAPPING.keys()))
    if model_choice:
        st.info(MODEL_LABEL_INFO.get(model_choice, "Model predicts Real, Fake or AI."))
    predict_all = st.checkbox("🔄 Predict using all models")

st.markdown("""
<h1 style='text-align: center; color: #DA0037;'>🧠 Deepfake Image Detector</h1>
<p style='text-align: center; font-size: 18px;'>
Upload multiple images and let AI detect if they're <b>Real, Fake, or AI-generated</b>.</p>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📤 Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and (model_choice or predict_all):
    st.divider()

    if predict_all:
        st.subheader("🖼️ Uploaded Images")
        cols = st.columns(5)
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            with cols[i % 5]:
                st.image(image, caption=f"Image {i+1}", use_container_width=True, clamp=True)
        st.divider()

        for model_name in MODEL_MAPPING.keys():
            with st.container():
                label_counter = Counter()
                avg_confidences = []
                cumulative_probs = defaultdict(float)

                with st.spinner(f"🔍 Analyzing with {model_name}..."):
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        label, confidence, class_probs = predict_image(model_name, image)
                        label_counter[label] += 1
                        avg_confidences.append(confidence)
                        for cls, prob in class_probs.items():
                            cumulative_probs[cls] += prob

                total_images = len(uploaded_files)
                most_common_label = label_counter.most_common(1)[0][0]
                avg_conf = sum(avg_confidences) / total_images

                # ✅ Styled output section
                color_map = {
                    "Real": "green",
                    "Fake": "red",
                    "AI": "orange",
                    "Deepfake": "red",
                    "Artificial": "orange"
                }
                prediction_color = color_map.get(most_common_label, "blue")

                st.markdown(
                    f"""
                    <div style='font-size: 16px; line-height: 1.8;'>
                    🔍 <b>Model:</b> <code>{model_name}</code><br>
                    🧾 <b>Majority Prediction:</b> <span style='color:{prediction_color}; font-weight:bold;'>{most_common_label}</span><br>
                    📊 <b>Average Confidence:</b> <span style='color:green;'>{avg_conf * 100:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📈 Average Class Probabilities")
                for cls, total_prob in cumulative_probs.items():
                    avg_prob = total_prob / total_images
                    st.markdown(f"**{cls}: {avg_prob * 100:.2f}%**")
                    st.progress(avg_prob)

                st.divider()

    else:
        label_counter = Counter()
        avg_confidences = []
        cumulative_probs = defaultdict(float)
        st.subheader("🖼️ Uploaded Images with Predictions")
        cols = st.columns(5)

        with st.spinner("🔍 Analyzing uploaded images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                label, confidence, class_probs = predict_image(model_choice, image)
                label_counter[label] += 1
                avg_confidences.append(confidence)
                for cls, prob in class_probs.items():
                    cumulative_probs[cls] += prob

                with cols[i % 5]:
                    st.image(image, caption=f"{label}\n({confidence * 100:.1f}% confidence)", use_container_width=True, clamp=True)

        total_images = len(uploaded_files)
        most_common_label = label_counter.most_common(1)[0][0]
        avg_conf = sum(avg_confidences) / total_images

        st.subheader("📊 Overall Prediction Summary")
        st.markdown(f"### 🧾 Majority Prediction: `{most_common_label}`")
        st.markdown(f"**Average Confidence:** `{avg_conf * 100:.2f}%`")

        st.subheader("📈 Average Class Probabilities")
        for cls, total_prob in cumulative_probs.items():
            avg_prob = total_prob / total_images
            st.markdown(f"**{cls}: {avg_prob * 100:.2f}%**")
            st.progress(avg_prob)
