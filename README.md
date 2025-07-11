
# Unmasking the Truth: AI-Powered Deepfake Image Detection for Public Safety

This project aims to combat the rising threat of deepfakes by leveraging state-of-the-art Vision Transformer (ViT) and CLIP models to detect fake, AI-generated, or manipulated images. With applications in media, law enforcement, branding, and public safety, our solution provides a web-based tool for deepfake image analysis.

We curated a diverse dataset covering multiple deepfake manipulation types—face swaps, attribute editing, and face synthesis—and evaluated seven pretrained models from Hugging Face. The system generates detailed predictions including class-wise probabilities and confidence scores, and identifies the best-performing model for each manipulation type.

Our user-friendly Streamlit application allows batch uploads, model comparisons, and result summaries to help users verify visual content quickly and accurately.

This project demonstrates a scalable approach to deepfake detection, offering flexibility across different use cases and paving the way for future extensions into video and audio deepfake detection.


## Features

- Upload and analyze multiple images at once
- Choose from 7 pretrained models (ViT, CLIP, etc.)
- Compare predictions across all models
- Detect real, fake, or AI-generated images
- View detailed confidence scores and per-class probabilities
- Light/Dark theme toggle for UI comfort
- Model-wise performance summary based on manipulation type
- User-friendly Streamlit web interface — no coding required

## Installation 
Follow these steps to set up and run the project locally:

#### 1.Clone the repository
git clone https://github.com/VRUSH1909/DeepFake_Image_Detection.git
cd DeepFake_Image_Detection

#### 2.Install dependencies
pip install -r requirements.txt

#### 3.Run the app
streamlit run app.py



## Datasets

To build a comprehensive deepfake detection system, we curated a diverse dataset from various trusted open-source platforms. Our goal was to cover multiple types of image manipulation techniques for broader evaluation.

- We collected real and fake images from **Kaggle**, **Hugging Face Datasets**, and **public deepfake repositories**.
- The fake images included:
  - **Face Swapping** (one person’s face replaced with another)
  - **Attribute Editing** (changes in facial features like age, gender, expressions)
  - **Face Synthesis** (AI-generated faces using GANs and other models)
  - **AI-generated animals, scenery, and art**
- Real images were sourced to ensure model comparison could distinguish between genuine and synthetic content.

The dataset was manually inspected and organized into labeled categories to help evaluate how well different models perform on various manipulation types. This structured approach allowed for targeted testing and better insights into model strengths and weaknesses.
## Models Used — A Table Listing the Models and Their Tasks

The project uses a variety of pretrained transformer-based models from Hugging Face for deepfake image classification. Each model is specialized to detect different types of manipulations:

| Model Name                                    | Model Type         | Prediction Classes            | Task / Focus Area                         |
|----------------------------------------------|--------------------|-------------------------------|-------------------------------------------|
| `ashish-001/deepfake-detection-using-ViT`    | ViT (Vision Transformer) | Real / Fake               | Best for edited faces (attribute changes) |
| `prithivMLmods/Deep-Fake-Detector-v2-Model`  | ViT                | Real / Fake                   | Performs well on face swap detection      |
| `dima806/ai_vs_real_image_detection`         | ViT                | Real / Fake                   | Identifies AI-generated vs real images    |
| `dima806/deepfake_vs_real_image_detection`   | ViT                | Real / Fake                   | Focused on detecting deepfakes            |
| `openai/clip-vit-base-patch32`               | CLIP (ViT + Text)  | Real / Fake (via prompt)      | Uses text-image comparison                |
| `prithivMLmods/AI-vs-Deepfake-vs-Real`       | ViT                | Real / Fake / AI-generated    | Handles multi-class classification        |
| `prithivMLmods/Deep-Fake-Detector-Model`     | ViT                | Real / Fake                   | General-purpose deepfake detection        |

Each model is integrated dynamically into the Streamlit application and tested on different image categories for comparison.

---

## Results and Evaluation — Model Performance Across Categories

The following table summarizes the accuracy (confidence scores) of each model across different deepfake categories such as face swapping, attribute editing, AI art, animals, and scenery. `R` = Real, `F` = Fake, `D` = Deepfake, `A` = AI-generated.

| Categories                    | Deep-Fake-Detector-Model | ashish-001 | AI-vs-Deepfake-vs-Real | CLIP | ai-vs-real-image-detection | deepfake-vs-real-image-detection | Deep-Fake-Detector-v2 |
|------------------------------|--------------------------|------------|-------------------------|------|-----------------------------|----------------------------------|------------------------|
| Face Swapping (Fake)         | R – 71.97%, F – 28.03%   | R – 94.26%, F – 5.74%   | R – 58.65%, D – 26.18%, A – 58.65% | R – 70.49%, F – 29.51% | R – 25.03%, F – 74.97%     | R – 94.73%, F – 5.27%           | R – 24.43%, D – 75.57%       |
| Attribute Editing (Real)     | R – 71.03%, F – 28.97%   | R – 80.61%, F – 20.39%  | R – 42.72%, D – 36.63%, A – 20.66% | R – 77.07%, F – 22.93% | R – 30.94%, F – 69.06%     | R – 72.34%, F – 27.66%          | R – 40.67%, F – 59.33%       |
| Attribute Editing (Fake)     | R – 79.56%, F – 20.44%   | R – 39.88%, F – 75.12%  | R – 32.60%, D – 47.05%, A – 20.35% | R – 65.17%, F – 34.83% | R – 20.29%, F – 79.71%     | R – 39.09%, F – 60.91%          | R – 45.48%, F – 54.52%       |
| Face Synthesis (Fake)        | R – 77.26%, F – 22.74%   | R – 17.01%, F – 82.99%  | R – 43.17%, D – 38.39%, A – 18.45% | R – 63.79%, F – 36.21% | R – 33.85%, F – 66.15%     | R – 43.13%, F – 56.87%          | R – 41.62%, F – 58.38%       |
| Animals (Real)               | R – 56.59%, F – 43.41%   | R – 68.10%, F – 31.90%  | R – 31.54%, D – 34.79%, A – 33.67% | R – 48.23%, F – 51.77% | R – 29.59%, F – 70.41%     | R – 69.13%, F – 30.87%          | R – 42.76%, F – 57.24%       |
| Animals (Fake)               | R – 53.28%, F – 46.72%   | R – 62.78%, F – 37.52%  | R – 24.20%, D – 32.09%, A – 43.71% | R – 41.95%, F – 58.05% | R – 25.41%, F – 74.59%     | R – 79.77%, F – 20.23%          | R – 34.15%, F – 65.85%       |
| Scenery (Real)               | R – 50.90%, F – 49.10%   | R – 40.46%, F – 59.54%  | R – 25.75%, D – 32.39%, A – 41.86% | R – 27.60%, F – 72.40% | R – 8.16%, F – 91.84%      | R – 66.98%, F – 33.02%          | R – 40.86%, F – 59.14%       |
| Scenery (Fake)               | R – 49.30%, F – 50.70%   | R – 59.70%, F – 40.30%  | R – 30.18%, D – 30.89%, A – 38.94% | R – 58.26%, F – 41.74% | R – 14.35%, F – 85.65%     | R – 81.10%, F – 18.90%          | R – 38.21%, F – 61.79%       |
| Art (Human)                  | R – 42.68%, F – 57.32%   | R – 44.03%, F – 55.97%  | R – 27.57%, D – 33.40%, A – 39.03% | R – 3.69%, F – 96.31%  | R – 17.95%, F – 82.05%     | R – 82.54%, F – 17.46%          | R – 36.96%, F – 63.04%       |
| Art (AI-generated)           | R – 46.58%, F – 53.42%   | R – 60.40%, F – 39.60%  | R – 23.24%, D – 32.52%, A – 44.24% | R – 22.48%, F – 77.52% | R – 9.27%, F – 90.73%      | R – 86.67%, F – 13.33%          | R – 34.07%, F – 65.93%       |

These results helped us understand that **no single model is perfect across all categories**. Instead, we recommend using **different models depending on the image type**, or implementing **model ensembling** in future development.

## Usage — How to Run the App and Use It

## 🚀 Usage

Once the app is running, you can interact with the Deepfake Detection system through a clean and intuitive web interface. Here's how to use it:

1. **Upload Images**  
   - Click the upload area to select one or multiple images (`.jpg`, `.jpeg`, or `.png` format).  
   - You can upload real, fake, or AI-generated images for analysis.

2. **Select a Model**  
   - Choose one specific pretrained model from the dropdown for individual predictions,  
   **or**  
   - Enable the **"Predict using all models"** option to compare results across all supported models.

3. **Get Predictions**  
   - Instantly view predictions for each image, including:
     - **Predicted label** (Real / Fake / AI-generated)
     - **Confidence score**
     - **Per-class probability breakdown**

4. **Review Summary**  
   - After batch analysis, the app provides:
     - **Majority prediction across all images**
     - **Average confidence score**
     - **Model-wise class probability summary**

This tool is designed to be used by journalists, forensic analysts, researchers, and the general public—no coding knowledge is needed.


## Future Work— Planned Improvements

While the current application effectively detects deepfakes in images using multiple AI models, there are several directions for expanding and enhancing the system:

### 🔹 1. Video Deepfake Detection
We plan to extend the solution to detect deepfakes in videos by analyzing individual frames. Since videos are just sequences of images, applying the same detection logic frame-by-frame will allow us to flag manipulated video content. Additional techniques like temporal consistency checks can further improve accuracy.

### 🔹 2. Audio Deepfake Analysis
Another major goal is to integrate audio deepfake detection. This would involve identifying synthetically generated voices, altered speech, or impersonation using waveform and spectrogram analysis, possibly leveraging pretrained models such as Wav2Vec or Whisper.

### 🔹 3. Report Generation
We aim to add the ability to generate automated, downloadable reports (in PDF format) summarizing:
- Prediction results for uploaded images/videos
- Confidence scores
- Class-wise probability distributions
- Model-wise comparison
This feature would be useful for law enforcement, media teams, or researchers who require audit-ready documentation.

### 🔹 4. Cloud Deployment
Plans are in place to deploy the application on platforms like **Streamlit Cloud**, **Hugging Face Spaces**, or even integrate it into a **browser extension** for real-time content validation.

### 🔹 5. Model Ensembling
To improve accuracy and reliability, we also intend to implement **model ensembling**—where multiple models contribute to a single prediction, using weighted averaging or voting mechanisms based on the image category.

These future improvements will make the system more comprehensive, scalable, and applicable to real-world scenarios involving misinformation, media verification, and public safety.

## License and Credits


### 🔐 License
This project is intended for educational and research purposes only.  
It uses publicly available pretrained models and open-source datasets for experimentation and evaluation.  
Please ensure compliance with individual model licenses if deploying commercially.

---

### 👩‍💻 Contributors
- **Vrushali Dhage** – Application development,performance analysis, evaluation , documentation.
- **Samiksha Jagtap** – Model research, dataset preparation, evaluation.

---

### 🧑‍🏫 Project Mentors / Guides
- **Kashish Arora** – Technical mentorship, model validation support
- **Rohita Munde** – Project guidance, architecture suggestions


---

We thank all open-source contributors, Hugging Face model creators, and the research community whose tools made this project possible.
