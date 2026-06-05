import os
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from groq import Groq

# =========================
# Load Model
# =========================

checkpoint = torch.load(
    "model.pth",
    map_location=torch.device("cpu")
)

class_names = checkpoint["class_names"]

model = models.mobilenet_v2(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# =========================
# Load Leaf Detector
# =========================

leaf_checkpoint = torch.load(
    "leaf_detector.pth",
    map_location=torch.device("cpu")
)

leaf_classes = leaf_checkpoint["class_names"]

leaf_model = models.mobilenet_v2(weights=None)

num_features = leaf_model.classifier[1].in_features

leaf_model.classifier[1] = nn.Linear(
    num_features,
    len(leaf_classes)
)

leaf_model.load_state_dict(
    leaf_checkpoint["model_state_dict"]
)

leaf_model.eval()


# =========================
# Groq AI Client
# =========================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# Image Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Disease Database
# =========================

disease_info = {

    "Pepper__bell___healthy": {
        "crop": "Bell Pepper",
        "severity": "🟢 Healthy",
        "health_score": 100,
        "recommendation": "Continue regular irrigation and crop monitoring.",
        "action": "Maintain current farming practices.",
        "about": "Bell pepper plant is healthy with no visible disease symptoms."
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "severity": "🟡 Moderate",
        "health_score": 70,
        "recommendation": "Apply fungicide and remove infected leaves.",
        "action": "Inspect plants every 2 days and remove affected foliage.",
        "about": "Early blight causes dark concentric spots on potato leaves."
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "severity": "🔴 Severe",
        "health_score": 40,
        "recommendation": "Apply preventive fungicide immediately.",
        "action": "Isolate infected plants and spray fungicide urgently.",
        "about": "Late blight spreads rapidly and can destroy potato crops."
    },

    "Potato___healthy": {
        "crop": "Potato",
        "severity": "🟢 Healthy",
        "health_score": 100,
        "recommendation": "Continue normal crop management.",
        "action": "Regular monitoring only.",
        "about": "Potato plant is healthy."
    },

    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "severity": "🟠 Moderate",
        "health_score": 60,
        "recommendation": "Use copper-based sprays and disease-free seeds.",
        "action": "Remove infected leaves and improve sanitation.",
        "about": "Bacterial Spot causes lesions on tomato leaves and fruits."
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "severity": "🟡 Moderate",
        "health_score": 65,
        "recommendation": "Apply fungicide and remove infected leaves.",
        "action": "Monitor every 2–3 days and remove infected tissue.",
        "about": "Early Blight causes dark spots on tomato leaves."
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "severity": "🔴 Severe",
        "health_score": 40,
        "recommendation": "Apply preventive fungicide immediately.",
        "action": "Isolate affected plants and avoid overhead watering.",
        "about": "Late Blight can rapidly destroy tomato crops."
    },

    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "severity": "🟡 Moderate",
        "health_score": 72,
        "recommendation": "Apply copper fungicide and improve airflow.",
        "action": "Remove infected leaves and inspect every 3 days.",
        "about": "Leaf Mold is a fungal disease affecting tomato leaves."
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "severity": "🟡 Moderate",
        "health_score": 68,
        "recommendation": "Remove infected leaves and improve airflow.",
        "action": "Use fungicide and avoid overhead irrigation.",
        "about": "Septoria Leaf Spot causes circular leaf lesions."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "severity": "🟠 Moderate",
        "health_score": 60,
        "recommendation": "Apply neem oil or approved miticides.",
        "action": "Monitor undersides of leaves regularly.",
        "about": "Spider mites feed on plant sap and weaken crops."
    },

    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "severity": "🟠 Moderate",
        "health_score": 65,
        "recommendation": "Apply fungicide and improve field hygiene.",
        "action": "Remove infected debris from the field.",
        "about": "Target Spot causes circular lesions on leaves."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "severity": "🔴 Severe",
        "health_score": 35,
        "recommendation": "Control whiteflies and remove infected plants.",
        "action": "Remove infected plants immediately.",
        "about": "A viral disease causing leaf curling and yellowing."
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "severity": "🔴 Severe",
        "health_score": 45,
        "recommendation": "Disinfect tools and remove infected plants.",
        "action": "Avoid handling healthy plants after infected ones.",
        "about": "Tomato Mosaic Virus reduces crop productivity."
    },

    "Tomato_healthy": {
        "crop": "Tomato",
        "severity": "🟢 Healthy",
        "health_score": 100,
        "recommendation": "Continue regular crop care.",
        "action": "Routine monitoring only.",
        "about": "No disease symptoms detected."
    },

    "Rice__Healthy_Rice_Leaf": {
        "crop": "Rice",
        "severity": "🟢 Healthy",
        "health_score": 100,
        "recommendation": "Continue regular crop care.",
        "action": "Routine monitoring only.",
        "about": "Healthy rice leaf with no disease symptoms."
    },

    "Rice__Bacterial_Leaf_Blight": {
        "crop": "Rice",
        "severity": "🔴 Severe",
        "health_score": 40,
        "recommendation": "Use resistant varieties and balanced fertilizer.",
        "action": "Improve drainage and remove infected plants.",
        "about": "Bacterial Leaf Blight is a serious disease causing leaf drying and yield loss."
    },

    "Rice__Brown_Spot": {
        "crop": "Rice",
        "severity": "🟠 Moderate",
        "health_score": 65,
        "recommendation": "Apply fungicide and improve soil nutrition.",
        "action": "Monitor affected plants and correct nutrient deficiencies.",
        "about": "Brown Spot causes brown lesions on rice leaves."
    },

    "Rice__Leaf_Blast": {
        "crop": "Rice",
        "severity": "🔴 Severe",
        "health_score": 35,
        "recommendation": "Apply appropriate fungicide immediately.",
        "action": "Inspect neighboring plants and isolate heavily infected areas.",
        "about": "Leaf Blast is one of the most destructive rice diseases."
    },
    
    "Rice__Leaf_scald": {
        "crop": "Rice",
        "severity": "🟡 Moderate",
        "health_score": 70,
        "recommendation": "Maintain balanced fertilization and field hygiene.",
        "action": "Remove infected debris and monitor crop regularly.",
        "about": "Leaf Scald causes drying and discoloration of rice leaves."
    },
    
    "Rice__Sheath_Blight": {
        "crop": "Rice",
        "severity": "🟠 Moderate",
        "health_score": 60,
        "recommendation": "Reduce excess nitrogen and apply fungicide.",
        "action": "Improve spacing between plants and monitor humidity.",
        "about": "Sheath Blight is a fungal disease affecting rice stems and leaves."
    },
}


def detect_leaf(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = leaf_model(img)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    return leaf_classes[predicted.item()]

    
# =========================
# Prediction Function
# =========================

def predict(image):

    leaf_result = detect_leaf(image)

    if leaf_result == "Non_Leaf":
        
        return """
# ⚠️ Invalid Image

The uploaded image is not a crop leaf.

Please upload:

🍅 Tomato Leaf
🥔 Potato Leaf
🫑 Bell Pepper Leaf

Requirements:
✅ Clear leaf image
✅ Good lighting
✅ Close-up image
"""

    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    confidence_percent = confidence.item() * 100

    if confidence_percent < 80:
        return """
# ⚠️ Invalid Image

Please upload a clear crop leaf image.

Supported Crops:

🍅 Tomato
🥔 Potato
🫑 Bell Pepper
"""

    disease = class_names[predicted.item()]

    info = disease_info.get(
        disease,
        {
            "crop": "Unknown",
            "severity": "Unknown",
            "health_score": 50,
            "recommendation": "Consult an agricultural expert.",
            "action": "Monitor crop and seek professional advice.",
            "about": "Information unavailable."
        }
    )

    result = f"""
# 🌱 AgriAI Crop Analysis

## 🌾 Crop
**{info['crop']}**

## 🦠 Disease
**{disease.replace('_', ' ')}**

## 🎯 Confidence
**{confidence_percent:.2f}%**

## ❤️ Health Score
**{info['health_score']}%**

## ⚠️ Severity
**{info['severity']}**

## 💊 Treatment
{info['recommendation']}

## 🚜 Farmer Action Plan
{info['action']}

## 📖 About Disease
{info['about']}
"""

    return result



def ask_agri_ai(question):

    if not question.strip():
        return "Please enter a question."

    try:

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are AgriAI, an expert agricultural assistant.

Provide practical advice for:
- Crop diseases
- Fertilizers
- Irrigation
- Pest management
- Yield improvement

Keep answers farmer-friendly and actionable.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.5,
            max_tokens=500
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"Error: {str(e)}"

        
# =========================
# Gradio UI
# =========================

with gr.Blocks(
    title="AgriAI"
) as demo:

    gr.Markdown(
        "# 🌱 AgriAI Smart Agriculture Platform"
    )

    with gr.Tabs():

        # Disease Detector Tab
        with gr.Tab("🌱 Disease Detector"):

            image_input = gr.Image(
                type="pil",
                label="Upload Crop Image"
            )

            disease_output = gr.Markdown()

            detect_btn = gr.Button(
                "Analyze Crop"
            )

            detect_btn.click(
                fn=predict,
                inputs=image_input,
                outputs=disease_output
            )

        # AI Chatbot Tab
        with gr.Tab("🤖 Ask AgriAI"):

            question_input = gr.Textbox(
                lines=4,
                placeholder="Ask any farming question..."
            )

            chatbot_output = gr.Markdown()

            ask_btn = gr.Button(
                "Ask AgriAI"
            )

            ask_btn.click(
                fn=ask_agri_ai,
                inputs=question_input,
                outputs=chatbot_output
            )

demo.launch()