print("🔥 REAL AI MODEL LOADED 🔥")

import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

MODEL_PATH = "models/plant_disease/model.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=torch.device("cpu")
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_disease(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

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

    disease = class_names[predicted.item()]

    disease_info = {

        "Tomato_Leaf_Mold": {
            "severity": "Moderate",
            "health_score": 72,
            "recommendation":
                "Apply copper fungicide and improve airflow around plants.",
            "about":
                "Leaf Mold is a fungal disease that affects tomato leaves and reduces yield."
        },

        "Tomato_Early_blight": {
            "severity": "Moderate",
            "health_score": 65,
            "recommendation":
                "Remove infected leaves and spray recommended fungicide.",
            "about":
                "Early Blight causes dark spots on leaves and reduces crop productivity."
        },

        "Tomato_Late_blight": {
            "severity": "Severe",
            "health_score": 40,
            "recommendation":
                "Apply preventive fungicide and avoid overhead watering.",
            "about":
                "Late Blight spreads rapidly and can destroy tomato crops."
        },

        "Tomato_Bacterial_spot": {
            "severity": "Moderate",
            "health_score": 60,
            "recommendation":
                "Use copper-based sprays and disease-free seeds.",
            "about":
                "Bacterial Spot causes lesions on leaves and fruits."
        },

        "Tomato_healthy": {
            "severity": "Healthy",
            "health_score": 100,
            "recommendation":
                "Plant appears healthy. Continue regular monitoring.",
            "about":
                "No disease symptoms detected."
        }
    }

    info = disease_info.get(
        disease,
        {
            "severity": "Unknown",
            "health_score": 50,
            "recommendation":
                "Consult an agricultural expert.",
            "about":
                "Information unavailable."
        }
    )

    print(
        "PREDICTED:",
        disease,
        confidence.item() * 100
    )

    return {
        "status": "success",
        "disease": disease,
        "confidence": round(
            confidence.item() * 100,
            2
        ),
        "severity": info["severity"],
        "health_score": info["health_score"],
        "recommendation": info["recommendation"],
        "about": info["about"]
    }