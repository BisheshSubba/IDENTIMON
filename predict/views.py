from django.shortcuts import render
import torch
import torch.nn as nn
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from torchvision import transforms, models
import io
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'predict', 'animals_resnet34.pth')


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.to(device)
model.eval()

classes = [
    "dog", "horse", "elephant", "butterfly", 
    "chicken", "cat", "cow", "sheep", "spider", "squirrel"
]

@api_view(["POST"])
def prediction(request):
    try:
        # Check if image file is in the request
        if 'image' not in request.FILES:
            return Response({"error": "No image file provided."}, status=400)
        
        image_file = request.FILES['image']
        
        # Validate file type
        if not image_file.content_type.startswith('image/'):
            return Response({"error": "File is not an image."}, status=400)
        
        # Read and process the image
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device) 

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            confidence, predicted = torch.max(torch.softmax(output, dim=1), 1)

        predicted_class = classes[predicted.item()]
        confidence_percent = round(confidence.item() * 100, 2)

        return Response({
            "predicted_class": predicted_class,
            "confidence_percent": confidence_percent
        })

    except Exception as e:
        print("Prediction error:", e)
        return Response({"error": str(e)}, status=500)
    

def display(request):
    return render(request,"test.html")