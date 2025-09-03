import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeModel  # your updated model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained deepfake model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeModel()
model.load_state_dict(torch.load("f5_resnet18.pth", map_location=device))  # new DFDC weights
model.eval()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "❌ Error: Image not found"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return "⚠️ No face detected → Cannot check deepfake"

    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]

    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    if prob > 0.5:
        return f"🟥 FAKE FACE (confidence: {prob*100:.2f}%)"
    else:
        return f"🟩 REAL FACE (confidence: {(1-prob)*100:.2f}%)"

# Example use
print(predict_image("test_images/test.jpg"))
