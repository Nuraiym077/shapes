from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import io
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn


class DatasetShapes(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.y = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 4)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.y(x)
    return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

shapes_app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DatasetShapes()
model.load_state_dict(torch.load("model_class.pth", map_location=device))
model.to(device)
model.eval()

@shapes_app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        image_data = await image.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": pred}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(shapes_app, host="127.0.0.1", port=8000)