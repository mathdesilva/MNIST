import pathlib
import torch
import torchvision
from PIL import Image
import sys
sys.path.append("../models")
from models.model_class import ModelClass

class Model():
    MODEL_PATH = pathlib.Path().absolute() / "models" / "model.pt"

    def __init__(self):
        # Load model
        self.device = torch.device("cpu")
        self.model = ModelClass().to(self.device)
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model.eval()
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                            "Sneaker", "Bag", "Ankle boot"]

    def predict(self, input):
        # Perform prediction
        input = input.unsqueeze(0)
        result = self.model(input)

        return_value = {
            "class_id": torch.argmax(result).item(),
            "class_name": self._get_class_name(torch.argmax(result).item()),
            "confidence": torch.max(result).item(),
            "probs": result.tolist(),
            "labels": self.class_names
        }
        return return_value

    def _get_class_name(self, class_id):
        # Get class name from class id
        return self.class_names[class_id]

    def preprocess(self, file):
        # Perform preprocessing
        image = Image.open(file.file)
        image = image.resize((28, 28))
        
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)[0]

        return image_tensor

    