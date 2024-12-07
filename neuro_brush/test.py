from PIL import Image
from torchvision import transforms , models
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((298, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

content = load_image(r'D:\Projects\GAN\NeuroBrush\neuro_brush\2bdb7f3b-5c93-4073-a877-2c9299abbabc_content.jpg')
print(content.shape)
# neuro_brush\7c9c8ca3-5ba8-4f13-9cb0-230ab997b7c3_content.jpg
# Load VGG19 model
vgg = models.vgg19(weights="IMAGENET1K_V1").features.to(device).eval()
print(vgg)