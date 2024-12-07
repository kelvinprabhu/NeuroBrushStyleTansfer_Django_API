import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from torchvision import transforms, models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((298, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
def load_image(image_path, title):
    image = Image.open(image_path)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()
    image = transform(image).unsqueeze(0)
    return image.to(device)
content_image_path = r"Wonders of World\Wonders of World\chichen_itza\0a66edf951.jpg"
content_image = load_image(content_image_path, "Content Image")
style_image_path = r"images\images\Alfred_Sisley\Alfred_Sisley_2.jpg"
style_image = load_image(style_image_path, "Style Image")


def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram
vgg = models.vgg19(weights=True).features.to(device).eval()

feature_layers = {}
layer_count = 0

for name, layer in vgg._modules.items():
    if isinstance(layer, torch.nn.Conv2d):
        layer_name = f'conv{layer_count//4 + 1}_{layer_count%4 + 1}'
        feature_layers[name] = layer_name
        layer_count += 1

content_layer = 'conv4_2'
num_style_layers = len(feature_layers) - 1
style_layers_dict = {layer: 1.0 / num_style_layers for layer in feature_layers.values()}
content_features = get_features(content_image, vgg, feature_layers)
style_features = get_features(style_image, vgg, feature_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
target = content_image.clone().requires_grad_(True).to(device)



style_weight = 1e6
content_weight = 1
optimizer = optim.Adam([target], lr=0.003)
steps = 1000


style_losses = []
content_losses = []

for epoch in range(steps + 1):
    optimizer.zero_grad()
    
    target_features = get_features(target, vgg, feature_layers)
    
    content_loss = F.mse_loss(target_features[content_layer], content_features[content_layer])
    content_losses.append(content_loss.item())
    
    style_loss = 0
    for layer in style_layers_dict:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_layers_dict[layer] * F.mse_loss(target_gram, style_gram)
        style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2] * target_feature.shape[3])

    style_losses.append(style_loss.item())
    
    neural_loss = content_weight * content_loss + style_weight * style_loss
    neural_loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{steps}], Content Loss: {content_loss.item():.2}, Style Loss {style_loss.item():.2}')

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image