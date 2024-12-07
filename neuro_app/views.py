from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import torch.nn.functional as F
from django.conf import settings
from PIL import Image
import torch
from torchvision import transforms, models
import os
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((298, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Helper Functions
def validate_and_save_images(request):
    """
    Validates input images and saves them to the server.
    """
    content_image = request.FILES.get('content_image')
    style_image = request.FILES.get('style_image')
    
    if not content_image or not style_image:
        raise ValueError("Both content and style images are required.")
    
    content_path = os.path.join(settings.MEDIA_ROOT, f"{uuid.uuid4()}_content.jpg")
    style_path = os.path.join(settings.MEDIA_ROOT, f"{uuid.uuid4()}_style.jpg")

    try:
        with open(content_path, 'wb+') as content_file:
            for chunk in content_image.chunks():
                content_file.write(chunk)

        with open(style_path, 'wb+') as style_file:
            for chunk in style_image.chunks():
                style_file.write(chunk)
    except Exception as e:
        raise IOError(f"Failed to save images: {str(e)}")

    return content_path, style_path


def load_and_preprocess_images(content_path, style_path):
    """
    Loads and preprocesses content and style images.
    """
    try:
        content = load_image(content_path).to(device)
        style = load_image(style_path).to(device)
    except Exception as e:
        raise ValueError(f"Failed to load images: {str(e)}")

    if content.shape[1:] != style.shape[1:]:
        raise ValueError("Content and style images must have the same dimensions.")
    
    return content, style


def setup_vgg19():
    """
    Sets up the VGG19 model and defines feature extraction layers.
    """
    try:
        vgg = models.vgg19(weights="IMAGENET1K_V1").features.to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load VGG19 model: {str(e)}")

    feature_layers = {}
    layer_count = 0
    for name, layer in vgg._modules.items():
        if isinstance(layer, torch.nn.Conv2d):
            layer_name = f'conv{layer_count // 4 + 1}_{layer_count % 4 + 1}'
            feature_layers[name] = layer_name
            layer_count += 1

    content_layer = 'conv4_2'
    num_style_layers = len(feature_layers) - 1
    style_layers_dict = {layer: 1.0 / num_style_layers for layer in feature_layers.values()}

    return vgg, feature_layers, content_layer, style_layers_dict


def extract_features(content, style, vgg, feature_layers):
    """
    Extracts content and style features from the VGG19 model.
    """
    try:
        content_features = get_features(content, vgg, feature_layers)
        style_features = get_features(style, vgg, feature_layers)
        style_grams = {
            layer: gram_matrix(style_features[layer]) for layer in style_features
        }
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

    return content_features, style_features, style_grams


def style_transfer(content, content_features, style_grams, feature_layers, content_layer, style_layers_dict, vgg):
    """
    Performs the style transfer optimization loop.
    """
    target = content.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=0.003)
    steps = 500

    try:
        for step in range(steps + 1):
            optimizer.zero_grad()
            target_features = get_features(target, vgg, feature_layers)

            # Compute content loss
            content_loss = F.mse_loss(
                target_features[content_layer], content_features[content_layer]
            )

            # Compute style loss
            style_loss = 0
            for layer in style_layers_dict:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_layers_dict[layer] * F.mse_loss(target_gram, style_gram)
                style_loss += layer_style_loss / (
                    target_feature.shape[1]
                    * target_feature.shape[2]
                    * target_feature.shape[3]
                )

            # Total loss
            total_loss = content_loss + 1e6 * style_loss
            total_loss.backward(retain_graph=True)  # Retain graph until the last step
            optimizer.step()
            if step % 10 == 0:
               print(f'Epoch [{step}/{steps}], Content Loss: {content_loss.item():.2}, Style Loss {style_loss.item():.2}')
    except Exception as e:
        raise RuntimeError(f"Style transfer process failed: {str(e)}")

    return target



# def save_result_image(target):
#     """
#     Converts the tensor to an image and saves it.
#     """
#     try:
#         result_image = im_convert(target)
#         result_filename = f"{uuid.uuid4()}_result.jpg"
#         result_path = os.path.join(settings.MEDIA_ROOT, "Public", result_filename)
#         result_image.save(result_path)
#     except Exception as e:
#         raise IOError(f"Failed to save result image: {str(e)}")

#     return f"{settings.MEDIA_URL}{os.path.basename(result_path)}"

def save_result_image(target):
    """
    Converts the tensor to an image, saves it, and returns a public URL.
    """
    try:
        result_image = im_convert(target)
        # Use the corrected MEDIA_ROOT
        result_path = os.path.join(settings.MEDIA_ROOT, f"{uuid.uuid4()}_result.jpg")
        result_image.save(result_path)
    except Exception as e:
        raise IOError(f"Failed to save result image: {str(e)}")

    # Return the public URL of the image
    return f"{settings.MEDIA_URL}{os.path.basename(result_path)}"



# Helper functions
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

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

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype('uint8'))

@csrf_exempt
@api_view(['POST'])
def style_transfer_api(request):
    """
    API to perform style transfer on content and style images uploaded by the user.
    """
    try:
        # Step 1: Validate and save input images
        content_path, style_path = validate_and_save_images(request)
        
        # Step 2: Load and preprocess images
        content, style = load_and_preprocess_images(content_path, style_path)
        
        # Step 3: Load the VGG19 model
        vgg, feature_layers, content_layer, style_layers_dict = setup_vgg19()

        # Step 4: Extract features
        content_features, style_features, style_grams = extract_features(
            content, style, vgg, feature_layers
        )

        # Step 5: Perform style transfer
        target = style_transfer(
            content,
            content_features,
            style_grams,
            feature_layers,
            content_layer,
            style_layers_dict,
            vgg
        )

        # Step 6: Convert target tensor to an image and save
        result_image_url = save_result_image(target)

        # Step 7: Return response
        return Response({'result_url': result_image_url})

    except Exception as e:
        return Response({'error': f"Unexpected error: {str(e)}"}, status=500)


@csrf_exempt
@api_view(['GET'])
def demo_api(request):
    data = {"message": "Hello, this is a demo API response!"}
    return Response(data)
