import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available else "cpu"

class AdaIN(nn.Module):
    def __init__(self, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, content, style):
        style_std, style_mean = torch.std_mean(style, dim=[2, 3], keepdim=True)
        content_std, content_mean = torch.std_mean(content, dim=[2, 3], keepdim=True)

        normalized_content = self.norm(content)
        stylized_content = style_std * (normalized_content - content_mean) / content_std + style_mean

        return stylized_content

class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.vgg19(models.VGG19_Weights.DEFAULT).features.eval()

        # AdaIN layers
        self.ada_in1 = AdaIN(512)
        self.ada_in2 = AdaIN(512)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, content, style):
        # Encode content and style images
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)

        # Apply AdaIN to match style features with content features
        stylized_feat = self.ada_in1(content_feat, style_feat)
        
        # Decode stylized features to generate the output image
        output = self.decoder(stylized_feat)

        return output

transform = transforms.Compose(
    [
        transforms.Resize((800, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)

"""Load Images"""
content_image = load_image('house/original_house.jpg')
style_image = load_image('house/cubism.jpg')

model = STN().to(device)
output_image = model(content_image, style_image)
save_image(output_image, "house/house_cubism_adaIn.jpg")


