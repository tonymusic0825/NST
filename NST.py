import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image


"""
Take the layers after the maxpool
"""

device = "cuda" if torch.cuda.is_available else "cpu"

"""
Neon art -> 1300 x 929
paris_tower -> 1280 x 720
"""
transform = transforms.Compose(
    [
        transforms.Resize((574, 864)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ]
)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)

"""Load Images"""
original_img = load_image('house/original_house.jpg')
style_img = load_image('house/romanticism.jpg')

generated_img = torch.randn(original_img.shape, device = device, requires_grad=True)
# generated_img = original_img.clone().requires_grad_(True)

# Parameters
model = VGG().to(device).eval()
epochs = 10000
lr = 0.01
alpha = 100 # Content weight
beta = 1000 # Style weight
optimizer = optim.Adam([generated_img], lr=lr)

for epoch in range(epochs):
    original_features = model(original_img)
    style_features = model(style_img)
    generated_features = model(generated_img)

    style_loss = original_loss = 0

    for gen_f, orig_f, style_f in zip(generated_features, original_features, style_features):
        batch_size, channel, height, width = gen_f.shape
        original_loss += torch.mean((gen_f - orig_f) ** 2)

        # Gram matrices
        G = gen_f.view(channel, height*width).mm(
            gen_f.view(channel, height*width).t()
        )

        S = style_f.view(channel, height*width).mm(
            style_f.view(channel, height*width).t()
        )

        style_loss += torch.mean((G - S)**2)

    total_loss = alpha*original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(total_loss)
        save_image(generated_img, "house/house_romanticism_whitenoise.jpg")