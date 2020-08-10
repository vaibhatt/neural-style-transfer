from main import run_style_transfer
from torchvision.utils import save_image
import config
from torchvision import models

cnn = models.vgg19(pretrained=True).features.to(config.device).eval()
if __name__ == "__main__":
    output = run_style_transfer(cnn)
    save_image(output, config.output_path)