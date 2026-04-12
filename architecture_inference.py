"""
Urban scene image segmentation - inference script

Usage:
    python architecture_inference.py --image photo.jpg
    python architecture_inference.py --image photo.jpg --checkpoint best_model_v2.pt --device cuda
    python architecture_inference.py --image photo.jpg --output result.png

Download weights from: https://drive.google.com/file/d/1kkUF_0pURXze-1QKRJFwJUW2Bd-tRDcO/view?usp=sharing 
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# ------------------- MODEL -------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        return out


# ------------------- INFERENCE -------------------
CLASS_NAMES = ["Background", "Person", "Car", "Bus", "Traffic light", "Skyscraper"]
CLASS_COLOURS = np.array([[0,0,0], [255,0,0], [0,0,255], [255,255,0], [0,255,0], [255,0,255]])


def load_model(checkpoint_path, device='cpu'):
    model = UNetResNet34(num_classes=6, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device='cpu'):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img_resized = img.resize((256, 256), Image.BILINEAR)
    
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).squeeze().cpu().numpy()
    
    pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST))
    return pred, np.array(img)


def visualize(image, pred, save_path=None):
    pred_color = CLASS_COLOURS[pred]
    blended = (image * 0.5 + pred_color * 0.5).astype(np.uint8)
    blended[pred == 0] = image[pred == 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis('off')
    axes[1].imshow(pred_color); axes[1].set_title("Prediction"); axes[1].axis('off')
    axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    print("\nDetected:")
    for u, c in zip(*np.unique(pred, return_counts=True)):
        print(f"  {CLASS_NAMES[u]}: {100*c/pred.size:.1f}%")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No args provided.")
        print("Usage: python architecture_inference.py --image photo.jpg")
        print("\nFor interactive use (Colab/Jupyter):")
        print("  from architecture_inference import load_model, predict, visualize")
        print("  model = load_model('best_model_v2.pt')")
        print("  pred, img = predict(model, 'photo.jpg')")
        print("  visualize(img, pred)")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description="Urban Scene Segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="best_model_v2.pt", help="Path to model weights")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detects)")
    parser.add_argument("--output", default=None, help="Save path (displays if not set)")
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(args.checkpoint, args.device)
    pred, img = predict(model, args.image, args.device)
    visualize(img, pred, args.output)
