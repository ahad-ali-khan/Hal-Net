import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ExifTags
from scipy.ndimage import gaussian_filter, sobel
import gradio as gr
import os

# ── Architecture ──────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class HALNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            ResBlock(128), ResBlock(128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3), nn.Sigmoid()
        )
    def forward(self, x):
        return self.regressor(self.features(x))

# ── Load model ────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = HALNet().to(DEVICE)
model.load_state_dict(torch.load("halnet_best.pt", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def fix_orientation(img):
    try:
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if val == 3:  img = img.rotate(180, expand=True)
                    elif val == 6: img = img.rotate(270, expand=True)
                    elif val == 8: img = img.rotate(90,  expand=True)
    except Exception:
        pass
    return img

def pad_to_square(img):
    w, h    = img.size
    max_s   = max(w, h)
    pad_w   = (max_s - w) // 2
    pad_h   = (max_s - h) // 2
    return TF.pad(img, (pad_w, pad_h, max_s-w-pad_w, max_s-h-pad_h), fill=0)

def apply_halation(img_array, radius_norm, intensity, warmth):
    img = img_array.astype(np.float32) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    sigma   = max(radius_norm * 32.0, 1.0)
    luma    = 0.299*R + 0.587*G + 0.114*B
    mask    = np.clip((luma - 0.65) / 0.35, 0, 1)
    bloom   = gaussian_filter(mask, sigma=sigma)
    bloom  /= (bloom.max() + 1e-6)
    out     = img.copy()
    out[:,:,0] = np.clip(out[:,:,0] + bloom * intensity * (0.6 + 0.4*warmth), 0, 1)
    out[:,:,1] = np.clip(out[:,:,1] + bloom * intensity * 0.12,               0, 1)
    out[:,:,2] = np.clip(out[:,:,2] + bloom * intensity * max(0, 0.08-0.08*warmth), 0, 1)
    return (out * 255).astype(np.uint8)

# ── Inference ─────────────────────────────────────────────────────────────────

def predict(input_image, manual_radius, manual_intensity, manual_warmth, use_manual):
    if input_image is None:
        return None, "Upload an image to get started."

    img    = fix_orientation(Image.fromarray(input_image).convert("RGB"))
    img_sq = pad_to_square(img)
    inp    = transform(img_sq).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p = model(inp).squeeze().cpu().numpy()

    if use_manual:
        radius    = manual_radius
        intensity = manual_intensity
        warmth    = manual_warmth
        mode      = "manual override"
    else:
        radius, intensity, warmth = float(p[0]), float(p[1]), float(p[2])
        mode = "HAL-Net prediction"

    result = apply_halation(np.array(img), radius, intensity, warmth)

    feedback = f"""Mode:        {mode}

HAL-Net raw output
  Radius:    {p[0]:.3f}  ({p[0]*32:.1f}px spread)
  Intensity: {p[1]:.3f}
  Warmth:    {p[2]:.3f}

Applied parameters
  Radius:    {radius:.3f}  ({radius*32:.1f}px spread)
  Intensity: {intensity:.3f}
  Warmth:    {warmth:.3f}

Warmth guide
  0.0 - 0.3  neutral / white glow
  0.3 - 0.6  warm golden
  0.6 - 1.0  CineStill red-orange"""

    return Image.fromarray(result), feedback

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Monochrome(), title="HAL-Net") as demo:
    gr.Markdown("""
# HAL-Net
### Analog Film Halation Estimation and Synthesis

A CNN trained on 100 authentic CineStill 800T film scans.
Upload any image — HAL-Net estimates halation parameters from highlight regions
and synthesizes a warm analog film glow.

Inspired by [FGA-NN](https://arxiv.org/abs/2506.14350) (Ameur et al., 2025).
""")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image", type="numpy")

            gr.Markdown("### Manual Override")
            gr.Markdown("Adjust sliders to override HAL-Net predictions.")

            use_manual     = gr.Checkbox(label="Use manual parameters", value=False)
            manual_radius  = gr.Slider(0.05, 1.0, value=0.3,  step=0.01, label="Radius (0=tight, 1=wide spread)")
            manual_intensity = gr.Slider(0.0,  1.0, value=0.4,  step=0.01, label="Intensity (glow strength)")
            manual_warmth  = gr.Slider(0.0,  1.0, value=0.7,  step=0.01, label="Warmth (0=white, 1=CineStill red)")

            run_btn = gr.Button("Apply HAL-Net Halation", variant="primary")

        with gr.Column():
            image_output   = gr.Image(label="Output", type="pil")
            params_output  = gr.Textbox(label="Parameter Readout", lines=14, interactive=False)

    gr.Markdown("""
---
**What is halation?**
Halation is the red-orange glow around bright light sources in analog film photography.
It occurs when light passes through the emulsion, reflects off the film base, and exposes
the silver halide crystals from behind. CineStill 800T is particularly known for this
effect because its anti-halation layer was removed during processing for cinema use.

**Dataset**
100 handpicked CineStill 800T scans from Flickr and Lomography exhibiting authentic halation.
Train / val / test split: 70 / 15 / 15.
""")

    run_btn.click(
        fn=predict,
        inputs=[image_input, manual_radius, manual_intensity, manual_warmth, use_manual],
        outputs=[image_output, params_output]
    )

demo.launch()