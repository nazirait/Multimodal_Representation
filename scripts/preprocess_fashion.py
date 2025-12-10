import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Paths
PROCESSED = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/processed/fashion')
SPLITS = ['train', 'val', 'test']  # Add/remove as needed

# Image transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ResNet18 (no classifier head)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

def embed_images(csv_path, output_path):
    df = pd.read_csv(csv_path)
    features = []
    for img_path in tqdm(df['image_path'], desc=f'Embedding {csv_path.name}'):
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(inp).cpu().numpy().flatten()
        features.append(feat)
    arr = np.stack(features)
    np.save(output_path, arr)
    print(f"Saved: {output_path} | Shape: {arr.shape}")

# Loop over all splits that exist
for split in SPLITS:
    csv_path = PROCESSED / f"{split}.csv"
    out_path = PROCESSED / f"{split}_image_emb.npy"
    if csv_path.exists():
        embed_images(csv_path, out_path)
