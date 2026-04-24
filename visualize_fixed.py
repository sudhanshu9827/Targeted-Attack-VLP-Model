import torch
import numpy as np
from PIL import Image
import json
import sys
import os
sys.path.append('.')
from torchvision import transforms
import ruamel.yaml as yaml
from utils import load_model
from torch.utils.data import DataLoader
from dataset import paired_dataset

# Setup
_yaml = yaml.YAML()
config = _yaml.load(open('configs/Retrieval_flickr_test.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
model, ref_model, tokenizer = load_model(
    'ALBEF',
    'checkpoint/ALBEF/flickr30k.pth',
    'bert-base-uncased',
    config,
    device
)
model = model.to(device)
model.eval()
print("✅ Model loaded!")

# Load UAP
uap_noise = torch.load(
    'output/ALBEF/flickr30k/uap_noise-5.pth',
    map_location=device
)
print("✅ UAP loaded!")

# Load ALL test texts from annotation file
print("Loading all test texts...")
with open('data_annotation/flickr30k_test.json', 'r') as f:
    test_data = json.load(f)

all_texts = []
all_image_paths = []
for item in test_data:
    all_texts.extend(item['caption'])
    all_image_paths.append(
        os.path.join(config['image_root'], item['image'])
    )

print(f"Total texts in database: {len(all_texts)}")
print(f"Total images: {len(all_image_paths)}")

# Encode ALL texts
print("Encoding all texts...")
batch_size = 64
all_text_feats = []

for i in range(0, len(all_texts), batch_size):
    batch_texts = all_texts[i:i+batch_size]
    with torch.no_grad():
        text_inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt'
        ).to(device)
        text_feat = model.inference_text(text_inputs)['text_feat']
        all_text_feats.append(text_feat.cpu())

all_text_feats = torch.cat(all_text_feats, dim=0)
print(f"✅ Text features: {all_text_feats.shape}")

# Transform
transform = transforms.Compose([
    transforms.Resize(
        (config['image_res'], config['image_res']),
        interpolation=Image.BICUBIC
    ),
    transforms.ToTensor(),
])

# Evaluate on first 10 images
print("\n=== ATTACK RESULTS ===\n")

for idx in range(min(10, len(all_image_paths))):
    # Load image
    img_path = all_image_paths[idx]
    if not os.path.exists(img_path):
        continue

    image = transform(Image.open(img_path).convert('RGB'))
    image = image.unsqueeze(0).to(device)

    # Ground truth text
    gt_text = test_data[idx]['caption'][0]

    with torch.no_grad():
        # Clean features
        clean_feat = model.inference_image(image)['image_feat'].cpu()

        # Adversarial features
        adv_image = torch.clamp(image + uap_noise.unsqueeze(0), 0, 1)
        adv_feat = model.inference_image(adv_image)['image_feat'].cpu() 



    # Compute similarities against ALL texts
    clean_sim = (clean_feat @ all_text_feats.T).squeeze()
    adv_sim = (adv_feat @ all_text_feats.T).squeeze()

    # Get top-1 retrieved text
    clean_top_idx = clean_sim.argmax().item()
    adv_top_idx = adv_sim.argmax().item()

    clean_retrieved = all_texts[clean_top_idx]
    adv_retrieved = all_texts[adv_top_idx]

    attack_success = clean_top_idx != adv_top_idx

    print(f"Image {idx+1}: {os.path.basename(img_path)}")
    print(f"  Ground Truth:     {gt_text}")
    print(f"  Clean Retrieved:  {clean_retrieved}")
    print(f"  Attack Retrieved: {adv_retrieved}")
    print(f"  Attack Success:   {'✅ YES' if attack_success else '❌ NO'}")
    print(f"  Semantic Change:  {'🎯 DIFFERENT' if attack_success else '😐 SIMILAR'}")
    print()

