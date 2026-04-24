import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
from torchvision import transforms
from PIL import Image
import ruamel.yaml as yaml
from utils import load_model


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--config',
    default='./targeted/configs/Retrieval_flickr_targeted_test.yaml')

parser.add_argument('--source_model', default='ALBEF')
parser.add_argument('--text_encoder', default='bert-base-uncased')

parser.add_argument('--checkpoint',
    default='./checkpoint/ALBEF/flickr30k.pth')

parser.add_argument('--uap_path', required=True)

parser.add_argument('--target_text', required=True)

parser.add_argument('--max_samples', type=int, default=1000)

args = parser.parse_args()


# =========================
# SETUP
# =========================
_yaml = yaml.YAML()
config = _yaml.load(open(args.config))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model, _, tokenizer = load_model(
    args.source_model,
    args.checkpoint,
    args.text_encoder,
    config,
    device
)

model = model.to(device)
model.eval()


# =========================
# LOAD UAP
# =========================
uap_noise = torch.load(args.uap_path, map_location=device)

if uap_noise.dim() == 3:
    uap_noise = uap_noise.unsqueeze(0)

print(f"✅ Loaded UAP from: {args.uap_path}")


# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize(
        (config['image_res'], config['image_res']),
        interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711)
)


# =========================
# LOAD DATA
# =========================
with open(config['annotation_file'], 'r') as f:
    test_data = json.load(f)

all_texts = []
all_image_paths = []

for item in test_data:
    all_texts.extend(item['caption'])
    all_image_paths.append(
        os.path.join(config['image_root'], item['image'])
    )


# =========================
# ENCODE ALL TEXTS
# =========================
print("Encoding all texts...")

all_text_feats = []

for i in range(0, len(all_texts), 64):
    batch = all_texts[i:i+64]

    with torch.no_grad():
        inp = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt'
        ).to(device)

        feat = model.inference_text(inp)['text_feat']
        feat = F.normalize(feat, dim=-1)

        all_text_feats.append(feat.cpu())

all_text_feats = torch.cat(all_text_feats, dim=0)


# =========================
# TARGET TEXT INDEX
# =========================
with torch.no_grad():
    target_input = tokenizer(
        [args.target_text],
        padding='max_length',
        truncation=True,
        max_length=30,
        return_tensors='pt'
    ).to(device)

    target_txt_feat = model.inference_text(target_input)['text_feat']
    target_txt_feat = F.normalize(target_txt_feat, dim=-1).cpu()

target_sim = (target_txt_feat @ all_text_feats.T).squeeze()
target_idx = target_sim.argmax().item()

print(f"\n🎯 Target: {args.target_text}")
print(f"Closest DB text: {all_texts[target_idx]}")


# =========================
# EVALUATION
# =========================
print("\nEvaluating Targeted Attack...")

total = 0
untargeted_success = 0
targeted_success = 0

caption_ptr = 0

for idx, item in enumerate(test_data[:args.max_samples]):

    img_path = os.path.join(config['image_root'], item['image'])

    if not os.path.exists(img_path):
        caption_ptr += len(item['caption'])
        continue

    image = transform(Image.open(img_path).convert('RGB'))
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        # Clean
        clean_feat = model.inference_image(
            normalize(image)
        )['image_feat']

        # Adversarial
        adv_image = torch.clamp(image + uap_noise, 0, 1)

        adv_feat = model.inference_image(
            normalize(adv_image)
        )['image_feat']

    # Normalize
    clean_feat = F.normalize(clean_feat, dim=-1).cpu()
    adv_feat = F.normalize(adv_feat, dim=-1).cpu()

    # Similarity
    clean_sim = (clean_feat @ all_text_feats.T).squeeze()
    adv_sim = (adv_feat @ all_text_feats.T).squeeze()

    clean_top = clean_sim.argmax().item()
    adv_top = adv_sim.argmax().item()

    # Ground truth indices
    gt_indices = list(range(
        caption_ptr,
        caption_ptr + len(item['caption'])
    ))

    caption_ptr += len(item['caption'])

    # Only evaluate correct clean retrieval
    if clean_top in gt_indices:

        total += 1

        # Untargeted success
        if adv_top not in gt_indices:
            untargeted_success += 1

        # Targeted success
        if adv_top == target_idx:
            targeted_success += 1


# =========================
# RESULTS
# =========================
print("\n" + "="*60)
print("TARGETED ATTACK RESULTS")
print("="*60)

print(f"Total valid samples: {total}")

print(f"\nUntargeted ASR: {100 * untargeted_success / max(total,1):.2f}%")
print(f"Targeted ASR:   {100 * targeted_success / max(total,1):.2f}%")

print("="*60)