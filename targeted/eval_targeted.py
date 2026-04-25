import argparse
import os
import torch
import torch.nn.functional as F
import json
from torchvision import transforms
from PIL import Image
import ruamel.yaml as yaml
from utils import load_model


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='../configs/Retrieval_flickr_test.yaml')
parser.add_argument('--source_model', default='ALBEF')
parser.add_argument('--text_encoder', default='bert-base-uncased')
parser.add_argument('--checkpoint', default='../checkpoint/ALBEF/flickr30k.pth')

parser.add_argument('--uap_path', required=True)
parser.add_argument('--target_text', required=True)

parser.add_argument('--max_samples', type=int, default=1000)
parser.add_argument('--eps', type=float, default=12)

args = parser.parse_args()


# =========================
# SETUP
# =========================
_yaml = yaml.YAML()
config = _yaml.load(open(args.config))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = args.eps / 255.0


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
# LOAD UAP (SAFE)
# =========================
ckpt = torch.load(args.uap_path, map_location=device)

if isinstance(ckpt, dict) and "generator" in ckpt:
    raise ValueError("This script expects DELTA. Use generator version if needed.")

uap_noise = ckpt.to(device)

# shape fix
if uap_noise.dim() == 3:
    uap_noise = uap_noise.unsqueeze(0)

if uap_noise.size(0) != 1:
    print("⚠️ Fixing non-universal UAP")
    uap_noise = uap_noise.mean(dim=0, keepdim=True)

# clamp
uap_noise = torch.clamp(uap_noise, -eps, eps)

print(f"✅ Loaded UAP: {uap_noise.shape}")


# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((config['image_res'], config['image_res'])),
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


# =========================
# BUILD TEXT DATABASE (FIXED)
# =========================
print("Encoding all texts...")

all_texts = []
for item in test_data:
    caps = item['caption']
    if not isinstance(caps, list):
        caps = [caps]
    all_texts.extend(caps)

batch_size = 64
all_text_feats = []

for i in range(0, len(all_texts), batch_size):

    batch = all_texts[i:i+batch_size]

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

all_text_feats = torch.cat(all_text_feats, dim=0).to(device)

print(f"Total captions: {len(all_texts)}")


# =========================
# TARGET INDEX
# =========================
with torch.no_grad():
    target_input = tokenizer([args.target_text], return_tensors='pt').to(device)

    target_feat = model.inference_text(target_input)['text_feat']
    target_feat = F.normalize(target_feat, dim=-1)

target_idx = (target_feat @ all_text_feats.T).argmax().item()

print(f"\n🎯 Target: {args.target_text}")
print(f"Closest DB text: {all_texts[target_idx]}")


# =========================
# EVALUATION
# =========================
print("\nEvaluating...")

total = 0
untargeted = 0
targeted_r1 = 0
targeted_r5 = 0
targeted_r10 = 0

caption_ptr = 0

for item in test_data[:args.max_samples]:

    captions = item['caption']
    if not isinstance(captions, list):
        captions = [captions]

    num_caps = len(captions)

    img_path = os.path.join(config['image_root'], item['image'])

    if not os.path.exists(img_path):
        caption_ptr += num_caps
        continue

    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():

        clean_feat = model.inference_image(normalize(image))['image_feat']
        clean_feat = F.normalize(clean_feat, dim=-1)

        delta = F.interpolate(uap_noise, size=image.shape[-2:], mode='bilinear')

        adv_image = torch.clamp(image + delta, 0, 1)

        adv_feat = model.inference_image(normalize(adv_image))['image_feat']
        adv_feat = F.normalize(adv_feat, dim=-1)

    clean_sim = (clean_feat @ all_text_feats.T).squeeze()
    adv_sim = (adv_feat @ all_text_feats.T).squeeze()

    clean_top = clean_sim.argmax().item()

    gt_range = list(range(caption_ptr, caption_ptr + num_caps))
    caption_ptr += num_caps

    if clean_top not in gt_range:
        continue

    total += 1

    adv_top = adv_sim.argmax().item()

    if adv_top not in gt_range:
        untargeted += 1

    # R@1
    if adv_top == target_idx:
        targeted_r1 += 1

    # R@5
    if target_idx in adv_sim.topk(5).indices:
        targeted_r5 += 1

    # R@10
    if target_idx in adv_sim.topk(10).indices:
        targeted_r10 += 1


# =========================
# RESULTS
# =========================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"Total samples: {total}")

print(f"\nUntargeted ASR: {100*untargeted/max(total,1):.2f}%")
print(f"Targeted R@1:   {100*targeted_r1/max(total,1):.2f}%")
print(f"Targeted R@5:   {100*targeted_r5/max(total,1):.2f}%")
print(f"Targeted R@10:  {100*targeted_r10/max(total,1):.2f}%")

print("="*60)