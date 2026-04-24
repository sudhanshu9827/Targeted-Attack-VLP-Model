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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Setup
_yaml = yaml.YAML()
config = _yaml.load(open('configs/Retrieval_flickr_test.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
model, ref_model, tokenizer = load_model(
    'ALBEF',    
    'checkpoint/ALBEF/flickr30k.pth',
    'bert-base-uncased', # text encoder
    config,
    device
)
model = model.to(device)
model.eval()
print("✅ Model loaded!")

# Load UAP - use epoch 9!
uap_noise = torch.load(
    'output/ALBEF/flickr30k/uap_noise-9.pth',
    map_location=device
)
print("✅ UAP loaded! Shape:", uap_noise.shape)

# Load test data
with open('data_annotation/flickr30k_test.json', 'r') as f:
    test_data = json.load(f)

all_texts = []
all_image_paths = []
for item in test_data:
    all_texts.extend(item['caption'])
    all_image_paths.append(
        os.path.join(config['image_root'], item['image'])
    )

print(f"Total texts: {len(all_texts)}")
print(f"Total images: {len(all_image_paths)}")

# Encode ALL texts
print("Encoding all texts...")
all_text_feats = []
for i in range(0, len(all_texts), 64):
    batch_texts = all_texts[i:i+64]
    with torch.no_grad():
        text_inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt'
        ).to(device)
        feat = model.inference_text(text_inputs)['text_feat']
        all_text_feats.append(feat.cpu())
all_text_feats = torch.cat(all_text_feats, dim=0)
print(f"✅ Text features: {all_text_feats.shape}")

# Transform
# Convert raw image → model-ready tensor
transform = transforms.Compose([
    transforms.Resize(
        (config['image_res'], config['image_res']),
        interpolation=Image.BICUBIC
    ),
    transforms.ToTensor(),
])

# Collect results
results = []
for idx in range(len(all_image_paths)):
    if len(results) >= 5:
        break

    img_path = all_image_paths[idx]
    if not os.path.exists(img_path):
        continue

    image = transform(Image.open(img_path).convert('RGB'))
    image = image.unsqueeze(0).to(device)
    gt_text = test_data[idx]['caption'][0]

    with torch.no_grad():
        clean_feat = model.inference_image(image)['image_feat'].cpu()
        # Ensure image stays in [0,1]
        adv_image = torch.clamp(image + uap_noise.unsqueeze(0), 0, 1)
        adv_feat = model.inference_image(adv_image)['image_feat'].cpu()

    clean_sim = (clean_feat @ all_text_feats.T).squeeze()
    adv_sim = (adv_feat @ all_text_feats.T).squeeze()

    clean_top_idx = clean_sim.argmax().item()
    adv_top_idx = adv_sim.argmax().item()
    attack_success = clean_top_idx != adv_top_idx

    # Only show successful attacks
    if not attack_success:
        continue

    results.append({
        'gt_text': gt_text,
        'clean_retrieved': all_texts[clean_top_idx],
        'adv_retrieved': all_texts[adv_top_idx],
        'attack_success': attack_success,
        'image': image[0].cpu(),
        'adv_image': adv_image[0].cpu(),
        'uap': uap_noise.cpu(),
    })

    print(f"✅ Found success {len(results)}/5: {os.path.basename(img_path)}")

print(f"\nGenerating visualization for {len(results)} examples...")

# Plot: 4 columns per example
# Col 1: Original image
# Col 2: UAP (perturbation amplified)
# Col 3: Adversarial image
# Col 4: Text info

fig = plt.figure(figsize=(20, 5 * len(results)))
fig.patch.set_facecolor('#0d0d0d')

for i, r in enumerate(results):
    # Original image
    ax1 = fig.add_subplot(len(results), 4, i*4 + 1)
    img_np = r['image'].permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    ax1.imshow(img_np)
    ax1.set_title('Original Image', color='white', fontsize=10, pad=8)
    ax1.axis('off')
    # Green border
    for spine in ax1.spines.values():
        spine.set_edgecolor('#00ff88')
        spine.set_linewidth(2)

    # UAP (amplified for visibility)
    ax2 = fig.add_subplot(len(results), 4, i*4 + 2)
    uap_np = r['uap'].detach().permute(1, 2, 0).numpy()
    # Amplify perturbation 10x for visibility
    uap_vis = np.clip((uap_np * 10) + 0.5, 0, 1)
    ax2.imshow(uap_vis)
    ax2.set_title('Perturbation (10x amplified)\nInvisible to human eye!',
                  color='white', fontsize=9, pad=8)
    ax2.axis('off')
    # Yellow border
    for spine in ax2.spines.values():
        spine.set_edgecolor('#ffdd00')
        spine.set_linewidth(2)

    # Adversarial image
    ax3 = fig.add_subplot(len(results), 4, i*4 + 3)
    adv_np = r['adv_image'].permute(1, 2, 0).numpy()
    adv_np = np.clip(adv_np, 0, 1)
    ax3.imshow(adv_np)
    ax3.set_title('Adversarial Image\n(Original + Perturbation)',
                  color='white', fontsize=9, pad=8)
    ax3.axis('off')
    # Red border
    for spine in ax3.spines.values():
        spine.set_edgecolor('#ff4444')
        spine.set_linewidth(2)

    # Text info panel
    ax4 = fig.add_subplot(len(results), 4, i*4 + 4)
    ax4.set_facecolor('#1a1a2e')
    ax4.axis('off')

    # Ground truth
    ax4.text(0.05, 0.92, 'Ground Truth:', color='#00ff88',
             fontsize=9, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.05, 0.80, r['gt_text'][:80] + ('...' if len(r['gt_text']) > 80 else ''),
             color='white', fontsize=8, wrap=True,
             transform=ax4.transAxes)

    # Clean retrieved
    ax4.text(0.05, 0.60, 'Clean Retrieved:', color='#00aaff',
             fontsize=9, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.05, 0.48, r['clean_retrieved'][:80] + ('...' if len(r['clean_retrieved']) > 80 else ''),
             color='#aaaaaa', fontsize=8, wrap=True,
             transform=ax4.transAxes)

    # Attack retrieved
    ax4.text(0.05, 0.28, '⚠ Attack Retrieved:', color='#ff4444',
             fontsize=9, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.05, 0.16, r['adv_retrieved'][:80] + ('...' if len(r['adv_retrieved']) > 80 else ''),
             color='#ffaaaa', fontsize=8, wrap=True,
             transform=ax4.transAxes)

    # Status
    ax4.text(0.05, 0.04, '✅ ATTACK SUCCESS',
             color='#ff4444', fontsize=10, fontweight='bold',
             transform=ax4.transAxes)

plt.suptitle('C-PGC Universal Adversarial Attack — ALBEF on Flickr30K\n'
             'One perturbation fools the model across all images',
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('attack_visualization_full.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0d0d0d')
print("✅ Saved: attack_visualization_full.png")