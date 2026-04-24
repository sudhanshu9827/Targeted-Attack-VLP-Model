import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import ruamel.yaml as yaml
import sys
import os
sys.path.append('.')
from utils import load_model
from dataset import paired_dataset
from torch.utils.data import DataLoader

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
print("Loading UAP...")
uap_noise = torch.load(
    'output/ALBEF/flickr30k/uap_noise-5.pth',
    map_location=device
)
print("✅ UAP loaded! Shape:", uap_noise.shape)

# Load dataset
transform = transforms.Compose([
    transforms.Resize(
        (config['image_res'], config['image_res']),
        interpolation=Image.BICUBIC
    ),
    transforms.ToTensor(),
])

dataset = paired_dataset(
    config['annotation_file'],
    transform,
    config['image_root']
)

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=2,
    collate_fn=dataset.collate_fn
)

print("\n=== VISUALIZING ATTACK RESULTS ===\n")

results = []
count = 0

with torch.no_grad():
    for batch_idx, (images, texts_group, img_ids, text_ids) in enumerate(loader):
        if count >= 10:  # Show 10 examples
            break

        images = images.to(device)
        texts = texts_group[0]  # First caption

        # Get ALL captions in dataset for retrieval
        all_texts = []
        for t_group in texts_group:
            all_texts.extend(t_group)

        # Clean image retrieval
        text_inputs = tokenizer(
            all_texts,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt'
        ).to(device)

        clean_img_feat = model.inference_image(images)['image_feat']
        txt_feats = model.inference_text(text_inputs)['text_feat']

        # Adversarial image
        adv_images = torch.clamp(images + uap_noise.unsqueeze(0), 0, 1)
        adv_img_feat = model.inference_image(adv_images)['image_feat']

        # Compute similarities
        clean_sim = (clean_img_feat @ txt_feats.T).squeeze()
        adv_sim = (adv_img_feat @ txt_feats.T).squeeze()

        # Get top retrieved text
        if clean_sim.dim() == 0:
            continue

        clean_top_idx = clean_sim.argmax().item()
        adv_top_idx = adv_sim.argmax().item()

        correct_text = all_texts[0]  # Ground truth
        clean_retrieved = all_texts[clean_top_idx]
        adv_retrieved = all_texts[adv_top_idx]

        # Check if attack succeeded
        attack_success = clean_top_idx != adv_top_idx

        results.append({
            'img_id': img_ids[0],
            'correct_text': correct_text,
            'clean_retrieved': clean_retrieved,
            'adv_retrieved': adv_retrieved,
            'attack_success': attack_success,
            'image': images[0].cpu(),
            'adv_image': adv_images[0].cpu()
        })

        count += 1
        print(f"Example {count}:")
        print(f"  Ground Truth:     {correct_text}")
        print(f"  Clean Retrieved:  {clean_retrieved}")
        print(f"  Attack Retrieved: {adv_retrieved}")
        print(f"  Attack Success:   {'✅ YES' if attack_success else '❌ NO'}")
        print()

# Save visualization
print("\nSaving visualization...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
fig.suptitle('C-PGC Universal Attack Visualization\nALBEF on Flickr30K', 
             fontsize=16, fontweight='bold')

for i, result in enumerate(results[:5]):
    # Original image
    img = result['image'].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(
        f'Clean Image\nGT: {result["correct_text"][:50]}...\nRetrieved: {result["clean_retrieved"][:50]}...',
        fontsize=8, color='green'
    )
    axes[i, 0].axis('off')

    # Adversarial image
    adv_img = result['adv_image'].permute(1, 2, 0).numpy()
    adv_img = np.clip(adv_img, 0, 1)
    axes[i, 1].imshow(adv_img)
    color = 'red' if result['attack_success'] else 'orange'
    status = '✅ ATTACK SUCCESS' if result['attack_success'] else '❌ ATTACK FAILED'
    axes[i, 1].set_title(
        f'Adversarial Image ({status})\nRetrieved: {result["adv_retrieved"][:50]}...',
        fontsize=8, color=color
    )
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('attack_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Saved as attack_visualization.png!")

# Print Summary
print("\n=== SUMMARY ===")
successes = sum(1 for r in results if r['attack_success'])
print(f"Total examples: {len(results)}")
print(f"Attack successes: {successes}")
print(f"Attack Success Rate: {100*successes/len(results):.1f}%")
