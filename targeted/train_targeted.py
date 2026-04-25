import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import ruamel.yaml as yaml
import json

from utils import load_model
from generator_targeted import Generator
from dataset import paired_dataset


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='../configs/Retrieval_flickr_train.yaml')
parser.add_argument('--checkpoint', default='../checkpoint/ALBEF/flickr30k.pth')
parser.add_argument('--target_text', required=True)

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=20)
parser.add_argument('--alpha', type=float, default=0.01)

args = parser.parse_args()


# =========================
# SETUP
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_yaml = yaml.YAML()
config = _yaml.load(open(args.config))

save_path = "../targeted_output/ALBEF/flickr30k/"
os.makedirs(save_path, exist_ok=True)

eps = args.eps / 255.0


# =========================
# LOAD MODEL
# =========================
print("Loading model...")

model, _, tokenizer = load_model(
    "ALBEF",
    args.checkpoint,
    "bert-base-uncased",
    config,
    device
)

model = model.to(device)
model.eval()


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
# DATASET
# =========================
dataset = paired_dataset(
    config['annotation_file'],
    transform,
    config['image_root']
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collate_fn
)


# =========================
# FIXED TEXT LOADING (IMPORTANT FIX)
# =========================
print("Encoding all captions...")

with open(config['annotation_file'], 'r') as f:
    data = json.load(f)

all_texts = []

for item in data:
    caps = item['caption']
    if not isinstance(caps, list):
        caps = [caps]
    all_texts.extend(caps)


# =========================
# TEXT ENCODING (BATCHED)
# =========================
all_text_feats = []

for i in tqdm(range(0, len(all_texts), 64)):
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

all_text_feats = torch.cat(all_text_feats, dim=0).to(device)

print("Total captions:", len(all_texts))


# =========================
# TARGET TEXT
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
    target_txt_feat = F.normalize(target_txt_feat, dim=-1)

target_sim = (target_txt_feat @ all_text_feats.T).squeeze()
target_index = target_sim.argmax().item()

print(f"🎯 Target index: {target_index}")


# =========================
# GENERATOR (FIXED STRONG VERSION)
# =========================
generator = Generator(
    input_dim=100,
    num_filters=[[512, 256], [128, 64], [32, 16]],
    output_dim=3,
    batch_size=args.batch_size,
    first_kernel_size=4,
    context_dim=target_txt_feat.shape[-1]
).to(device)

optimizer = optim.Adam(generator.parameters(), lr=args.lr)


# =========================
# FIXED LATENT VECTOR
# =========================
z = torch.randn(1, 100, 3, 3).to(device)


# =========================
# TRAINING
# =========================
best_loss = float('inf')

print("\n🚀 Starting Training...\n")

for epoch in range(args.epochs):

    total_loss = 0

    for images, _, _, _ in tqdm(loader):

        images = images.to(device)
        B = images.size(0)

        # 🔥 add small noise (improves robustness)
        images = images + 0.01 * torch.randn_like(images)
        images = torch.clamp(images, 0, 1)

        z_batch = z.repeat(B, 1, 1, 1)
        target_batch = target_txt_feat.expand(B, -1)

        delta = generator(z_batch, target_batch)

        delta = F.interpolate(
            delta,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        delta = delta * eps
        delta = torch.clamp(delta, -eps, eps)

        adv_images = torch.clamp(images + delta, 0, 1)

        # =========================
        # FEATURES
        # =========================
        adv_feat = model.inference_image(
            normalize(adv_images)
        )['image_feat']

        adv_feat = F.normalize(adv_feat, dim=-1)

        with torch.no_grad():
            clean_feat = model.inference_image(
                normalize(images)
            )['image_feat']
            clean_feat = F.normalize(clean_feat, dim=-1)

        # =========================
        # LOSS
        # =========================
        logits = adv_feat @ all_text_feats.T / 0.07

        target_labels = torch.full(
            (B,),
            target_index,
            dtype=torch.long,
            device=device
        )

        loss_target = F.cross_entropy(logits, target_labels)

        # LDis
        sim_clean = F.cosine_similarity(adv_feat, clean_feat, dim=-1)
        loss_dis = sim_clean.mean()

        loss_reg = torch.mean(delta ** 2)

        loss = loss_target + 0.5 * loss_dis + args.alpha * loss_reg

        # =========================
        # BACKPROP
        # =========================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")

    # =========================
    # SAVE CORRECTLY
    # =========================
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save({
            "generator": generator.state_dict(),
            "z": z.detach()
        }, os.path.join(save_path, "best_targeted_uap.pth"))

        print("✅ Saved BEST GENERATOR")

print("\n🎉 Training Complete!")