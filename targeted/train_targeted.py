import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import ruamel.yaml as yaml

from generator_targeted import Generator
from dataset import paired_dataset
from utils import load_model


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=40)   # 🔥 increased
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=12)
parser.add_argument('--alpha', type=float, default=0.01)  # 🔥 reduced

parser.add_argument('--target_text', required=True)

parser.add_argument('--source_model', default='ALBEF')
parser.add_argument('--source_text_encoder', default='bert-base-uncased')
parser.add_argument('--source_ckpt', default='../checkpoint/ALBEF/flickr30k.pth')

parser.add_argument('--save_dir', default='../targeted_output')

args = parser.parse_args()


# =========================
# SETUP
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# =========================
# LOAD CONFIG
# =========================
_yaml = yaml.YAML()

config_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "configs",
    "Retrieval_flickr_train.yaml"
)

config = _yaml.load(open(config_path))


# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model, _, tokenizer = load_model(
    args.source_model,
    args.source_ckpt,
    args.source_text_encoder,
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
    transforms.ToTensor()
])

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711)
)


# =========================
# TARGET TEXT
# =========================
with torch.no_grad():
    text_inputs = tokenizer(
        [args.target_text],
        padding='max_length',
        truncation=True,
        max_length=30,
        return_tensors='pt'
    ).to(device)

    target_txt_feat = model.inference_text(text_inputs)['text_feat']
    target_txt_feat = F.normalize(target_txt_feat, dim=-1).detach()

print(f"🎯 Target text: {args.target_text}")


# =========================
# DATASET
# =========================
train_dataset = paired_dataset(
    config['annotation_file'],
    transform,
    config['image_root']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate_fn,
    drop_last=True
)


# =========================
# GENERATOR
# =========================
z_dim = 100

generator = Generator(
    input_dim=z_dim,
    num_filters=[[1024], [512], [256], [128]],
    output_dim=3,
    batch_size=args.batch_size,
    first_kernel_size=4
).to(device)

optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

eps = args.eps / 255.0


# =========================
# UNIVERSAL NOISE
# =========================
z = torch.randn(1, z_dim, 1, 1).to(device)


# =========================
# SAVE DIR
# =========================
save_path = os.path.join(
    args.save_dir,
    args.source_model,
    "flickr30k"
)
os.makedirs(save_path, exist_ok=True)

best_loss = float('inf')


# =========================
# TRAINING LOOP
# =========================
print("\n🚀 Starting Training...\n")

for epoch in range(args.epochs):

    total_loss = 0

    for images, _, _, _ in tqdm(train_loader):

        images = images.to(device)

        # -----------------------
        # NOISE + TARGET
        # -----------------------
        z_batch = z.repeat(images.size(0), 1, 1, 1)
        target_batch = target_txt_feat.expand(images.size(0), -1)

        # -----------------------
        # GENERATE PERTURBATION
        # -----------------------
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

        # -----------------------
        # FEATURES
        # -----------------------
        with torch.no_grad():
            clean_feat = model.inference_image(
                normalize(images)
            )['image_feat']
            clean_feat = F.normalize(clean_feat, dim=-1)

        adv_feat = model.inference_image(
            normalize(adv_images)
        )['image_feat']
        adv_feat = F.normalize(adv_feat, dim=-1)

        # -----------------------
        # SIMILARITY
        # -----------------------
        sim_target = torch.sum(adv_feat * target_batch, dim=-1) / 0.03
        sim_clean = torch.sum(adv_feat * clean_feat, dim=-1)

        # -----------------------
        # LOSSES
        # -----------------------
        loss_target = -sim_target.mean()
        loss_away = sim_clean.mean()
        loss_reg = torch.mean(delta ** 2)

        loss = loss_target + 0.5 * loss_away + args.alpha * loss_reg

        # -----------------------
        # BACKPROP
        # -----------------------
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | Sim: {sim_target.mean().item():.4f}")

    # -----------------------
    # SAVE BEST
    # -----------------------
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            delta.detach(),
            os.path.join(save_path, "best_targeted_uap.pth")
        )
        print("✅ Saved BEST UAP")


print("\n✅ Training Complete")