import torch
import torch.nn.functional as F
import torch.optim as optim


class ImageAttackerTargeted:
    """
    Targeted Universal Adversarial Attacker

    Learns perturbation δ such that:
    image + δ → aligns with target text
    """

    def __init__(
        self,
        generator,
        normalization,
        eps,
        device='cuda',
        lr=1e-4,
        alpha=0.01,         # 🔥 reduced
        temperature=0.03    # 🔥 stronger pull
    ):
        self.generator = generator.to(device)
        self.normalization = normalization
        self.eps = eps
        self.device = device
        self.alpha = alpha
        self.temperature = temperature

        self.optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr
        )

    def attack(self, model, images, z, target_txt_feat):
        """
        images: (B, 3, H, W)
        z: (1, z_dim, 1, 1)
        target_txt_feat: (1, D)
        """

        images = images.to(self.device)
        b = images.size(0)

        # =========================
        # UNIVERSAL NOISE
        # =========================
        z_batch = z.repeat(b, 1, 1, 1)
        target_batch = target_txt_feat.expand(b, -1)

        # =========================
        # GENERATE PERTURBATION
        # =========================
        delta = self.generator(z_batch, target_batch)

        # 🔥 Fix 1: Match image resolution
        delta = F.interpolate(
            delta,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 🔥 Fix 2: Scale to epsilon
        delta = delta * self.eps

        # 🔥 Clamp perturbation
        delta = torch.clamp(delta, -self.eps, self.eps)

        # =========================
        # APPLY ATTACK
        # =========================
        adv_images = torch.clamp(images + delta, 0, 1)

        # =========================
        # FEATURE EXTRACTION
        # =========================
        with torch.no_grad():
            clean_feat = model.inference_image(
                self.normalization(images)
            )['image_feat']
            clean_feat = F.normalize(clean_feat, dim=-1)

        adv_feat = model.inference_image(
            self.normalization(adv_images)
        )['image_feat']
        adv_feat = F.normalize(adv_feat, dim=-1)

        target_norm = F.normalize(target_batch, dim=-1)

        # =========================
        # SIMILARITY
        # =========================
        sim_target = torch.sum(adv_feat * target_norm, dim=-1) / self.temperature
        sim_clean  = torch.sum(adv_feat * clean_feat, dim=-1)

        # =========================
        # LOSSES
        # =========================
        loss_target = -sim_target.mean()      # pull to target
        loss_away   = sim_clean.mean()        # push away from original
        loss_reg    = torch.mean(delta ** 2)  # smoothness

        loss = loss_target + 0.5 * loss_away + self.alpha * loss_reg

        # =========================
        # BACKPROP
        # =========================
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss, delta