"""
DDPM.py
=======
Fase 4 del flujo de trabajo: modelo de difusión (DDPM) condicionado por clase,
con classifier-free guidance (CFG).

Idea general, sin tecnicismos:
  - Durante el ENTRENAMIENTO cogemos una radiografía real, le echamos ruido
    (más o menos según un "paso de tiempo" t) y le pedimos a una red (U-Net) que
    adivine qué ruido le hemos echado. La red también recibe la clase
    (normal / neumonía). De vez en cuando (15 % de las veces) le ocultamos la
    clase, para que aprenda también a generar "sin saber la clase".
  - Para GENERAR partimos de ruido puro y vamos quitándolo poco a poco. En cada
    paso mezclamos lo que la red predice sabiendo la clase y sin saberla:
        ruido_final = (1 + w) * con_clase - w * sin_clase
    Con w = 0 sale más variedad; con w grande salen imágenes más "típicas" de la
    clase pero menos variadas (ojo con esto para el análisis de atajos).

IMPORTANTE (protocolo de escasez): el generador SOLO se entrena con la submuestra
del 10 % (la que crea Cleaning.py). Nunca ve el 90 % restante ni el test.

Uso:
    # 1) Entrenar (seed = la submuestra del 10 % que quieres usar)
    python DDPM.py train --seed 0 --steps 20000

    # 2) Generar imágenes sintéticas (500 de cada clase, con guidance w = 2)
    python DDPM.py generate --seed 0 --n-normal 500 --n-pneumonia 500 --guidance 2.0

Salidas en outputs/ddpm/:
    ddpm_seed0.pt                 pesos del modelo (versión EMA, que es la buena)
    preview_seed0_stepXXXX.png    muestras de control durante el entrenamiento
    synthetic_seed0_w2.0.npz      imágenes generadas (uint8) y sus etiquetas
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUTPUT_DIR, load_subsample, pick_device, set_seed

DDPM_DIR = OUTPUT_DIR / "ddpm"
NULL_LABEL = 2  # la "clase" que significa "no te digo la clase"


# ==========================================================================
# 1. La red (U-Net pequeña)
# ==========================================================================
def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Convierte el paso de tiempo (un número) en un vector de senos y cosenos."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Dos convoluciones + un 'atajo' que suma la entrada. Recibe tiempo y clase."""

    def __init__(self, c_in, c_out, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]  # aquí entran tiempo y clase
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class Attention(nn.Module):
    """Deja que cada zona de la imagen "mire" a todas las demás (global)."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).permute(1, 0, 3, 2).unsqueeze(2).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)  # (b, 1, hw, c)
        a = a.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(a)


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    U-Net condicionada por tiempo y clase.
    Baja de resolución (28 -> 14 -> 7), procesa en el centro con atención y
    vuelve a subir reutilizando los "recuerdos" (skips) de la bajada.
    """

    def __init__(self, channels=64, mult=(1, 2, 2), n_res=2, emb_dim=256, n_classes=2, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        # tiempo -> vector ; clase -> vector (la última fila es la "clase vacía")
        self.mlp_t = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(n_classes + 1, emb_dim)

        self.inp = nn.Conv2d(1, channels, 3, padding=1)

        # ---- bajada ----
        self.down = nn.ModuleList()
        skip_channels = [channels]
        ch = channels
        for i, m in enumerate(mult):
            for _ in range(n_res):
                self.down.append(ResidualBlock(ch, channels * m, emb_dim, dropout))
                ch = channels * m
                skip_channels.append(ch)
            if i < len(mult) - 1:
                self.down.append(Down(ch))
                skip_channels.append(ch)

        # ---- centro ----
        self.mid1 = ResidualBlock(ch, ch, emb_dim, dropout)
        self.attn = Attention(ch)
        self.mid2 = ResidualBlock(ch, ch, emb_dim, dropout)

        # ---- subida ----
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                self.up.append(ResidualBlock(ch + skip_channels.pop(), channels * m, emb_dim, dropout))
                ch = channels * m
            if i > 0:
                self.up.append(Up(ch))

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t, y):
        emb = self.mlp_t(time_embedding(t, self.emb_dim)) + self.class_emb(y)
        h = self.inp(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResidualBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, ResidualBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.out(h)


# ==========================================================================
# 2. La difusión (añadir ruido / quitarlo)
# ==========================================================================
class Diffusion:
    """Guarda el calendario de ruido y sabe entrenar y muestrear."""

    def __init__(self, T=1000, device="cpu"):
        self.T = T
        # calendario "coseno": va añadiendo ruido de forma suave
        s = 0.008
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(0, 0.999)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = torch.cat([torch.ones(1, dtype=torch.float64), ac[:-1]])

        def g(v):  # a float32 y al dispositivo
            return v.float().to(device)

        self.ac, self.sqrt_ac, self.sqrt_1mac = g(ac), g(ac.sqrt()), g((1 - ac).sqrt())
        # coeficientes del paso inverso (forma "predigo la imagen limpia y de ahí deduzco el paso")
        self.post_var = g(betas * (1 - ac_prev) / (1 - ac))
        self.coef_x0 = g(betas * ac_prev.sqrt() / (1 - ac))
        self.coef_xt = g((1 - ac_prev) * alphas.sqrt() / (1 - ac))
        self.device = device

    # ---------------- entrenamiento ----------------
    def loss(self, model, x0, y, p_null=0.15):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.sqrt_ac[t][:, None, None, None] * x0 + self.sqrt_1mac[t][:, None, None, None] * noise
        # con probabilidad p_null ocultamos la clase
        y_train = torch.where(torch.rand(b, device=x0.device) < p_null, torch.full_like(y, NULL_LABEL), y)
        return F.mse_loss(model(xt, t, y_train), noise)

    # ---------------- generación ----------------
    def _guided_noise(self, model, x, t, y, w):
        """Mezcla 'con clase' y 'sin clase' (classifier-free guidance)."""
        t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        if w == 0:
            return model(x, t_vec, y)
        e = model(torch.cat([x, x]), torch.cat([t_vec, t_vec]),
                  torch.cat([y, torch.full_like(y, NULL_LABEL)]))
        with_class, without_class = e.chunk(2)
        return (1 + w) * with_class - w * without_class

    @torch.no_grad()
    def sample(self, model, y, w=2.0, steps=None, size=28):
        """
        Genera una imagen por cada etiqueta de 'y'.
        steps=None (o >= T): DDPM clásico con los T pasos (lento, es el "oficial").
        steps=50, 100...: versión rápida (DDIM) saltando pasos.
        Devuelve imágenes en [-1, 1].
        """
        model.eval()
        b = y.shape[0]
        x = torch.randn(b, 1, size, size, device=y.device)
        fast = steps is not None and steps < self.T
        sequence = (torch.linspace(self.T - 1, 0, steps).round().long().tolist()
                    if fast else list(range(self.T - 1, -1, -1)))

        for i, t in enumerate(sequence):
            eps = self._guided_noise(model, x, t, y, w)
            # imagen limpia que "cree" la red; la recortamos a [-1,1] para que no se dispare con w alto
            x0 = ((x - self.sqrt_1mac[t] * eps) / self.sqrt_ac[t]).clamp(-1, 1)
            if fast:
                t_prev = sequence[i + 1] if i + 1 < len(sequence) else -1
                ac_prev = self.ac[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
                eps = (x - self.sqrt_ac[t] * x0) / self.sqrt_1mac[t]
                x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
            else:
                mean = self.coef_x0[t] * x0 + self.coef_xt[t] * x
                x = mean + self.post_var[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        return x.clamp(-1, 1)


# ==========================================================================
# 3. EMA (media móvil de los pesos: da imágenes más limpias)
# ==========================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.copy = copy.deepcopy(model).eval()
        for p in self.copy.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # al principio se adapta más rápido
        for pe, p in zip(self.copy.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)


# ==========================================================================
# 4. Utilidades
# ==========================================================================
def to_uint8(x: torch.Tensor) -> np.ndarray:
    """De [-1,1] a imágenes uint8 (N, H, W)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()


def save_grid(imgs: np.ndarray, labels: np.ndarray, path, title=""):
    """Dibuja una fila por clase con las imágenes generadas."""
    classes = sorted(set(labels.tolist()))
    n = max((labels == c).sum() for c in classes)
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 1.1, len(classes) * 1.3), squeeze=False)
    names = ["normal", "pneumonia"]
    for r, c in enumerate(classes):
        sel = imgs[labels == c]
        for j in range(n):
            axes[r, j].axis("off")
            if j < len(sel):
                axes[r, j].imshow(sel[j], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(names[c], fontsize=8, loc="left")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_model(cfg: dict, device) -> UNet:
    return UNet(channels=cfg["channels"], mult=tuple(cfg["mult"]), n_res=cfg["n_res"],
                emb_dim=cfg["emb_dim"], dropout=cfg["dropout"]).to(device)


# ==========================================================================
# 5. Comandos: train y generate
# ==========================================================================
def train(args):
    dev = pick_device()
    set_seed(args.seed)
    DDPM_DIR.mkdir(parents=True, exist_ok=True)

    # Solo la submuestra del 10 % (el resto del train y el test NO se cargan aquí)
    x10, y10, _, _ = load_subsample(args.seed)
    if x10.shape[1] % 4 != 0:
        raise ValueError("Image size must be a multiple of 4 for this U-Net")
    size = x10.shape[1]
    X = torch.from_numpy(x10).float().div(127.5).sub(1).unsqueeze(1).to(dev)  # a [-1, 1]
    Y = torch.from_numpy(y10).long().to(dev)
    print(f"Device: {dev} | training images: {len(X)} "
          f"(normal {(y10 == 0).sum()}, pneumonia {(y10 == 1).sum()})")

    cfg = dict(channels=args.channels, mult=[1, 2, 2], n_res=2, emb_dim=256, dropout=args.dropout, T=args.T, size=size)
    model = build_model(cfg, dev)
    print(f"Network parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    ema = EMA(model, args.ema)
    diff = Diffusion(args.T, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Mezclamos el orden en cada vuelta (época) para ver todas las imágenes por igual
    order, pos = torch.randperm(len(X), device=dev), 0
    t0, accumulated = time.time(), 0.0
    for step in range(1, args.steps + 1):
        model.train()
        if pos + args.batch > len(X):
            order, pos = torch.randperm(len(X), device=dev), 0
        idx = order[pos:pos + args.batch]
        pos += args.batch

        loss = diff.loss(model, X[idx], Y[idx], args.p_null)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        accumulated += loss.item()

        if step % args.log_every == 0:
            print(f"step {step:6d}/{args.steps} | loss {accumulated / args.log_every:.4f} | {time.time() - t0:.0f}s")
            accumulated = 0.0

        if step % args.preview_every == 0 or step == args.steps:
            y_v = torch.tensor([0] * 8 + [1] * 8, device=dev)
            samples = diff.sample(ema.copy, y_v, w=2.0, steps=50, size=size)
            save_grid(to_uint8(samples), y_v.cpu().numpy(), DDPM_DIR / f"preview_seed{args.seed}_step{step}.png",
                      f"step {step} (EMA, w=2, DDIM 50 steps)")
            path = DDPM_DIR / f"ddpm_seed{args.seed}.pt"
            torch.save({"ema_model": ema.copy.state_dict(), "cfg": cfg, "seed": args.seed, "step": step}, path)

    print(f"Model saved to {path}")


def generate(args):
    dev = pick_device()
    path = DDPM_DIR / f"ddpm_seed{args.seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Train first with:  python DDPM.py train --seed {args.seed}")
    ck = torch.load(path, map_location=dev)
    cfg = ck["cfg"]
    model = build_model(cfg, dev)
    model.load_state_dict(ck["ema_model"])
    diff = Diffusion(cfg["T"], dev)
    set_seed(args.gen_seed)

    labels = np.array([0] * args.n_normal + [1] * args.n_pneumonia)
    imgs = []
    t0 = time.time()
    for start in range(0, len(labels), args.batch):
        y = torch.from_numpy(labels[start:start + args.batch]).long().to(dev)
        imgs.append(to_uint8(diff.sample(model, y, w=args.guidance, steps=args.sampling_steps, size=cfg["size"])))
        print(f"generated {min(start + args.batch, len(labels))}/{len(labels)} ({time.time() - t0:.0f}s)")
    imgs = np.concatenate(imgs)

    out = DDPM_DIR / f"synthetic_seed{args.seed}_w{args.guidance}.npz"
    np.savez_compressed(out, images=imgs, labels=labels, guidance=args.guidance, seed=args.seed)
    # una rejilla de control con 10 de cada clase
    sel = np.concatenate([np.where(labels == c)[0][:10] for c in (0, 1) if (labels == c).any()])
    save_grid(imgs[sel], labels[sel], out.with_suffix(".png"), f"synthetic, w={args.guidance}")
    print(f"Saved to {out}")


def main():
    ap = argparse.ArgumentParser(description="Class-conditional DDPM with classifier-free guidance")
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("train", help="trains the DDPM on the 10%% subsample")
    e.add_argument("--seed", type=int, default=0, help="which 10%% subsample to use (and training seed)")
    e.add_argument("--steps", type=int, default=20000, help="training steps")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--lr", type=float, default=2e-4)
    e.add_argument("--channels", type=int, default=64, help="network width (32 if short on memory/time)")
    e.add_argument("--dropout", type=float, default=0.1)
    e.add_argument("--p-null", type=float, default=0.15, help="prob. of hiding the class during training")
    e.add_argument("--ema", type=float, default=0.999)
    e.add_argument("--T", type=int, default=1000, help="diffusion steps")
    e.add_argument("--log-every", type=int, default=250)
    e.add_argument("--preview-every", type=int, default=2000, help="every how many steps to save preview images")
    e.set_defaults(func=train)

    g = sub.add_parser("generate", help="generates synthetic images with an already trained model")
    g.add_argument("--seed", type=int, default=0, help="which model to load (the one for that subsample)")
    g.add_argument("--gen-seed", type=int, default=1234, help="noise seed when generating")
    g.add_argument("--n-normal", type=int, default=500)
    g.add_argument("--n-pneumonia", type=int, default=500)
    g.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance scale w (0 = no guidance)")
    g.add_argument("--sampling-steps", type=int, default=None, help="empty = full DDPM (T steps); 100 = fast (DDIM)")
    g.add_argument("--batch", type=int, default=250)
    g.set_defaults(func=generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
