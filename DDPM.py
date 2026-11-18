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
del 10 % (la que crea Limpieza.py). Nunca ve el 90 % restante ni el test.

Uso:
    # 1) Entrenar (semilla = la submuestra del 10 % que quieres usar)
    python DDPM.py entrenar --semilla 0 --pasos 20000

    # 2) Generar imágenes sintéticas (500 de cada clase, con guidance w = 2)
    python DDPM.py generar --semilla 0 --n-normal 500 --n-neumonia 500 --guidance 2.0

Salidas en outputs/ddpm/:
    ddpm_semilla0.pt              pesos del modelo (versión EMA, que es la buena)
    vista_semilla0_pasoXXXX.png   muestras de control durante el entrenamiento
    sinteticas_semilla0_w2.0.npz  imágenes generadas (uint8) y sus etiquetas
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

from comun import DIR_SALIDAS, cargar_submuestra, elegir_dispositivo, fijar_semilla

DIR_DDPM = DIR_SALIDAS / "ddpm"
ETIQUETA_VACIA = 2  # la "clase" que significa "no te digo la clase"


# ==========================================================================
# 1. La red (U-Net pequeña)
# ==========================================================================
def embedding_tiempo(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Convierte el paso de tiempo (un número) en un vector de senos y cosenos."""
    mitad = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class BloqueResidual(nn.Module):
    """Dos convoluciones + un 'atajo' que suma la entrada. Recibe tiempo y clase."""

    def __init__(self, c_in, c_out, dim_emb, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(dim_emb, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.atajo = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]  # aquí entran tiempo y clase
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.atajo(x)


class Atencion(nn.Module):
    """Deja que cada zona de la imagen "mire" a todas las demás (global)."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.salida = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).permute(1, 0, 3, 2).unsqueeze(2).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)  # (b, 1, hw, c)
        a = a.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.salida(a)


class Bajar(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Subir(nn.Module):
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

    def __init__(self, canales=64, mult=(1, 2, 2), n_res=2, dim_emb=256, n_clases=2, dropout=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.n_clases = n_clases
        # tiempo -> vector ; clase -> vector (la última fila es la "clase vacía")
        self.mlp_t = nn.Sequential(nn.Linear(dim_emb, dim_emb), nn.SiLU(), nn.Linear(dim_emb, dim_emb))
        self.emb_clase = nn.Embedding(n_clases + 1, dim_emb)

        self.entrada = nn.Conv2d(1, canales, 3, padding=1)

        # ---- bajada ----
        self.bajada = nn.ModuleList()
        canales_skip = [canales]
        ch = canales
        for i, m in enumerate(mult):
            for _ in range(n_res):
                self.bajada.append(BloqueResidual(ch, canales * m, dim_emb, dropout))
                ch = canales * m
                canales_skip.append(ch)
            if i < len(mult) - 1:
                self.bajada.append(Bajar(ch))
                canales_skip.append(ch)

        # ---- centro ----
        self.medio1 = BloqueResidual(ch, ch, dim_emb, dropout)
        self.atencion = Atencion(ch)
        self.medio2 = BloqueResidual(ch, ch, dim_emb, dropout)

        # ---- subida ----
        self.subida = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                self.subida.append(BloqueResidual(ch + canales_skip.pop(), canales * m, dim_emb, dropout))
                ch = canales * m
            if i > 0:
                self.subida.append(Subir(ch))

        self.salida = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t, y):
        emb = self.mlp_t(embedding_tiempo(t, self.dim_emb)) + self.emb_clase(y)
        h = self.entrada(x)
        skips = [h]
        for capa in self.bajada:
            h = capa(h, emb) if isinstance(capa, BloqueResidual) else capa(h)
            skips.append(h)
        h = self.medio2(self.atencion(self.medio1(h, emb)), emb)
        for capa in self.subida:
            if isinstance(capa, BloqueResidual):
                h = capa(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = capa(h)
        return self.salida(h)


# ==========================================================================
# 2. La difusión (añadir ruido / quitarlo)
# ==========================================================================
class Difusion:
    """Guarda el calendario de ruido y sabe entrenar y muestrear."""

    def __init__(self, T=1000, dispositivo="cpu"):
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
            return v.float().to(dispositivo)

        self.ac, self.sqrt_ac, self.sqrt_1mac = g(ac), g(ac.sqrt()), g((1 - ac).sqrt())
        # coeficientes del paso inverso (forma "predigo la imagen limpia y de ahí deduzco el paso")
        self.var_post = g(betas * (1 - ac_prev) / (1 - ac))
        self.coef_x0 = g(betas * ac_prev.sqrt() / (1 - ac))
        self.coef_xt = g((1 - ac_prev) * alphas.sqrt() / (1 - ac))
        self.dispositivo = dispositivo

    # ---------------- entrenamiento ----------------
    def perdida(self, modelo, x0, y, p_vacia=0.15):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        ruido = torch.randn_like(x0)
        xt = self.sqrt_ac[t][:, None, None, None] * x0 + self.sqrt_1mac[t][:, None, None, None] * ruido
        # con probabilidad p_vacia ocultamos la clase
        y_ent = torch.where(torch.rand(b, device=x0.device) < p_vacia, torch.full_like(y, ETIQUETA_VACIA), y)
        return F.mse_loss(modelo(xt, t, y_ent), ruido)

    # ---------------- generación ----------------
    def _ruido_guiado(self, modelo, x, t, y, w):
        """Mezcla 'con clase' y 'sin clase' (classifier-free guidance)."""
        t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        if w == 0:
            return modelo(x, t_vec, y)
        e = modelo(torch.cat([x, x]), torch.cat([t_vec, t_vec]),
                   torch.cat([y, torch.full_like(y, ETIQUETA_VACIA)]))
        con_clase, sin_clase = e.chunk(2)
        return (1 + w) * con_clase - w * sin_clase

    @torch.no_grad()
    def muestrear(self, modelo, y, w=2.0, pasos=None, tam=28):
        """
        Genera una imagen por cada etiqueta de 'y'.
        pasos=None (o >= T): DDPM clásico con los T pasos (lento, es el "oficial").
        pasos=50, 100...: versión rápida (DDIM) saltando pasos.
        Devuelve imágenes en [-1, 1].
        """
        modelo.eval()
        b = y.shape[0]
        x = torch.randn(b, 1, tam, tam, device=y.device)
        rapido = pasos is not None and pasos < self.T
        secuencia = (torch.linspace(self.T - 1, 0, pasos).round().long().tolist()
                     if rapido else list(range(self.T - 1, -1, -1)))

        for i, t in enumerate(secuencia):
            eps = self._ruido_guiado(modelo, x, t, y, w)
            # imagen limpia que "cree" la red; la recortamos a [-1,1] para que no se dispare con w alto
            x0 = ((x - self.sqrt_1mac[t] * eps) / self.sqrt_ac[t]).clamp(-1, 1)
            if rapido:
                t_prev = secuencia[i + 1] if i + 1 < len(secuencia) else -1
                ac_prev = self.ac[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
                eps = (x - self.sqrt_ac[t] * x0) / self.sqrt_1mac[t]
                x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
            else:
                media = self.coef_x0[t] * x0 + self.coef_xt[t] * x
                x = media + self.var_post[t].sqrt() * torch.randn_like(x) if t > 0 else media
        return x.clamp(-1, 1)


# ==========================================================================
# 3. EMA (media móvil de los pesos: da imágenes más limpias)
# ==========================================================================
class EMA:
    def __init__(self, modelo, decay=0.999):
        self.decay = decay
        self.copia = copy.deepcopy(modelo).eval()
        for p in self.copia.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def actualizar(self, modelo):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # al principio se adapta más rápido
        for pe, p in zip(self.copia.parameters(), modelo.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)


# ==========================================================================
# 4. Utilidades
# ==========================================================================
def a_uint8(x: torch.Tensor) -> np.ndarray:
    """De [-1,1] a imágenes uint8 (N, H, W)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()


def guardar_rejilla(imgs: np.ndarray, etiquetas: np.ndarray, ruta, titulo=""):
    """Dibuja una fila por clase con las imágenes generadas."""
    clases = sorted(set(etiquetas.tolist()))
    n = max((etiquetas == c).sum() for c in clases)
    fig, ejes = plt.subplots(len(clases), n, figsize=(n * 1.1, len(clases) * 1.3), squeeze=False)
    nombres = ["normal", "neumonía"]
    for f, c in enumerate(clases):
        sel = imgs[etiquetas == c]
        for j in range(n):
            ejes[f, j].axis("off")
            if j < len(sel):
                ejes[f, j].imshow(sel[j], cmap="gray", vmin=0, vmax=255)
        ejes[f, 0].set_title(nombres[c], fontsize=8, loc="left")
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def construir_modelo(cfg: dict, dispositivo) -> UNet:
    return UNet(canales=cfg["canales"], mult=tuple(cfg["mult"]), n_res=cfg["n_res"],
                dim_emb=cfg["dim_emb"], dropout=cfg["dropout"]).to(dispositivo)


# ==========================================================================
# 5. Comandos: entrenar y generar
# ==========================================================================
def entrenar(args):
    dev = elegir_dispositivo()
    fijar_semilla(args.semilla)
    DIR_DDPM.mkdir(parents=True, exist_ok=True)

    # Solo la submuestra del 10 % (el resto del train y el test NO se cargan aquí)
    x10, y10, _, _ = cargar_submuestra(args.semilla)
    if x10.shape[1] % 4 != 0:
        raise ValueError("El tamaño de imagen debe ser múltiplo de 4 para esta U-Net")
    tam = x10.shape[1]
    X = torch.from_numpy(x10).float().div(127.5).sub(1).unsqueeze(1).to(dev)  # a [-1, 1]
    Y = torch.from_numpy(y10).long().to(dev)
    print(f"Dispositivo: {dev} | imágenes de entrenamiento: {len(X)} "
          f"(normal {(y10 == 0).sum()}, neumonía {(y10 == 1).sum()})")

    cfg = dict(canales=args.canales, mult=[1, 2, 2], n_res=2, dim_emb=256, dropout=args.dropout, T=args.T, tam=tam)
    modelo = construir_modelo(cfg, dev)
    print(f"Parámetros de la red: {sum(p.numel() for p in modelo.parameters()) / 1e6:.2f} M")
    ema = EMA(modelo, args.ema)
    dif = Difusion(args.T, dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    # Mezclamos el orden en cada vuelta (época) para ver todas las imágenes por igual
    orden, pos = torch.randperm(len(X), device=dev), 0
    t0, acumulada = time.time(), 0.0
    for paso in range(1, args.pasos + 1):
        modelo.train()
        if pos + args.batch > len(X):
            orden, pos = torch.randperm(len(X), device=dev), 0
        idx = orden[pos:pos + args.batch]
        pos += args.batch

        loss = dif.perdida(modelo, X[idx], Y[idx], args.p_vacia)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        ema.actualizar(modelo)
        acumulada += loss.item()

        if paso % args.log_cada == 0:
            print(f"paso {paso:6d}/{args.pasos} | pérdida {acumulada / args.log_cada:.4f} | {time.time() - t0:.0f}s")
            acumulada = 0.0

        if paso % args.vista_cada == 0 or paso == args.pasos:
            y_v = torch.tensor([0] * 8 + [1] * 8, device=dev)
            muestras = dif.muestrear(ema.copia, y_v, w=2.0, pasos=50, tam=tam)
            guardar_rejilla(a_uint8(muestras), y_v.cpu().numpy(), DIR_DDPM / f"vista_semilla{args.semilla}_paso{paso}.png",
                            f"paso {paso} (EMA, w=2, DDIM 50 pasos)")
            ruta = DIR_DDPM / f"ddpm_semilla{args.semilla}.pt"
            torch.save({"modelo_ema": ema.copia.state_dict(), "cfg": cfg, "semilla": args.semilla, "paso": paso}, ruta)

    print(f"Modelo guardado en {ruta}")


def generar(args):
    dev = elegir_dispositivo()
    ruta = DIR_DDPM / f"ddpm_semilla{args.semilla}.pt"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Entrena antes con:  python DDPM.py entrenar --semilla {args.semilla}")
    ck = torch.load(ruta, map_location=dev)
    cfg = ck["cfg"]
    modelo = construir_modelo(cfg, dev)
    modelo.load_state_dict(ck["modelo_ema"])
    dif = Difusion(cfg["T"], dev)
    fijar_semilla(args.semilla_gen)

    etiquetas = np.array([0] * args.n_normal + [1] * args.n_neumonia)
    imgs = []
    t0 = time.time()
    for ini in range(0, len(etiquetas), args.batch):
        y = torch.from_numpy(etiquetas[ini:ini + args.batch]).long().to(dev)
        imgs.append(a_uint8(dif.muestrear(modelo, y, w=args.guidance, pasos=args.pasos_muestreo, tam=cfg["tam"])))
        print(f"generadas {min(ini + args.batch, len(etiquetas))}/{len(etiquetas)} ({time.time() - t0:.0f}s)")
    imgs = np.concatenate(imgs)

    salida = DIR_DDPM / f"sinteticas_semilla{args.semilla}_w{args.guidance}.npz"
    np.savez_compressed(salida, images=imgs, labels=etiquetas, guidance=args.guidance, semilla=args.semilla)
    # una rejilla de control con 10 de cada clase
    sel = np.concatenate([np.where(etiquetas == c)[0][:10] for c in (0, 1) if (etiquetas == c).any()])
    guardar_rejilla(imgs[sel], etiquetas[sel], salida.with_suffix(".png"), f"sintéticas, w={args.guidance}")
    print(f"Guardado en {salida}")


def main():
    ap = argparse.ArgumentParser(description="DDPM condicional por clase con classifier-free guidance")
    sub = ap.add_subparsers(dest="comando", required=True)

    e = sub.add_parser("entrenar", help="entrena el DDPM con la submuestra del 10 %%")
    e.add_argument("--semilla", type=int, default=0, help="qué submuestra del 10 %% usar (y semilla del entrenamiento)")
    e.add_argument("--pasos", type=int, default=20000, help="pasos de entrenamiento")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--lr", type=float, default=2e-4)
    e.add_argument("--canales", type=int, default=64, help="anchura de la red (32 si vas justo de memoria/tiempo)")
    e.add_argument("--dropout", type=float, default=0.1)
    e.add_argument("--p-vacia", type=float, default=0.15, help="prob. de ocultar la clase durante el entrenamiento")
    e.add_argument("--ema", type=float, default=0.999)
    e.add_argument("--T", type=int, default=1000, help="pasos de difusión")
    e.add_argument("--log-cada", type=int, default=250)
    e.add_argument("--vista-cada", type=int, default=2000, help="cada cuántos pasos guardar imágenes de control")
    e.set_defaults(func=entrenar)

    g = sub.add_parser("generar", help="genera imágenes sintéticas con un modelo ya entrenado")
    g.add_argument("--semilla", type=int, default=0, help="qué modelo cargar (el de esa submuestra)")
    g.add_argument("--semilla-gen", type=int, default=1234, help="semilla del ruido al generar")
    g.add_argument("--n-normal", type=int, default=500)
    g.add_argument("--n-neumonia", type=int, default=500)
    g.add_argument("--guidance", type=float, default=2.0, help="escala w de classifier-free guidance (0 = sin guidance)")
    g.add_argument("--pasos-muestreo", type=int, default=None, help="vacío = DDPM completo (T pasos); 100 = rápido (DDIM)")
    g.add_argument("--batch", type=int, default=250)
    g.set_defaults(func=generar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
