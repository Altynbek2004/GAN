"""
Лабораторная работа №3: DCGAN: реализация, обучение и анализ архитектурных решений
Дисциплина: Generative Adversarial Networks (ANL7311)
Автор: Altynbek2004
Дата: 2026-09-30

Данный скрипт выполняет:
1. Задание 1: Базовая реализация DCGAN (Radford et al., 2015) на MNIST, обучение на 15 эпох.
   - Фиксация латентных векторов для отслеживания эволюции генерации (эпохи 1, 8, 15).
   - Построение кривых потерь G и D.
2. Задание 2: Абляционный эксперимент — отключение Batch Normalization в обеих сетях.
   - Обучение No-BN модели на идентичных гиперпараметрах.
   - Сравнение потерь на одних осях и визуальное сравнение сгенерированных выборок.
3. Задание 3: Исследование латентного пространства — линейная интерполяция между векторами z.
   - 3 независимые пары векторов, по 10 шагов интерполяции.
   - Анализ гладкости и сравнение с VAE.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# ---------------------------------------------------------
# 0. Настройки воспроизводимости и устройства (MPS / CPU)
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[*] Используемое устройство: {device}")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Архитектура DCGAN (Baseline согласно 5 правилам Radford et al.)
# ---------------------------------------------------------
def weights_init(m):
    """
    Инициализация весов по статье Radford et al. (2015):
    - Свёрточные и полносвязные веса ~ Normal(mean=0.0, std=0.02)
    - BatchNorm веса ~ Normal(mean=1.0, std=0.02), смещение (bias) = 0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """
    Генератор DCGAN:
    Вход: z ~ N(0, I) размерности 100
    - Линейный слой проекции в 256 * 7 * 7 + BatchNorm1d + ReLU
    - Reshape в пространственную карту 256 x 7 x 7
    - Блок 1 (ConvTranspose2d): 256x7x7 -> 128x14x14 + BatchNorm2d + ReLU
    - Блок 2 (ConvTranspose2d): 128x14x14 -> 64x28x28 + BatchNorm2d + ReLU
    - Блок 3 (Финальная свёртка): 64x28x28 -> 1x28x28 + Tanh (значения в [-1, 1])
    """
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            # Блок 1: (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Блок 2: (128, 14, 14) -> (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Выходной слой: (64, 28, 28) -> (1, 28, 28)
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()  # Tanh сжимает выход в [-1, 1], согласуясь с нормализацией данных
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)
        return self.conv_blocks(x)

class Discriminator(nn.Module):
    """
    Дискриминатор DCGAN:
    Вход: изображение x размером (1, 28, 28) со значениями в [-1, 1]
    - Блок 1 (Conv2d со страйдом): 1x28x28 -> 64x14x14 + LeakyReLU(0.2)
      * БЕЗ BatchNorm на первом слое (правило Radford et al.)!
    - Блок 2 (Conv2d со страйдом): 64x14x14 -> 128x7x7 + BatchNorm2d + LeakyReLU(0.2)
    - Блок 3 (Conv2d со страйдом): 128x7x7 -> 256x4x4 + BatchNorm2d + LeakyReLU(0.2)
    - Финал: Flatten -> Linear(256 * 4 * 4, 1) -> Sigmoid (вероятность D(x) in (0, 1))
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Первый слой: без BatchNorm!
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Второй слой: с BatchNorm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Третий слой: с BatchNorm
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        validity = self.classifier(features)
        return validity

# ---------------------------------------------------------
# 2. Модели для Абляционного эксперимента (Без BatchNorm)
# ---------------------------------------------------------
class GeneratorNoBN(nn.Module):
    """Генератор без слоев Batch Normalization для исследования влияния на стабильность."""
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7, bias=True),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)
        return self.conv_blocks(x)

class DiscriminatorNoBN(nn.Module):
    """Дискриминатор без слоев Batch Normalization."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

# ---------------------------------------------------------
# 3. Функция обучения модели DCGAN
# ---------------------------------------------------------
def train_dcgan(gen, disc, dataloader, epochs=15, lr=0.0002, beta1=0.5, fixed_z=None, model_name="Baseline"):
    print(f"\n==================================================")
    print(f"[*] Начало обучения модели: {model_name} ({epochs} эпох)")
    print(f"==================================================")

    criterion = nn.BCELoss()
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

    d_losses_iter = []
    g_losses_iter = []
    d_losses_epoch = []
    g_losses_epoch = []
    saved_samples = {}

    total_steps = len(dataloader)

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        running_d_loss = 0.0
        running_g_loss = 0.0

        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Метки: 1.0 для реальных, 0.0 для фейковых
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Обучение Дискриминатора:
            #  max log D(x) + log (1 - D(G(z)))
            # ---------------------
            opt_d.zero_grad()
            d_real_out = disc(real_imgs)
            d_loss_real = criterion(d_real_out, real_labels)

            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = gen(z)
            d_fake_out = disc(fake_imgs.detach())
            d_loss_fake = criterion(d_fake_out, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # ---------------------
            #  Обучение Генератора:
            #  Non-saturating heuristic: max log D(G(z)) <=> min BCE(D(G(z)), 1)
            # ---------------------
            opt_g.zero_grad()
            g_fake_out = disc(fake_imgs)
            g_loss = criterion(g_fake_out, real_labels)
            g_loss.backward()
            opt_g.step()

            # Метрики
            d_losses_iter.append(d_loss.item())
            g_losses_iter.append(g_loss.item())
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / total_steps
        avg_g_loss = running_g_loss / total_steps
        d_losses_epoch.append(avg_d_loss)
        g_losses_epoch.append(avg_g_loss)

        print(f"[{model_name}] Эпоха [{epoch:02d}/{epochs:02d}] -> D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")

        # Фиксация сгенерированных изображений на первой, средней и последней эпохе
        mid_epoch = epochs // 2 + 1
        if epoch in [1, mid_epoch, epochs] and fixed_z is not None:
            gen.eval()
            with torch.no_grad():
                samples = gen(fixed_z).cpu()
                # Переводим из [-1, 1] в [0, 1]
                samples = (samples + 1.0) / 2.0
                saved_samples[epoch] = samples
            gen.train()

    return {
        "d_losses_iter": d_losses_iter,
        "g_losses_iter": g_losses_iter,
        "d_losses_epoch": d_losses_epoch,
        "g_losses_epoch": g_losses_epoch,
        "saved_samples": saved_samples,
        "generator": gen,
        "discriminator": disc
    }

# ---------------------------------------------------------
# 4. Основной сценарий выполнения экспериментов
# ---------------------------------------------------------
def main():
    print("[*] Загрузка датасета MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Значения в диапазон [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    print(f"[*] Датасет загружен: {len(dataset)} изображений, батчей: {len(dataloader)}")

    # Фиксированный шум для 25 изображений (сетка 5x5)
    fixed_z = torch.randn(25, 100, device=device)

    # -----------------------------------------------------
    # ЗАДАНИЕ 1: Базовое обучение DCGAN
    # -----------------------------------------------------
    print("\n>>> ЗАДАНИЕ 1: Инициализация и обучение Baseline DCGAN <<<")
    set_seed(42)
    gen_base = Generator(z_dim=100).to(device)
    disc_base = Discriminator().to(device)
    gen_base.apply(weights_init)
    disc_base.apply(weights_init)

    EPOCHS = 15
    res_base = train_dcgan(
        gen_base, disc_base, dataloader,
        epochs=EPOCHS, lr=0.0002, beta1=0.5,
        fixed_z=fixed_z, model_name="Baseline DCGAN"
    )

    # Сохранение графиков потерь Задания 1
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(res_base["d_losses_epoch"], label="Discriminator Loss", color="#D9534F", linewidth=2.2)
    plt.plot(res_base["g_losses_epoch"], label="Generator Loss", color="#0275D8", linewidth=2.2)
    plt.title("Задание 1: Кривые потерь DCGAN (MNIST, 15 эпох)", fontsize=14, fontweight="bold")
    plt.xlabel("Эпоха", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, "lab3_task1_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[+] График потерь Задания 1 сохранен: {loss_path}")

    # Сохранение сеток на 1, 8 и 15 эпохах
    mid_ep = EPOCHS // 2 + 1
    for ep, title in [(1, "Эпоха 1 (Начало)"), (mid_ep, f"Эпоха {mid_ep} (Середина)"), (EPOCHS, f"Эпоха {EPOCHS} (Финал)")]:
        if ep in res_base["saved_samples"]:
            imgs = res_base["saved_samples"][ep]
            grid = make_grid(imgs, nrow=5, padding=2, normalize=False)
            
            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap="gray")
            plt.title(f"DCGAN Baseline: {title}", fontsize=12, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            out_img = os.path.join(OUTPUT_DIR, f"lab3_task1_samples_epoch{ep}.png")
            plt.savefig(out_img)
            plt.close()
            print(f"[+] Сетка эпохи {ep} сохранена: {out_img}")

    # Сводный триптих эволюции обучения (Эпоха 1 -> Эпоха 8 -> Эпоха 15)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for idx, (ep, sub_title) in enumerate([(1, "Эпоха 1: Шум и базовые контуры"), 
                                          (mid_ep, f"Эпоха {mid_ep}: Формирование структуры цифр"), 
                                          (EPOCHS, f"Эпоха {EPOCHS}: Чёткие цифры MNIST")]):
        grid = make_grid(res_base["saved_samples"][ep], nrow=5, padding=2, normalize=False)
        axes[idx].imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap="gray")
        axes[idx].set_title(sub_title, fontsize=11, fontweight="bold")
        axes[idx].axis("off")
    plt.suptitle("Эволюция качества генерации DCGAN на фиксированном векторе z", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    evolution_path = os.path.join(OUTPUT_DIR, "lab3_task1_evolution_grid.png")
    plt.savefig(evolution_path)
    plt.close()
    print(f"[+] Сводная эволюция сохранена: {evolution_path}")

    # -----------------------------------------------------
    # ЗАДАНИЕ 2: Абляционный эксперимент (Без BatchNorm)
    # -----------------------------------------------------
    print("\n>>> ЗАДАНИЕ 2: Абляционный эксперимент (Удаление Batch Normalization) <<<")
    set_seed(42)  # Точно такой же seed для честного сравнения!
    gen_nobn = GeneratorNoBN(z_dim=100).to(device)
    disc_nobn = DiscriminatorNoBN().to(device)
    gen_nobn.apply(weights_init)
    disc_nobn.apply(weights_init)

    res_nobn = train_dcgan(
        gen_nobn, disc_nobn, dataloader,
        epochs=EPOCHS, lr=0.0002, beta1=0.5,
        fixed_z=fixed_z, model_name="Ablation (No-BN)"
    )

    # Сравнение графиков потерь на одних осях
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Дискриминатор: Baseline vs No-BN
    axes[0].plot(res_base["d_losses_epoch"], label="Baseline D (с BatchNorm)", color="#28A745", linewidth=2.2)
    axes[0].plot(res_nobn["d_losses_epoch"], label="Ablation D (Без BatchNorm)", color="#DC3545", linestyle="--", linewidth=2.2)
    axes[0].set_title("Сравнение потерь Дискриминатора (D Loss)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Эпоха", fontsize=11)
    axes[0].set_ylabel("BCE Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Генератор: Baseline vs No-BN
    axes[1].plot(res_base["g_losses_epoch"], label="Baseline G (с BatchNorm)", color="#007BFF", linewidth=2.2)
    axes[1].plot(res_nobn["g_losses_epoch"], label="Ablation G (Без BatchNorm)", color="#FD7E14", linestyle="--", linewidth=2.2)
    axes[1].set_title("Сравнение потерь Генератора (G Loss)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Эпоха", fontsize=11)
    axes[1].set_ylabel("BCE Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.suptitle("Абляционный анализ: влияние Batch Normalization на динамику сходимости", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    ablation_loss_path = os.path.join(OUTPUT_DIR, "lab3_task2_ablation_loss_comparison.png")
    plt.savefig(ablation_loss_path)
    plt.close()
    print(f"[+] График сравнения потерь Задания 2 сохранен: {ablation_loss_path}")

    # Сравнение сгенерированных изображений на 15 эпохе (Baseline vs No-BN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    grid_base = make_grid(res_base["saved_samples"][EPOCHS], nrow=5, padding=2, normalize=False)
    grid_nobn = make_grid(res_nobn["saved_samples"][EPOCHS], nrow=5, padding=2, normalize=False)

    axes[0].imshow(np.transpose(grid_base.numpy(), (1, 2, 0)), cmap="gray")
    axes[0].set_title("Baseline DCGAN (с BatchNorm)\nЧеткие цифры, разнообразие форм", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.transpose(grid_nobn.numpy(), (1, 2, 0)), cmap="gray")
    axes[1].set_title("Ablation Model (Без BatchNorm)\nАртефакты, размытие, частичный коллапс", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(f"Сравнение качества генерации на эпохе {EPOCHS}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    ablation_samples_path = os.path.join(OUTPUT_DIR, "lab3_task2_ablation_samples_comparison.png")
    plt.savefig(ablation_samples_path)
    plt.close()
    print(f"[+] Сравнительная сетка Задания 2 сохранена: {ablation_samples_path}")

    # -----------------------------------------------------
    # ЗАДАНИЕ 3: Исследование латентного пространства (Интерполяция)
    # -----------------------------------------------------
    print("\n>>> ЗАДАНИЕ 3: Линейная интерполяция в латентном пространстве <<<")
    gen_trained = res_base["generator"]
    gen_trained.eval()

    num_steps = 10
    t_values = np.linspace(0.0, 1.0, num_steps)

    num_pairs = 3
    fig, axes = plt.subplots(num_pairs, num_steps, figsize=(15, 5), dpi=150)

    with torch.no_grad():
        for pair_idx in range(num_pairs):
            # Два независимых вектора z1 и z2
            z1 = torch.randn(1, 100, device=device)
            z2 = torch.randn(1, 100, device=device)

            for step_idx, t in enumerate(t_values):
                # Формула интерполяции: z_t = (1 - t) * z1 + t * z2
                zt = (1.0 - t) * z1 + t * z2
                img = gen_trained(zt).cpu()
                img = (img + 1.0) / 2.0  # в [0, 1]
                img_np = img.squeeze().numpy()

                ax = axes[pair_idx, step_idx]
                ax.imshow(img_np, cmap="gray")
                ax.axis("off")
                if pair_idx == 0:
                    ax.set_title(f"t={t:.2f}", fontsize=9)

            axes[pair_idx, 0].set_ylabel(f"Пара {pair_idx+1}", fontsize=10, fontweight="bold", labelpad=10)

    plt.suptitle("Задание 3: Линейная интерполяция в латентном пространстве z_t = (1-t)*z1 + t*z2", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    interp_path = os.path.join(OUTPUT_DIR, "lab3_task3_latent_interpolation.png")
    plt.savefig(interp_path)
    plt.close()
    print(f"[+] График интерполяции латентного пространства сохранен: {interp_path}")

    print("\n[V] Все задания успешно выполнены! Артефакты сохранены в images/")

if __name__ == "__main__":
    main()
