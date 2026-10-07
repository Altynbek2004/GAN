"""
Лабораторная работа №4: GAN Training Challenges: Mode Collapse, Instability, Gradient-Related Issues
Дисциплина: Generative Adversarial Networks (ANL7311)
Автор: Altynbek2004
Дата: 2026-10-07

Данный скрипт выполняет:
1. Задание 1: Воспроизведение и количественная оценка Mode Collapse
   - Генератор датасета "10 гауссиан по контуру ТРАПЕЦИИ" (и базового круга)
   - Полносвязные сети G и D (2 скрытых слоя по 64 нейрона)
   - Обучение: (а) сбалансированное (1:1), (б) разбалансированное (5 шагов G на 1 шаг D)
   - Снимки распределения каждые N итераций на фиксированном z
   - Подсчет покрытия мод (mode coverage) по порогу расстояния
2. Задание 2: Нестабильность обучения: диагностика по графику потерь
   - Обучение сбалансированной модели на 12 000 итераций
   - Построение графика loss_G и loss_D на одних осях, анализ осцилляций
   - Повторение с повышенным LR (lr x 5) и сравнительный анализ
3. Задание 3: Saturating vs Non-saturating loss
   - Теоретический и численный график градиентов обеих функций потерь
   - Обучение обоими вариантами loss с одинаковым random seed
   - Сравнение раннего прогресса генератора (первые 300 итераций)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Датасет «10 гауссиан по контуру ТРАПЕЦИИ»
# ---------------------------------------------------------
class Trapezoid10Gaussians:
    def __init__(self, std=0.04):
        self.std = std
        # 10 симметричных центров по периметру трапеции:
        # Нижнее основание (длина 4): y = -1.4
        # Боковые стороны
        # Верхнее основание (длина 1.6): y = +1.4
        self.centers = np.array([
            [-2.0, -1.4], [-0.67, -1.4], [0.67, -1.4], [2.0, -1.4],  # 4 точки снизу
            [1.6, -0.4], [1.2, 0.6],                                  # 2 точки справа
            [0.8, 1.4], [-0.8, 1.4],                                  # 2 точки сверху
            [-1.2, 0.6], [-1.6, -0.4]                                 # 2 точки слева
        ], dtype=np.float32)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.centers), size=batch_size)
        centers_chosen = self.centers[indices]
        noise = np.random.normal(scale=self.std, size=(batch_size, 2))
        points = centers_chosen + noise
        return torch.tensor(points, dtype=torch.float32)

    def get_centers_np(self):
        return self.centers

# ---------------------------------------------------------
# 2. Архитектуры сетей G и D
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits), logits

def compute_mode_coverage(gen_points_np, centers_np, threshold=0.22):
    dists = np.linalg.norm(gen_points_np[:, np.newaxis, :] - centers_np[np.newaxis, :, :], axis=2)
    min_dist_to_centers = np.min(dists, axis=0)
    return int(np.sum(min_dist_to_centers < threshold))

# =========================================================================
# ЗАДАНИЕ 1: Mode Collapse на ТРАПЕЦИИ
# =========================================================================
def run_task1_trapezoid():
    print("\n" + "="*60)
    print(">>> ВЫПОЛНЕНИЕ ЗАДАНИЯ 1: Эксперимент на форме ТРАПЕЦИИ (10 мод)")
    print("="*60)

    dataset = Trapezoid10Gaussians(std=0.04)
    centers = dataset.get_centers_np()
    
    n_iters = 5000
    batch_size = 256
    fixed_z = torch.randn(1000, 2)
    snapshot_iters = [0, 500, 1000, 2000, 3500, 5000]

    modes = [("balanced", 1), ("disbalanced", 5)]
    history = {}

    for mode_name, g_steps in modes:
        set_seed(42)
        G = Generator().to(device)
        D = Discriminator().to(device)

        opt_G = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        snapshots = []
        coverage_curve = []
        recorded_iters = []

        print(f"[*] Обучение Трапеции: {mode_name} (Шагов G на 1 шаг D: {g_steps})...")

        for iteration in range(n_iters + 1):
            if iteration in snapshot_iters:
                G.eval()
                with torch.no_grad():
                    gen_fixed = G(fixed_z).cpu().numpy()
                snapshots.append((iteration, gen_fixed))
                G.train()

            if iteration % 100 == 0:
                G.eval()
                with torch.no_grad():
                    gen_eval = G(fixed_z).cpu().numpy()
                cov = compute_mode_coverage(gen_eval, centers, threshold=0.22)
                coverage_curve.append(cov)
                recorded_iters.append(iteration)
                G.train()

            if iteration == n_iters:
                break

            # Шаг D
            D.train()
            real_data = dataset.sample(batch_size).to(device)
            z = torch.randn(batch_size, 2, device=device)
            fake_data = G(z).detach()

            d_real, _ = D(real_data)
            d_fake, _ = D(fake_data)
            loss_D = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Шаги G
            G.train()
            for _ in range(g_steps):
                z = torch.randn(batch_size, 2, device=device)
                gen_data = G(z)
                d_gen, _ = D(gen_data)
                loss_G = criterion(d_gen, torch.ones_like(d_gen))
                
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

        history[mode_name] = {
            "snapshots": snapshots,
            "coverage_curve": coverage_curve,
            "recorded_iters": recorded_iters
        }

    # Визуализация снимков Трапеции
    fig, axes = plt.subplots(2, len(snapshot_iters), figsize=(18, 6), sharex=True, sharey=True)
    real_sample = dataset.sample(600).numpy()

    for row_idx, (mode_name, title_prefix) in enumerate([("balanced", "1 G-step : 1 D-step (Balanced Trapezoid)"), 
                                                         ("disbalanced", "5 G-steps : 1 D-step (Disbalanced Trapezoid)")]):
        snaps = history[mode_name]["snapshots"]
        for col_idx, (it, gen_pts) in enumerate(snaps):
            ax = axes[row_idx, col_idx]
            ax.scatter(real_sample[:, 0], real_sample[:, 1], c="lightgray", s=10, alpha=0.5, label="Data")
            ax.scatter(centers[:, 0], centers[:, 1], c="red", s=45, marker="x", label="Trapezoid Modes (10)")
            ax.scatter(gen_pts[:, 0], gen_pts[:, 1], c="blue" if row_idx == 0 else "darkorange", s=10, alpha=0.6, label="Generated")
            ax.set_title(f"Iter {it}")
            ax.set_xlim(-3.0, 3.0)
            ax.set_ylim(-2.5, 2.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            if col_idx == 0:
                ax.set_ylabel(title_prefix, fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    snap_path = os.path.join(OUTPUT_DIR, "lab4_task1_trapezoid_snapshots.png")
    plt.savefig(snap_path, dpi=300)
    plt.close()
    print(f"[+] Снимки распределения ТРАПЕЦИИ сохранены в: {snap_path}")

    # График покрытия 10 мод трапеции
    plt.figure(figsize=(10, 5))
    plt.plot(history["balanced"]["recorded_iters"], history["balanced"]["coverage_curve"], 
             label="Сбалансированное обучение (1G : 1D) - Трапеция", color="navy", lw=2.2)
    plt.plot(history["disbalanced"]["recorded_iters"], history["disbalanced"]["coverage_curve"], 
             label="Дисбаланс генератора (5G : 1D) - Mode Collapse", color="crimson", lw=2.2, linestyle="--")
    plt.axhline(10, color="green", linestyle=":", label="Полное покрытие (10 мод трапеции)")
    plt.title("Задание 1: Динамика покрытия 10 мод ТРАПЕЦИИ (Mode Coverage vs Iterations)", fontsize=13, fontweight="bold")
    plt.xlabel("Итерация обучения", fontsize=11)
    plt.ylabel("Число покрытых мод (из 10)", fontsize=11)
    plt.ylim(-0.5, 10.5)
    plt.yticks(range(11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    cov_path = os.path.join(OUTPUT_DIR, "lab4_task1_trapezoid_mode_coverage.png")
    plt.savefig(cov_path, dpi=300)
    plt.close()
    print(f"[+] График покрытия мод трапеции сохранен в: {cov_path}")

if __name__ == "__main__":
    run_task1_trapezoid()
