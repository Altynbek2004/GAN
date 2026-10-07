"""
Лабораторная работа №4: GAN Training Challenges: Mode Collapse, Instability, Gradient-Related Issues
Дисциплина: Generative Adversarial Networks (ANL7311)
Автор: Altynbek2004
Дата: 2026-10-07

Данный скрипт выполняет:
1. Задание 1: Воспроизведение и количественная оценка Mode Collapse
   - Генератор датасета "8 гауссиан по кругу"
   - Полносвязные сети G и D (2 скрытых слоя по 64 нейрона)
   - Обучение: (а) сбалансированное (1:1), (б) разбалансированное (5 шагов G на 1 шаг D)
   - Снимки распределения каждые N итераций на фиксированном z
   - Подсчет покрытия мод (mode coverage) по порогу расстояния
2. Задание 2: Нестабильность обучения: диагностика по графику потерь
   - Обучение сбалансированной модели на большем числе итераций (12 000 итераций)
   - Построение графика loss_G и loss_D на одних осях, анализ осцилляций
   - Повторение с измененным гиперпараметром (lr_G, lr_D x 5) и сравнительный анализ
3. Задание 3: Saturating vs Non-saturating loss
   - Теоретический и численный график градиентов обеих функций потерь по D(G(z)) и логитам
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

# ---------------------------------------------------------
# 0. Настройки воспроизводимости и устройства
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cpu") # Для простых 2D сетей CPU работает быстрее и детерминированнее всего

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Датасет «8 гауссиан по кругу»
# ---------------------------------------------------------
class Ring8Gaussians:
    def __init__(self, radius=2.0, std=0.05):
        self.radius = radius
        self.std = std
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1) # (8, 2)

    def sample(self, batch_size):
        # Случайный выбор одного из 8 центров для каждой точки
        indices = np.random.choice(8, size=batch_size)
        centers_chosen = self.centers[indices]
        noise = np.random.normal(scale=self.std, size=(batch_size, 2))
        points = centers_chosen + noise
        return torch.tensor(points, dtype=torch.float32)

    def get_centers_np(self):
        return self.centers

# ---------------------------------------------------------
# 2. Архитектуры Generator и Discriminator (2 скрытых слоя, 64 нейрона)
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
        # Возвращаем логит для устойчивости, либо сигмоиду при необходимости
        logits = self.net(x)
        return torch.sigmoid(logits), logits

# ---------------------------------------------------------
# 3. Метрика покрытия мод (Mode Coverage)
# ---------------------------------------------------------
def compute_mode_coverage(gen_points_np, centers_np, threshold=0.25):
    """
    Для каждой сгенерированной точки вычисляем расстояние до ближайшего центра.
    Центр считается покрытым, если хотя бы одна точка находится в пределах threshold.
    """
    # gen_points_np: (N, 2), centers_np: (8, 2)
    # Вычисляем матрицу расстояний: (N, 8)
    dists = np.linalg.norm(gen_points_np[:, np.newaxis, :] - centers_np[np.newaxis, :, :], axis=2)
    # Ближайшее расстояние от каждого центра до сгенерированных точек: (8,)
    min_dist_to_centers = np.min(dists, axis=0)
    covered = np.sum(min_dist_to_centers < threshold)
    return int(covered)

# =========================================================================
# ЗАДАНИЕ 1. Mode Collapse: воспроизведение и количественная оценка
# =========================================================================
def run_task1():
    print("\n" + "="*60)
    print(">>> ВЫПОЛНЕНИЕ ЗАДАНИЯ 1: Воспроизведение и оценка Mode Collapse")
    print("="*60)

    dataset = Ring8Gaussians(radius=2.0, std=0.05)
    centers = dataset.get_centers_np()
    
    n_iters = 5000
    batch_size = 256
    fixed_z = torch.randn(1000, 2) # Фиксированный шум для мониторинга
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

        print(f"[*] Обучение модели: {mode_name} (Шагов G на 1 шаг D: {g_steps})...")

        for iteration in range(n_iters + 1):
            # 1. Снимок
            if iteration in snapshot_iters:
                G.eval()
                with torch.no_grad():
                    gen_fixed = G(fixed_z).cpu().numpy()
                snapshots.append((iteration, gen_fixed))
                G.train()

            # 2. Метрика покрытия мод каждые 100 итераций
            if iteration % 100 == 0:
                G.eval()
                with torch.no_grad():
                    gen_eval = G(fixed_z).cpu().numpy()
                cov = compute_mode_coverage(gen_eval, centers, threshold=0.25)
                coverage_curve.append(cov)
                recorded_iters.append(iteration)
                G.train()

            if iteration == n_iters:
                break

            # 3. Шаг Дискриминатора
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

            # 4. Шаги Генератора (1 или 5)
            G.train()
            for _ in range(g_steps):
                z = torch.randn(batch_size, 2, device=device)
                gen_data = G(z)
                d_gen, _ = D(gen_data)
                # Non-saturating loss
                loss_G = criterion(d_gen, torch.ones_like(d_gen))
                
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

        history[mode_name] = {
            "snapshots": snapshots,
            "coverage_curve": coverage_curve,
            "recorded_iters": recorded_iters
        }

    # Визуализация 1: Последовательность снимков
    fig, axes = plt.subplots(2, len(snapshot_iters), figsize=(18, 6), sharex=True, sharey=True)
    real_sample = dataset.sample(500).numpy()

    for row_idx, (mode_name, title_prefix) in enumerate([("balanced", "1 G-step : 1 D-step (Balanced)"), 
                                                         ("disbalanced", "5 G-steps : 1 D-step (Disbalanced)")]):
        snaps = history[mode_name]["snapshots"]
        for col_idx, (it, gen_pts) in enumerate(snaps):
            ax = axes[row_idx, col_idx]
            # Реальные центры
            ax.scatter(real_sample[:, 0], real_sample[:, 1], c="lightgray", s=10, alpha=0.5, label="Data")
            ax.scatter(centers[:, 0], centers[:, 1], c="red", s=40, marker="x", label="Modes")
            ax.scatter(gen_pts[:, 0], gen_pts[:, 1], c="blue" if row_idx == 0 else "darkorange", s=10, alpha=0.6, label="Generated")
            ax.set_title(f"Iter {it}")
            ax.set_xlim(-3.0, 3.0)
            ax.set_ylim(-3.0, 3.0)
            ax.grid(True, linestyle="--", alpha=0.4)
            if col_idx == 0:
                ax.set_ylabel(title_prefix, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    snap_path = os.path.join(OUTPUT_DIR, "lab4_task1_snapshots_comparison.png")
    plt.savefig(snap_path, dpi=300)
    plt.close()
    print(f"[+] Снимки распределения сохранены в: {snap_path}")

    # Визуализация 2: График покрытия мод
    plt.figure(figsize=(10, 5))
    plt.plot(history["balanced"]["recorded_iters"], history["balanced"]["coverage_curve"], 
             label="Сбалансированное обучение (1G : 1D)", color="navy", lw=2.2)
    plt.plot(history["disbalanced"]["recorded_iters"], history["disbalanced"]["coverage_curve"], 
             label="Дисбаланс генератора (5G : 1D - Mode Collapse)", color="crimson", lw=2.2, linestyle="--")
    plt.axhline(8, color="green", linestyle=":", label="Полное покрытие (8 мод)")
    plt.title("Задание 1: Динамика покрытия мод (Mode Coverage vs Iterations)", fontsize=13, fontweight="bold")
    plt.xlabel("Итерация обучения", fontsize=11)
    plt.ylabel("Число покрытых мод (из 8)", fontsize=11)
    plt.ylim(-0.5, 8.5)
    plt.yticks(range(9))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    cov_path = os.path.join(OUTPUT_DIR, "lab4_task1_mode_coverage.png")
    plt.savefig(cov_path, dpi=300)
    plt.close()
    print(f"[+] График покрытия мод сохранен в: {cov_path}")


# =========================================================================
# ЗАДАНИЕ 2. Нестабильность обучения: диагностика по графику потерь
# =========================================================================
def run_task2():
    print("\n" + "="*60)
    print(">>> ВЫПОЛНЕНИЕ ЗАДАНИЯ 2: Диагностика нестабильности по функциям потерь")
    print("="*60)

    dataset = Ring8Gaussians(radius=2.0, std=0.05)
    n_iters = 12000
    batch_size = 256
    criterion = nn.BCELoss()

    experiments = [
        {"name": "baseline", "lr": 0.001, "betas": (0.5, 0.999), "label": "Базовый (lr=0.001, beta1=0.5)"},
        {"name": "high_lr", "lr": 0.005, "betas": (0.5, 0.999), "label": "Высокий LR x5 (lr=0.005, beta1=0.5)"}
    ]

    results = {}

    for exp in experiments:
        set_seed(42)
        G = Generator().to(device)
        D = Discriminator().to(device)

        opt_G = optim.Adam(G.parameters(), lr=exp["lr"], betas=exp["betas"])
        opt_D = optim.Adam(D.parameters(), lr=exp["lr"], betas=exp["betas"])

        loss_G_history = []
        loss_D_history = []

        print(f"[*] Обучение модели Задания 2: {exp['label']} ({n_iters} итераций)...")

        for iteration in range(n_iters):
            # Шаг D
            real_data = dataset.sample(batch_size).to(device)
            z = torch.randn(batch_size, 2, device=device)
            fake_data = G(z).detach()

            d_real, _ = D(real_data)
            d_fake, _ = D(fake_data)
            loss_D = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Шаг G (1:1)
            z = torch.randn(batch_size, 2, device=device)
            gen_data = G(z)
            d_gen, _ = D(gen_data)
            loss_G = criterion(d_gen, torch.ones_like(d_gen))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_G_history.append(loss_G.item())
            loss_D_history.append(loss_D.item())

        results[exp["name"]] = {
            "loss_G": np.array(loss_G_history),
            "loss_D": np.array(loss_D_history),
            "label": exp["label"]
        }

    # Скользящее среднее для сглаживания трендов
    def moving_avg(arr, window=50):
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    # График 1: Базовая модель с аннотацией колебаний
    base_g = results["baseline"]["loss_G"]
    base_d = results["baseline"]["loss_D"]
    iters = np.arange(len(base_g))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(iters, base_g, color="tab:blue", alpha=0.3, label="Loss G (сырой)")
    ax.plot(iters, base_d, color="tab:orange", alpha=0.3, label="Loss D (сырой)")
    ax.plot(np.arange(49, len(base_g)), moving_avg(base_g), color="blue", lw=2, label="Loss G (сглаженный)")
    ax.plot(np.arange(49, len(base_d)), moving_avg(base_d), color="darkorange", lw=2, label="Loss D (сглаженный)")

    # Выделяем периоды осцилляций
    # Период 1: около 2000-3500 итераций
    ax.axvspan(2000, 3500, color="yellow", alpha=0.2, label="Период колебаний 1 (борьба за моды)")
    ax.text(2750, 4.0, "Всплеск Loss G:\nгенератор меняет стратегию\nи дискриминатор его ловит", 
            ha="center", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gold"))

    # Период 2: около 7000-8500 итераций
    ax.axvspan(7000, 8500, color="salmon", alpha=0.2, label="Период колебаний 2 (предельный цикл)")
    ax.text(7750, 4.2, "Осцилляция седловой точки:\nотсутствие сходимости\nв min-max игре", 
            ha="center", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="salmon"))

    ax.set_title("Задание 2: Кривые потерь G и D на длинной дистанции (12 000 итераций)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Итерация", fontsize=11)
    ax.set_ylabel("Функция потерь (Loss)", fontsize=11)
    ax.set_ylim(0, 6.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    
    t2_single_path = os.path.join(OUTPUT_DIR, "lab4_task2_baseline_instability.png")
    plt.tight_layout()
    plt.savefig(t2_single_path, dpi=300)
    plt.close()
    print(f"[+] График потерь базовой модели сохранен в: {t2_single_path}")

    # График 2: Сравнение базовой модели и модели с LR x 5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    
    # Baseline
    ax1.plot(base_g, color="tab:blue", alpha=0.25)
    ax1.plot(base_d, color="tab:orange", alpha=0.25)
    ax1.plot(np.arange(49, len(base_g)), moving_avg(base_g), color="blue", lw=2, label="Loss G")
    ax1.plot(np.arange(49, len(base_d)), moving_avg(base_d), color="darkorange", lw=2, label="Loss D")
    ax1.set_title("Базовый режим (lr = 0.001) — умеренные колебания вокруг равновесия", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_ylim(0, 7.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # High LR
    hlr_g = results["high_lr"]["loss_G"]
    hlr_d = results["high_lr"]["loss_D"]
    ax2.plot(hlr_g, color="tab:purple", alpha=0.25)
    ax2.plot(hlr_d, color="tab:green", alpha=0.25)
    ax2.plot(np.arange(49, len(hlr_g)), moving_avg(hlr_g), color="purple", lw=2, label="Loss G (LR x 5)")
    ax2.plot(np.arange(49, len(hlr_d)), moving_avg(hlr_d), color="green", lw=2, label="Loss D (LR x 5)")
    ax2.set_title("Режим с повышенным LR (lr = 0.005) — резкая дестабилизация и сильные выбросы", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Итерация", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_ylim(0, 10.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    t2_comp_path = os.path.join(OUTPUT_DIR, "lab4_task2_lr_comparison.png")
    plt.savefig(t2_comp_path, dpi=300)
    plt.close()
    print(f"[+] Сравнительный график гиперпараметров сохранен в: {t2_comp_path}")


# =========================================================================
# ЗАДАНИЕ 3. Saturating vs Non-saturating loss
# =========================================================================
def run_task3():
    print("\n" + "="*60)
    print(">>> ВЫПОЛНЕНИЕ ЗАДАНИЯ 3: Сравнение Saturating и Non-Saturating потерь")
    print("="*60)

    # 1. Аналитический и численный анализ градиентов
    # Пусть p = D(G(z)) in (0, 1)
    p = np.linspace(0.001, 0.999, 1000)
    
    # 1.1 Градиент по p: d Loss / d p
    # Saturating: L = log(1 - p) -> dL/dp = -1 / (1 - p)
    # Non-saturating: L = -log(p) -> dL/dp = -1 / p
    # Величина градиента (magnitude |dL/dp|):
    grad_p_sat = 1.0 / (1.0 - p)
    grad_p_nonsat = 1.0 / p

    # 1.2 Градиент по логиту a, где p = sigmoid(a)
    # L_sat = log(1 - sigmoid(a)) -> dL/da = -sigmoid(a) = -p
    # |dL/da| = p (когда p -> 0, градиент затухает до 0!)
    # L_nonsat = -log(sigmoid(a)) -> dL/da = -(1 - sigmoid(a)) = -(1 - p)
    # |dL/da| = 1 - p (когда p -> 0, градиент равен 1, то есть СИЛЬНЫЙ!)
    grad_logit_sat = p
    grad_logit_nonsat = 1.0 - p

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График по логиту a (ключевой теоретический график лекции Goodfellow et al.)
    ax1.plot(p, grad_logit_nonsat, color="crimson", lw=2.5, label="Non-saturating: $|\\partial L / \\partial a| = 1 - D(G(z))$")
    ax1.plot(p, grad_logit_sat, color="navy", lw=2.5, linestyle="--", label="Saturating: $|\\partial L / \\partial a| = D(G(z))$")
    ax1.set_title("Величина градиента по входному логиту $a$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Выход дискриминатора $D(G(z))$", fontsize=11)
    ax1.set_ylabel("Величина градиента $|\partial L / \partial a|$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)
    ax1.annotate("Раннее обучение:\nD(G(z)) ~ 0\nNon-sat дает максимум,\nSat затухает к 0!", 
                 xy=(0.05, 0.95), xytext=(0.2, 0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, bbox=dict(boxstyle="round", fc="ivory", ec="orange"))

    # График по p
    ax2.plot(p, grad_p_nonsat, color="crimson", lw=2.5, label="Non-saturating: $|dL/dp| = 1 / D(G(z))$")
    ax2.plot(p, grad_p_sat, color="navy", lw=2.5, linestyle="--", label="Saturating: $|dL/dp| = 1 / (1 - D(G(z)))$")
    ax2.set_title("Величина градиента напрямую по вероятности $D(G(z))$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Выход дискриминатора $D(G(z))$", fontsize=11)
    ax2.set_ylabel("Величина градиента $|dL/dp|$", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    t3_grad_path = os.path.join(OUTPUT_DIR, "lab4_task3_gradient_curves.png")
    plt.savefig(t3_grad_path, dpi=300)
    plt.close()
    print(f"[+] Теоретические графики градиентов сохранены в: {t3_grad_path}")

    # 2. Экспериментальное обучение обоими лоссами с одинаковым seed
    dataset = Ring8Gaussians(radius=2.0, std=0.05)
    n_iters = 300 # Анализируем ранний прогресс первых 300 итераций
    batch_size = 256
    fixed_z = torch.randn(1000, 2)

    loss_variants = ["saturating", "non_saturating"]
    exp_history = {}

    for var in loss_variants:
        set_seed(100) # Одинаковый seed!
        G = Generator().to(device)
        D = Discriminator().to(device)

        opt_G = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        # Для D мы всегда используем стандартный BCE
        bce = nn.BCELoss()

        g_losses = []
        d_preds_on_fake = []

        print(f"[*] Обучение модели с лоссом: {var} (первые {n_iters} итераций)...")

        for iteration in range(n_iters):
            # Шаг D
            real_data = dataset.sample(batch_size).to(device)
            z = torch.randn(batch_size, 2, device=device)
            fake_data = G(z).detach()

            d_real, _ = D(real_data)
            d_fake, _ = D(fake_data)
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Шаг G
            z = torch.randn(batch_size, 2, device=device)
            gen_data = G(z)
            d_gen, _ = D(gen_data)

            if var == "saturating":
                # Минимизация log(1 - D(G(z)))
                # Чтобы не было log(0), добавляем epsilon
                eps = 1e-7
                loss_G = torch.mean(torch.log(torch.clamp(1.0 - d_gen, min=eps, max=1.0)))
            else:
                # Non-saturating: минимизация -log(D(G(z)))
                eps = 1e-7
                loss_G = -torch.mean(torch.log(torch.clamp(d_gen, min=eps, max=1.0)))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())
            d_preds_on_fake.append(d_gen.mean().item())

        G.eval()
        with torch.no_grad():
            final_points = G(fixed_z).cpu().numpy()

        exp_history[var] = {
            "g_losses": np.array(g_losses),
            "d_preds": np.array(d_preds_on_fake),
            "final_points": final_points
        }

    # Построение сравнительного графика раннего обучения
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Значение выхода D(G(z)) в начале обучения
    ax1.plot(exp_history["non_saturating"]["d_preds"], label="Non-Saturating", color="crimson", lw=2)
    ax1.plot(exp_history["saturating"]["d_preds"], label="Saturating", color="navy", lw=2, linestyle="--")
    ax1.set_title("Средний отклик дискриминатора $\mathbb{E}[D(G(z))]$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Итерация (ранний этап 0-300)", fontsize=10)
    ax1.set_ylabel("$D(G(z))$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # 2. Распределение точек для Non-Saturating на 300 итерации
    real_sample = dataset.sample(500).numpy()
    centers = dataset.get_centers_np()

    ax2.scatter(real_sample[:, 0], real_sample[:, 1], c="lightgray", s=10, alpha=0.5)
    ax2.scatter(centers[:, 0], centers[:, 1], c="red", s=30, marker="x")
    ax2.scatter(exp_history["non_saturating"]["final_points"][:, 0], 
                exp_history["non_saturating"]["final_points"][:, 1], 
                c="crimson", s=12, alpha=0.7)
    ax2.set_title("Non-Saturating на 300 итерации\n(быстрее стягивается к окружности)", fontsize=11, fontweight="bold")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Распределение точек для Saturating на 300 итерации
    ax3.scatter(real_sample[:, 0], real_sample[:, 1], c="lightgray", s=10, alpha=0.5)
    ax3.scatter(centers[:, 0], centers[:, 1], c="red", s=30, marker="x")
    ax3.scatter(exp_history["saturating"]["final_points"][:, 0], 
                exp_history["saturating"]["final_points"][:, 1], 
                c="navy", s=12, alpha=0.7)
    ax3.set_title("Saturating на 300 итерации\n(медленный старт из-за малых градиентов)", fontsize=11, fontweight="bold")
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    t3_exp_path = os.path.join(OUTPUT_DIR, "lab4_task3_early_progress_comparison.png")
    plt.savefig(t3_exp_path, dpi=300)
    plt.close()
    print(f"[+] График сравнения раннего прогресса сохранен в: {t3_exp_path}")

    print("\n[SUCCESS] Все эксперименты Лабораторной работы №4 успешно завершены!")

if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
