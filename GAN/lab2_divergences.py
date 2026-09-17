"""
Лабораторная работа 2: Дивергенции как инструмент отладки GAN
Проект: GAN

Задачи:
1. Задача 1: KL vs JS: где ломается KL
   - Два дискретных распределения на 5 исходах:
     P = (0.4, 0.3, 0.2, 0.1, 0.0)
     Q = (0.0, 0.1, 0.2, 0.3, 0.4)
   - Расчет D_KL(P || Q), D_KL(Q || P), D_JS(P || Q).
   - Плавный сдвиг Q к P (6 шагов), таблица значений и график.
   - Анализ предельного значения JS (log 2 ≈ 0.693) и проблемы деления на 0 в KL.

2. Задача 2: Проблема носителей: почему GAN не учится
   - Два облака точек в 2D (по 1000 точек):
     P ~ N((-3, 0), I), Q ~ N((3, 0), I)
   - Оценка D_JS(P || Q) через сетку 100x100 (дискретизация и сглаживание по выборке).
   - Плавный сдвиг Q к P (10 шагов), вычисление JS(d).
   - График "расстояние между центрами vs JS", демонстрация плато log 2 и исчезающего градиента.

3. Задача 3: Wasserstein vs KL vs JS: сравнение на одном графике
   - Одномерные гауссианы: P = N(0, 1), Q = N(mu, 1), mu in [0, 10].
   - Аналитический KL (квадратичный: mu^2 / 2).
   - Численный JS (насыщение на log 2).
   - Wasserstein W2 (линейный: mu).
   - Совместный график и таблица при mu in {0, 1, 3, 5, 10}.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.stats as stats

# Стилизация графиков для максимальной наглядности
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.9

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДИВЕРГЕНЦИЙ
# ==============================================================================

def kl_divergence_discrete(p: np.ndarray, q: np.ndarray, eps: float = 0.0) -> float:
    """
    Вычисляет D_KL(P || Q) = sum P(x) * ln(P(x) / Q(x)).
    Если для некоторого x P(x) > 0 и Q(x) == 0, возвращает np.inf (если eps == 0).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    if eps > 0:
        q = np.clip(q, eps, 1.0)
        q = q / np.sum(q)
    
    # Где P > 0, но Q == 0 -> строго бесконечность
    zero_q_mask = (p > 0) & (q == 0)
    if np.any(zero_q_mask):
        return np.inf

    # Ненулевые элементы P
    nonzero = p > 0
    return float(np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero])))


def js_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """
    Вычисляет дивергенцию Йенсена-Шеннона:
    M = 0.5 * (P + Q)
    D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    Гарантированно конечна и лежит в диапазоне [0, ln(2)].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    
    nz_p = p > 0
    nz_q = q > 0
    
    kl_p_m = np.sum(p[nz_p] * np.log(p[nz_p] / m[nz_p]))
    kl_q_m = np.sum(q[nz_q] * np.log(q[nz_q] / m[nz_q]))
    
    return float(0.5 * (kl_p_m + kl_q_m))


# ==============================================================================
# ЗАДАЧА 1. KL vs JS: где ломается KL
# ==============================================================================

def run_task_1():
    print("=" * 80)
    print("ЗАДАЧА 1. KL vs JS: где ломается KL")
    print("=" * 80)

    # 1. Исходные дискретные распределения
    P = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    Q = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    print(f"P = {P}")
    print(f"Q = {Q}")
    print()

    # 2. Плавный сдвиг Q к P (интерполяция Q_t = (1 - t)*Q + t*P) за 6 шагов (t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    steps = 6
    t_values = np.linspace(0.0, 1.0, steps)

    table_data = []
    kl_pq_list = []
    kl_qp_list = []
    js_list = []

    print(f"{'Шаг':<5} | {'t':<5} | {'Q(x)':<34} | {'D_KL(P||Q)':<12} | {'D_KL(Q||P)':<12} | {'D_JS(P||Q)':<12}")
    print("-" * 90)

    for i, t in enumerate(t_values):
        Q_t = (1.0 - t) * Q + t * P
        d_kl_pq = kl_divergence_discrete(P, Q_t)
        d_kl_qp = kl_divergence_discrete(Q_t, P)
        d_js = js_divergence_discrete(P, Q_t)

        kl_pq_list.append(d_kl_pq)
        kl_qp_list.append(d_kl_qp)
        js_list.append(d_js)

        q_str = "[" + ", ".join(f"{x:.2f}" for x in Q_t) + "]"
        kl_pq_str = f"{d_kl_pq:.4f}" if np.isfinite(d_kl_pq) else "∞ (inf)"
        kl_qp_str = f"{d_kl_qp:.4f}" if np.isfinite(d_kl_qp) else "∞ (inf)"
        print(f"{i:<5} | {t:<5.1f} | {q_str:<34} | {kl_pq_str:<12} | {kl_qp_str:<12} | {d_js:<12.4f}")

        table_data.append({
            'step': i,
            't': t,
            'Q_t': Q_t,
            'kl_pq': d_kl_pq,
            'kl_qp': d_kl_qp,
            'js': d_js
        })

    # Теоретический максимум JS при полностью непересекающихся носителях
    log2_val = np.log(2.0)
    print()
    print(f"Теоретический верхний предел D_JS(P || Q) = ln(2) ≈ {log2_val:.6f}")
    print(f"На шаге 0 при частично совпадающих носителях: D_JS(P || Q) = {js_list[0]:.4f} <= ln(2).")

    # Построение графиков (2 подграфика: один с визуализацией распределений, второй с кривыми дивергенций)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Левый график: профили вероятностей P и Q на разных шагах
    outcomes = np.arange(1, 6)
    width = 0.22
    axes[0].bar(outcomes - width, P, width=width, label='P (Целевое)', color='#1f77b4', alpha=0.9)
    axes[0].bar(outcomes, Q, width=width, label='Q (Шаг 0, t=0.0)', color='#d62728', alpha=0.8)
    axes[0].bar(outcomes + width, table_data[2]['Q_t'], width=width, label='Q (Шаг 2, t=0.4)', color='#2ca02c', alpha=0.8)
    axes[0].set_title('Дискретные распределения вероятностей на 5 исходах', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Исход (x)', fontsize=11)
    axes[0].set_ylabel('Вероятность', fontsize=11)
    axes[0].set_xticks(outcomes)
    axes[0].set_ylim(0, 0.5)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Правый график: Кривые дивергенций vs номер шага
    steps_arr = np.arange(steps)

    kl_pq_plot = np.array(kl_pq_list)
    kl_qp_plot = np.array(kl_qp_list)

    # Заменим inf на репрезентативное значение для наглядного отображения разрыва
    max_finite = max(np.max(kl_pq_plot[np.isfinite(kl_pq_plot)]), np.max(kl_qp_plot[np.isfinite(kl_qp_plot)]))
    inf_repr = max_finite * 1.35

    # Строим кривые для конечных шагов (шаги 1..5)
    axes[1].plot(steps_arr[1:], kl_pq_plot[1:], 'o-', color='#e74c3c', linewidth=2.2, label=r'$D_{KL}(P \parallel Q)$')
    axes[1].plot(steps_arr[1:], kl_qp_plot[1:], 's-', color='#e67e22', linewidth=2.2, label=r'$D_{KL}(Q \parallel P)$')
    axes[1].plot(steps_arr, js_list, '^-', color='#2980b9', linewidth=2.5, markersize=8, label=r'$D_{JS}(P \parallel Q)$')

    # Отмечаем точки шага 0 (где KL уходит в бесконечность)
    axes[1].plot([0], [inf_repr], marker='x', markersize=12, markeredgewidth=3, color='#c0392b')
    axes[1].plot([0, 1], [inf_repr, kl_pq_plot[1]], ':', color='#e74c3c', linewidth=2)
    axes[1].annotate(r'$D_{KL}(P \parallel Q) = \infty$' + '\n(так как $Q(1)=0$, $P(1)>0$)',
                     xy=(0, inf_repr), xytext=(0.2, inf_repr * 0.92),
                     arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                     fontsize=9.5, fontweight='bold', color='#c0392b',
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", alpha=0.8))

    # Горизонтальная линия предела log 2
    axes[1].axhline(log2_val, color='#8e44ad', linestyle='--', linewidth=1.8, label=r'Предел $\ln(2) \approx 0.6931$')

    axes[1].set_title(r'Поведение дивергенций при сдвиге $Q \to P$', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Номер шага (0: Q, 5: P)', fontsize=11)
    axes[1].set_ylabel('Значение дивергенции', fontsize=11)
    axes[1].set_xticks(steps_arr)
    axes[1].set_ylim(-0.05, inf_repr * 1.1)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'lab2_task1_kl_vs_js.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"График Задачи 1 сохранен в: {plot_path}")
    print()

    return table_data


# ==============================================================================
# ЗАДАЧА 2. Проблема носителей: почему GAN не учится
# ==============================================================================

def run_task_2():
    print("=" * 80)
    print("ЗАДАЧА 2. Проблема носителей: почему GAN не учится")
    print("=" * 80)

    np.random.seed(42)
    n_points = 1000

    # P ~ N((-3, 0), I)
    mean_P = np.array([-3.0, 0.0])
    cov_I = np.eye(2)
    data_P = np.random.multivariate_normal(mean_P, cov_I, size=n_points)

    # Исходный центр Q: (3, 0), целевой центр: (-3, 0)
    # 10 шагов сдвига от d = 6.0 до d = 0.0
    num_steps = 11  # 10 интервалов (шаги 0..10)
    shift_ratios = np.linspace(0.0, 1.0, num_steps)

    # Сетка 100x100 в границах x in [-7, 7], y in [-4, 4]
    grid_size = 100
    x_lin = np.linspace(-7.0, 7.0, grid_size)
    y_lin = np.linspace(-4.0, 4.0, grid_size)
    X, Y = np.meshgrid(x_lin, y_lin)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    # Вычисляем распределение P на сетке 100x100 двумя способами:
    # 1. По выборке (KDE по 1000 сгенерированным точкам)
    kde_P = stats.gaussian_kde(data_P.T)
    prob_P_kde = kde_P(grid_coords).reshape(grid_size, grid_size)
    prob_P_kde /= np.sum(prob_P_kde)

    # 2. Точная теоретическая плотность на сетке 100x100
    dens_P_theor = np.exp(-0.5 * ((X - mean_P[0])**2 + (Y - mean_P[1])**2))
    prob_P_theor = dens_P_theor / np.sum(dens_P_theor)

    # Базовые точки для выборки Q
    base_points_Q = np.random.multivariate_normal([0.0, 0.0], cov_I, size=n_points)

    distances = []
    js_values_sample_grid = []
    js_values_theor_grid = []

    print(f"{'Шаг':<5} | {'Центр Q':<18} | {'Расстояние d':<14} | {'JS (сетка по выборке)':<22} | {'JS (сетка теоретич.)':<22}")
    print("-" * 90)

    table_data = []

    for step_idx, ratio in enumerate(shift_ratios):
        # Центр Q плавно смещается от (3, 0) к (-3, 0)
        current_center_Q = (1.0 - ratio) * np.array([3.0, 0.0]) + ratio * mean_P
        dist = float(np.linalg.norm(current_center_Q - mean_P))

        # 1. Облако точек выборки Q
        data_Q = base_points_Q + current_center_Q
        kde_Q = stats.gaussian_kde(data_Q.T)
        prob_Q_kde = kde_Q(grid_coords).reshape(grid_size, grid_size)
        prob_Q_kde /= np.sum(prob_Q_kde)

        # Вычисление JS на сетке по выборке
        prob_M_kde = 0.5 * (prob_P_kde + prob_Q_kde)
        kl_p_k = np.sum(prob_P_kde * np.log(prob_P_kde / prob_M_kde))
        kl_q_k = np.sum(prob_Q_kde * np.log(prob_Q_kde / prob_M_kde))
        js_sample = float(0.5 * (kl_p_k + kl_q_k))

        # 2. Точная дискретизация плотностей на сетке 100x100
        dens_Q_theor = np.exp(-0.5 * ((X - current_center_Q[0])**2 + (Y - current_center_Q[1])**2))
        prob_Q_theor = dens_Q_theor / np.sum(dens_Q_theor)
        prob_M_theor = 0.5 * (prob_P_theor + prob_Q_theor)
        kl_p_t = np.sum(prob_P_theor * np.log(prob_P_theor / prob_M_theor))
        kl_q_t = np.sum(prob_Q_theor * np.log(prob_Q_theor / prob_M_theor))
        js_theor = float(0.5 * (kl_p_t + kl_q_t))

        distances.append(dist)
        js_values_sample_grid.append(js_sample)
        js_values_theor_grid.append(js_theor)

        center_str = f"({current_center_Q[0]:.2f}, {current_center_Q[1]:.2f})"
        print(f"{step_idx:<5} | {center_str:<18} | {dist:<14.2f} | {js_sample:<22.4f} | {js_theor:<22.4f}")

        table_data.append({
            'step': step_idx,
            'center_Q': current_center_Q,
            'dist': dist,
            'js_sample': js_sample,
            'js_theor': js_theor
        })

    # Построение многопанельного графика
    fig = plt.figure(figsize=(16, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

    # Панель 1: Облака точек при d = 6.0 (начальное состояние)
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(data_P[:, 0], data_P[:, 1], color='#2980b9', alpha=0.4, s=15, label=r'$P \sim \mathcal{N}((-3, 0), I)$')
    data_Q_start = base_points_Q + np.array([3.0, 0.0])
    ax1.scatter(data_Q_start[:, 0], data_Q_start[:, 1], color='#e74c3c', alpha=0.4, s=15, label=r'$Q \sim \mathcal{N}((3, 0), I)$')
    ax1.plot([-3], [0], 'k+', markersize=14, markeredgewidth=2.5)
    ax1.plot([3], [0], 'k+', markersize=14, markeredgewidth=2.5)
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(-4, 4)
    ax1.set_title(r'Шаг 0: $d=6.0$ (Носители разделены)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('$X_1$')
    ax1.set_ylabel('$X_2$')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Панель 2: Облака точек при d = 1.8 (фаза пересечения)
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(data_P[:, 0], data_P[:, 1], color='#2980b9', alpha=0.4, s=15, label=r'$P$')
    data_Q_mid = base_points_Q + np.array([-1.2, 0.0])
    ax2.scatter(data_Q_mid[:, 0], data_Q_mid[:, 1], color='#e74c3c', alpha=0.4, s=15, label=r'$Q$ ($d=1.8$)')
    ax2.plot([-3], [0], 'k+', markersize=14, markeredgewidth=2.5)
    ax2.plot([-1.2], [0], 'k+', markersize=14, markeredgewidth=2.5)
    ax2.set_xlim(-7, 7)
    ax2.set_ylim(-4, 4)
    ax2.set_title(r'Шаг 7: $d=1.8$ (Начало наложения)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('$X_1$')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Панель 3: Кривая "Расстояние между центрами vs JS"
    ax3 = fig.add_subplot(gs[2])
    d_arr = np.array(distances)
    ax3.plot(d_arr, js_values_sample_grid, 'o-', color='#c0392b', linewidth=2.2, markersize=7, label=r'$D_{JS}$ (сетка $100 \times 100$ по точкам)')
    ax3.plot(d_arr, js_values_theor_grid, '--', color='#2c3e50', linewidth=2.0, label=r'$D_{JS}$ (теоретич. дискретизация)')

    log2_const = np.log(2.0)
    ax3.axhline(log2_const, color='#8e44ad', linestyle=':', linewidth=2, label=r'Плато: $\ln(2) \approx 0.6931$')

    # Аннотация плато и нулевого градиента
    ax3.annotate(r'Плато: $D_{JS} \approx \ln(2)$' + '\n' + r'Градиент $\nabla JS \approx 0$ (GAN зависает)',
                 xy=(4.5, log2_const), xytext=(2.2, 0.45),
                 arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#4a148c',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="#8e44ad", alpha=0.9))

    ax3.annotate('Резкий спад к 0\nПоявление градиента',
                 xy=(1.0, 0.12), xytext=(1.8, 0.1),
                 arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                 fontsize=9, fontweight='bold', color='#b71c1c')

    ax3.set_title(r'Зависимость $D_{JS}$ от расстояния между центрами', fontsize=12, fontweight='bold')
    ax3.set_xlabel(r'Расстояние между центрами $d = ||\mu_P - \mu_Q||$', fontsize=11)
    ax3.set_ylabel(r'$D_{JS}(P \parallel Q)$', fontsize=11)
    ax3.set_ylim(-0.03, 0.8)
    ax3.legend(loc='center right', frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'lab2_task2_support_problem.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"График Задачи 2 сохранен в: {plot_path}")
    print()

    return table_data


# ==============================================================================
# ЗАДАЧА 3. Wasserstein vs KL vs JS: сравнение на одном графике
# ==============================================================================

def run_task_3():
    print("=" * 80)
    print("ЗАДАЧА 3. Wasserstein vs KL vs JS: сравнение на одном графике")
    print("=" * 80)

    # P = N(0, 1), Q = N(mu, 1), mu in [0, 10]
    mu_dense = np.linspace(0.0, 10.0, 300)

    # 1. D_KL(P || Q) по аналитической формуле:
    # D_KL(N(0, 1) || N(mu, 1)) = ln(1/1) + (1 + (0 - mu)^2)/(2 * 1) - 1/2 = mu^2 / 2
    kl_values_dense = 0.5 * (mu_dense ** 2)

    # 2. W(P, Q) по формуле W = sqrt((mu1 - mu2)^2 + (sigma1 - sigma2)^2) = sqrt(mu^2 + 0) = mu
    w_values_dense = mu_dense.copy()

    # 3. D_JS(P || Q) численно через численное интегрирование
    def compute_js_gaussian_1d(mu_val: float) -> float:
        if abs(mu_val) < 1e-12:
            return 0.0
        p = lambda x: stats.norm.pdf(x, loc=0.0, scale=1.0)
        q = lambda x: stats.norm.pdf(x, loc=mu_val, scale=1.0)
        m = lambda x: 0.5 * (p(x) + q(x))

        integrand_p = lambda x: p(x) * np.log(np.maximum(p(x) / m(x), 1e-25))
        integrand_q = lambda x: q(x) * np.log(np.maximum(q(x) / m(x), 1e-25))

        x_min = min(0.0, mu_val) - 6.0
        x_max = max(0.0, mu_val) + 6.0

        val_p, _ = quad(integrand_p, x_min, x_max, limit=200)
        val_q, _ = quad(integrand_q, x_min, x_max, limit=200)
        return float(0.5 * (val_p + val_q))

    # Для построения плавной кривой JS вычислим на 60 точках
    mu_js_sample = np.linspace(0.0, 10.0, 60)
    js_sample = np.array([compute_js_gaussian_1d(m) for m in mu_js_sample])

    # Точные контрольные точки: mu = 0, 1, 3, 5, 10
    control_mus = [0.0, 1.0, 3.0, 5.0, 10.0]
    control_table = []

    print(f"{'μ':<6} | {'KL (μ²/2)':<16} | {'JS (численно)':<16} | {'Wasserstein (μ)':<16}")
    print("-" * 62)

    for m in control_mus:
        kl_val = 0.5 * (m ** 2)
        js_val = compute_js_gaussian_1d(m)
        w_val = m
        print(f"{m:<6.1f} | {kl_val:<16.4f} | {js_val:<16.4f} | {w_val:<16.4f}")
        control_table.append({
            'mu': m,
            'kl': kl_val,
            'js': js_val,
            'w': w_val
        })

    # Построение графиков: общий масштаб и детальный масштаб при малых mu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Полный диапазон mu in [0, 10]
    ax1.plot(mu_dense, kl_values_dense, '-', color='#e74c3c', linewidth=2.5, label=r'$D_{KL}(P \parallel Q) = \frac{1}{2}\mu^2$ (Квадратичная)')
    ax1.plot(mu_dense, w_values_dense, '-', color='#27ae60', linewidth=2.5, label=r'$W(P, Q) = \mu$ (Линейная)')
    ax1.plot(mu_js_sample, js_sample, '-', color='#2980b9', linewidth=2.5, label=r'$D_{JS}(P \parallel Q)$ (Плато $\ln 2$)')

    # Контрольные точки на основном графике
    ctrl_mus_arr = np.array(control_mus)
    ctrl_kl = 0.5 * (ctrl_mus_arr ** 2)
    ctrl_w = ctrl_mus_arr
    ctrl_js = np.array([row['js'] for row in control_table])

    ax1.scatter(ctrl_mus_arr, ctrl_kl, color='#c0392b', s=50, zorder=5)
    ax1.scatter(ctrl_mus_arr, ctrl_w, color='#1e8449', s=50, zorder=5)
    ax1.scatter(ctrl_mus_arr, ctrl_js, color='#1b4f72', s=50, zorder=5)

    ax1.axhline(np.log(2.0), color='#8e44ad', linestyle='--', linewidth=1.5, label=r'Предел JS: $\ln(2) \approx 0.6931$')

    ax1.set_title(r'Сравнение метрик при $\mu \in [0, 10]$', fontsize=12, fontweight='bold')
    ax1.set_xlabel(r'Смещение среднего $\mu$', fontsize=11)
    ax1.set_ylabel('Значение метрики / дивергенции', fontsize=11)
    ax1.set_ylim(-0.5, 52)
    ax1.set_xlim(-0.2, 10.2)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Аннотация взрыва KL
    ax1.annotate(r'Взрыв $D_{KL} = 50.0$' + '\nВзрыв градиентов!',
                 xy=(10.0, 50.0), xytext=(7.0, 42.0),
                 arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", alpha=0.8))

    # 2. Масштабированный график при малых mu in [0, 3] для демонстрации поведения вблизи нуля
    mask_small = mu_dense <= 3.0
    ax2.plot(mu_dense[mask_small], kl_values_dense[mask_small], '-', color='#e74c3c', linewidth=2.5, label=r'$D_{KL} = \frac{1}{2}\mu^2$')
    ax2.plot(mu_dense[mask_small], w_values_dense[mask_small], '-', color='#27ae60', linewidth=2.5, label=r'$W = \mu$ (стабильный градиент = 1)')

    mask_js_small = mu_js_sample <= 3.0
    ax2.plot(mu_js_sample[mask_js_small], js_sample[mask_js_small], '-', color='#2980b9', linewidth=2.5, label=r'$D_{JS}$ (насыщение к $\ln 2$)')
    ax2.axhline(np.log(2.0), color='#8e44ad', linestyle='--', linewidth=1.5, label=r'Предел $\ln(2) \approx 0.6931$')

    ax2.set_title(r'Поведение вблизи нуля ($\mu \in [0, 3]$)', fontsize=12, fontweight='bold')
    ax2.set_xlabel(r'Смещение среднего $\mu$', fontsize=11)
    ax2.set_ylabel('Значение метрики', fontsize=11)
    ax2.set_ylim(-0.05, 4.8)
    ax2.set_xlim(-0.1, 3.1)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'lab2_task3_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"График Задачи 3 сохранен в: {plot_path}")
    print()

    return control_table


# ==============================================================================
# ОСНОВНОЙ ВХОД
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "#" * 80)
    print("СТАРТ ВЫПОЛНЕНИЯ ЛАБОРАТОРНОЙ РАБОТЫ №2: ДИВЕРГЕНЦИИ В GAN")
    print("#" * 80 + "\n")

    t1_data = run_task_1()
    t2_data = run_task_2()
    t3_data = run_task_3()

    print("\n" + "#" * 80)
    print("ВСЕ 3 ЗАДАЧИ УСПЕШНО ВЫПОЛНЕНЫ! ГРАФИКИ СОХРАНЕНЫ В images/")
    print("#" * 80 + "\n")
