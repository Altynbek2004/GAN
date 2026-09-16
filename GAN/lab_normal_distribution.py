"""
Лабораторная работа: Нормальное распределение в машинном обучении. От теории к практике.
Проект: GAN

Содержание:
1. Задание 1: Центральная предельная теорема (ЦПТ) — доказательство на пальцах.
   - Моделирование суммы величин из равномерного распределения (U(0, 1)).
   - Проверка формы при n_sum = 50 и сравнение с теоретическим нормальным распределением.
   - Исследование изменения формы при увеличении n_sum до 100.
   - Сравнение с экспоненциальным распределением (Exp(lambda)).
   - Построение гистограмм, KDE, нормальной кривой и Q-Q plot.
   - Оценка минимального n_sum для визуальной нормальности.

2. Задание 2: Исследование влияния параметров на хвосты распределения.
   - Моделирование N(0, sigma^2) для различных sigma.
   - Расчет эмпирической и теоретической вероятности выбросов (|x| > 3).
   - Построение графика зависимости вероятности от sigma.
   - Определение порогового значения sigma, при котором вероятность выброса > 5%.
   - Анализ связи с GAN, Truncation Trick, качеством генерации и затухающими градиентами.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для стабильной генерации графиков без GUI
import matplotlib.pyplot as plt
import scipy.stats as stats

# Настройка эстетики графиков
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def run_task_1():
    print("=" * 70)
    print("ЗАДАНИЕ 1. ЦЕНТРАЛЬНАЯ ПРЕДЕЛЬНАЯ ТЕОРЕМА (ЦПТ) — ДОКАЗАТЕЛЬСТВО НА ПАЛЬЦАХ")
    print("=" * 70)

    np.random.seed(42)
    sample_size = 50_000

    # 1. Генерация одного равномерного распределения U(0, 1)
    single_uniform = np.random.uniform(0, 1, sample_size)

    # 2. Сумма 50 равномерных случайных величин
    n_sum = 50
    # Генерация матрицы (sample_size, n_sum) и суммирование по строкам
    samples_50 = np.random.uniform(0, 1, (sample_size, n_sum)).sum(axis=1)

    # Теоретические параметры для суммы 50 величин U(0, 1):
    # E[X_i] = 0.5, Var[X_i] = 1/12
    # E[S_50] = 50 * 0.5 = 25
    # Var[S_50] = 50 * (1/12) = 50 / 12 = 4.1667
    # Std[S_50] = sqrt(50 / 12) = 2.0412
    mu_theor_50 = n_sum * 0.5
    std_theor_50 = np.sqrt(n_sum / 12.0)

    # Эмпирические параметры
    mu_emp_50 = np.mean(samples_50)
    std_emp_50 = np.std(samples_50)
    skewness_50 = stats.skew(samples_50)
    kurtosis_50 = stats.kurtosis(samples_50)  # excess kurtosis (0 для нормального)

    print(f"Параметры для n_sum = {n_sum}:")
    print(f"  Теоретическое среднее (mu): {mu_theor_50:.4f}, Эмпирическое: {mu_emp_50:.4f}")
    print(f"  Теоретическое станд. откл. (sigma): {std_theor_50:.4f}, Эмпирическое: {std_emp_50:.4f}")
    print(f"  Асимметрия (Skewness): {skewness_50:.5f} (близко к 0)")
    print(f"  Эксцесс (Excess Kurtosis): {kurtosis_50:.5f} (близко к 0)")

    # -------------------------------------------------------------
    # График 1: Исходное U(0,1) vs Сумма 50 vs Q-Q Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Подграфик A: Исходное распределение
    axes[0].hist(single_uniform, bins=50, density=True, color='#4A90E2', alpha=0.7, edgecolor='white')
    axes[0].axhline(1.0, color='#D0021B', linestyle='--', linewidth=2, label='Теоретическая плотность U(0,1)')
    axes[0].set_title("1. Одиночное U(0, 1)\n(Явно не колокол, плоское)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Значение x")
    axes[0].set_ylabel("Плотность вероятности")
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, 1.4)

    # Подграфик B: Сумма 50 величин + Нормальная кривая
    count, bins, _ = axes[1].hist(samples_50, bins=60, density=True, color='#50E3C2', alpha=0.75,
                                  edgecolor='white', label=f'Гистограмма суммы (n={n_sum})')
    x_grid = np.linspace(bins[0], bins[-1], 300)
    pdf_norm = stats.norm.pdf(x_grid, loc=mu_theor_50, scale=std_theor_50)
    axes[1].plot(x_grid, pdf_norm, color='#9013FE', linewidth=2.5,
                 label=f'Теор. N({mu_theor_50:.1f}, {std_theor_50**2:.2f})')
    axes[1].set_title(f"2. Сумма {n_sum} слагаемых U(0, 1)\n(Формирование идеального колокола)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f"Сумма $S_{{{n_sum}}}$")
    axes[1].set_ylabel("Плотность вероятности")
    axes[1].legend(loc='upper right')

    # Подграфик C: Q-Q Plot для суммы 50
    stats.probplot(samples_50, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_markerfacecolor('#4A90E2')
    axes[2].get_lines()[0].set_markeredgecolor('none')
    axes[2].get_lines()[0].set_alpha(0.3)
    axes[2].get_lines()[0].set_markersize(4)
    axes[2].get_lines()[1].set_color('#D0021B')
    axes[2].get_lines()[1].set_linewidth(2)
    axes[2].set_title(f"3. Q-Q Plot для суммы {n_sum}\n(Точки строго на прямой -> распределение нормально)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Теоретические квантили")
    axes[2].set_ylabel("Квантили выборки")

    plt.tight_layout()
    plot1_path = os.path.join(IMAGES_DIR, 'task1_clt_uniform_50.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"-> График 1 сохранен: {plot1_path}")

    # -------------------------------------------------------------
    # График 2: Прогрессия n_sum: [1, 2, 5, 10, 30, 50, 100]
    # Исследование изменения формы при увеличении n_sum до 100
    # -------------------------------------------------------------
    n_sums = [1, 2, 5, 10, 30, 50, 100]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for idx, n in enumerate(n_sums):
        # Стандартизированная сумма Z_n = (S_n - n*mu) / (sqrt(n)*sigma) для прямого визуального сравнения
        raw_sums = np.random.uniform(0, 1, (sample_size, n)).sum(axis=1)
        z_scores = (raw_sums - n * 0.5) / np.sqrt(n / 12.0)

        ax = axes[idx]
        ax.hist(z_scores, bins=50, density=True, alpha=0.65, color='#4A90E2', edgecolor='white')
        z_grid = np.linspace(-4, 4, 200)
        ax.plot(z_grid, stats.norm.pdf(z_grid, 0, 1), 'r-', lw=2, label=r'$\mathcal{N}(0, 1)$')
        
        sk = stats.skew(raw_sums)
        kt = stats.kurtosis(raw_sums)
        ax.set_title(f"$n_{{sum}} = {n}$\nSkew={sk:.3f}, Kurt={kt:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.45)
        if idx == 0:
            ax.legend(loc='upper right')

    # Сравнение сырых (нестандартизированных) сумм для n=50 и n=100 в 8-м подграфике
    raw_50 = np.random.uniform(0, 1, (sample_size, 50)).sum(axis=1)
    raw_100 = np.random.uniform(0, 1, (sample_size, 100)).sum(axis=1)
    ax8 = axes[7]
    ax8.hist(raw_50, bins=40, density=True, alpha=0.5, color='#F5A623', label=r'Сумма $n=50$ ($\mu=25, \sigma\approx2.04$)')
    ax8.hist(raw_100, bins=40, density=True, alpha=0.5, color='#7ED321', label=r'Сумма $n=100$ ($\mu=50, \sigma\approx2.89$)')
    ax8.set_title("Сравнение сырых сумм:\n$n=50$ vs $n=100$", fontsize=11, fontweight='bold')
    ax8.set_xlabel("Значение суммы")
    ax8.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plot2_path = os.path.join(IMAGES_DIR, 'task1_clt_progression.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"-> График 2 (Прогрессия n_sum до 100) сохранен: {plot2_path}")

    # -------------------------------------------------------------
    # График 3: Экспоненциальное распределение и ЦПТ
    # -------------------------------------------------------------
    exp_ns = [1, 2, 5, 20, 50, 100]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, n in enumerate(exp_ns):
        # Exp(lambda=1): mean=1, std=1
        exp_sums = np.random.exponential(scale=1.0, size=(sample_size, n)).sum(axis=1)
        z_exp = (exp_sums - n * 1.0) / np.sqrt(n * 1.0)

        ax = axes[idx]
        ax.hist(z_exp, bins=50, density=True, alpha=0.65, color='#BD10E0', edgecolor='white')
        z_grid = np.linspace(-4, 5, 200)
        ax.plot(z_grid, stats.norm.pdf(z_grid, 0, 1), 'r-', lw=2, label=r'$\mathcal{N}(0, 1)$')

        sk = stats.skew(exp_sums)
        ax.set_title(f"Exp($\\lambda=1$), $n_{{sum}} = {n}$\nАсимметрия (Skew) = {sk:.3f}",
                     fontsize=11, fontweight='bold')
        ax.set_xlim(-4, 5)
        if idx == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plot3_path = os.path.join(IMAGES_DIR, 'task1_clt_exponential.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"-> График 3 (Экспоненциальное распределение и ЦПТ) сохранен: {plot3_path}")
    print()


def run_task_2():
    print("=" * 70)
    print("ЗАДАНИЕ 2. ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ НА ХВОСТЫ РАСПРЕДЕЛЕНИЯ")
    print("=" * 70)

    np.random.seed(42)
    sample_size = 100_000
    threshold = 3.0

    sigmas = np.arange(0.1, 3.05, 0.05)
    empirical_probs = []
    theoretical_probs = []

    for s in sigmas:
        samples = np.random.normal(loc=0.0, scale=s, size=sample_size)
        # Эмпирическая вероятность |x| > 3
        emp_p = np.mean(np.abs(samples) > threshold)
        # Теоретическая вероятность 2 * (1 - Phi(3 / s))
        theor_p = 2 * stats.norm.sf(threshold / s)

        empirical_probs.append(emp_p)
        theoretical_probs.append(theor_p)

    sigmas = np.array(sigmas)
    empirical_probs = np.array(empirical_probs)
    theoretical_probs = np.array(theoretical_probs)

    # Поиск sigma, при котором вероятность превышает 5% (0.05)
    # Аналитически: 2 * (1 - Phi(3 / sigma)) = 0.05 => 1 - Phi(3 / sigma) = 0.025
    # => 3 / sigma = z_0.025 = 1.959964 => sigma = 3 / 1.959964 = 1.5306
    sigma_5pct_analytical = threshold / stats.norm.ppf(0.975)

    idx_5pct = np.where(theoretical_probs > 0.05)[0][0]
    sigma_5pct_grid = sigmas[idx_5pct]

    print(f"Порог выброса: |x| > {threshold}")
    print(f"Вероятность выброса при sigma = 1.0 (стандартное):")
    p_sigma_1_theor = 2 * stats.norm.sf(3.0 / 1.0)
    print(f"  Теоретическая: {p_sigma_1_theor * 100:.3f}% ({p_sigma_1_theor:.5f})")
    print(f"  Эмпирическая:  {empirical_probs[np.isclose(sigmas, 1.0)][0] * 100:.3f}%")
    print(f"Вероятность выброса при sigma = 2.0:")
    p_sigma_2_theor = 2 * stats.norm.sf(3.0 / 2.0)
    print(f"  Теоретическая: {p_sigma_2_theor * 100:.3f}% ({p_sigma_2_theor:.5f})")
    print(f"Вероятность выброса при sigma = 0.5:")
    p_sigma_05_theor = 2 * stats.norm.sf(3.0 / 0.5)
    print(f"  Теоретическая: {p_sigma_05_theor * 100:.6f}% ({p_sigma_05_theor:.8f})")
    print("-" * 50)
    print(f"Критическая sigma, при которой P(|x| > 3) > 5%:")
    print(f"  Аналитическое значение: sigma = {sigma_5pct_analytical:.4f}")
    print(f"  По сеточной дискретизации: sigma = {sigma_5pct_grid:.2f} (вероятность: {theoretical_probs[idx_5pct]*100:.2f}%)")

    # -------------------------------------------------------------
    # График 4: График зависимости P(|x| > 3) от sigma
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sigmas, theoretical_probs * 100, label='Теоретическая вероятность $2(1 - \\Phi(3/\\sigma))$',
            color='#0072B2', linewidth=2.5)
    ax.scatter(sigmas[::2], empirical_probs[::2] * 100, color='#D55E00', s=35, alpha=0.8,
               label='Эмпирическая выборка ($N=100\\,000$)', zorder=4)

    # Линия 5% порога
    ax.axhline(5.0, color='#E69F00', linestyle='--', linewidth=1.8, label='Порог 5%')
    ax.axvline(sigma_5pct_analytical, color='#CC79A7', linestyle=':', linewidth=2,
               label=f'$\\sigma_{{критич}} = {sigma_5pct_analytical:.2f}$')

    # Точка пересечения
    ax.scatter([sigma_5pct_analytical], [5.0], color='#D0021B', s=100, zorder=5, edgecolor='black')
    ax.annotate(f'$\\sigma \\approx {sigma_5pct_analytical:.2f}$\n$P(|x|>3) = 5\\%$',
                xy=(sigma_5pct_analytical, 5.0), xytext=(sigma_5pct_analytical + 0.15, 12.0),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                fontweight='bold', fontsize=11)

    # Отметка для sigma = 1 (GAN baseline)
    ax.scatter([1.0], [p_sigma_1_theor * 100], color='#009E73', s=90, zorder=5)
    ax.annotate(f'$\\sigma = 1.0$ (GAN)\n$P = {p_sigma_1_theor*100:.2f}\\%$',
                xy=(1.0, p_sigma_1_theor * 100), xytext=(0.4, 8.0),
                arrowprops=dict(facecolor='#009E73', shrink=0.08, width=1, headwidth=6),
                color='#009E73', fontweight='bold', fontsize=11)

    ax.set_title("Зависимость вероятности выброса $P(|x| > 3)$ от стандартного отклонения $\\sigma$",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Стандартное отклонение $\\sigma$", fontsize=12)
    ax.set_ylabel("Вероятность выброса (%)", fontsize=12)
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(-1, 35)
    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    plot4_path = os.path.join(IMAGES_DIR, 'task2_tail_probability_vs_sigma.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"-> График 4 (Зависимость вероятности выброса от sigma) сохранен: {plot4_path}")

    # -------------------------------------------------------------
    # График 5: Сравнение плотностей и хвостов для sigma in [0.5, 1.0, 1.53, 2.0]
    # Иллюстрация выбросов для GAN
    # -------------------------------------------------------------
    sigmas_compare = [0.5, 1.0, round(sigma_5pct_analytical, 2), 2.0]
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x_range = np.linspace(-6, 6, 500)

    for idx, (s, col) in enumerate(zip(sigmas_compare, colors)):
        ax = axes[idx]
        pdf = stats.norm.pdf(x_range, loc=0, scale=s)
        p_out = 2 * stats.norm.sf(threshold / s)

        ax.plot(x_range, pdf, color=col, lw=2.5, label=f'$\\mathcal{{N}}(0, {s}^2)$')

        # Закрашивание хвостов |x| > 3
        tail_left = x_range <= -threshold
        tail_right = x_range >= threshold
        ax.fill_between(x_range[tail_left], pdf[tail_left], color='#E74C3C', alpha=0.5, label='Выбросы $|x| > 3$')
        ax.fill_between(x_range[tail_right], pdf[tail_right], color='#E74C3C', alpha=0.5)

        # Линии порога
        ax.axvline(-threshold, color='red', linestyle='--', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', alpha=0.7)

        ax.set_title(f"$\\sigma = {s}$  |  Вероятность выброса: {p_out * 100:.2f}%",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("x (шум генератора)")
        ax.set_ylabel("Плотность вероятности")
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 0.85)
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plot5_path = os.path.join(IMAGES_DIR, 'task2_distribution_tails_comparison.png')
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"-> График 5 (Сравнение хвостов распределений) сохранен: {plot5_path}")
    print()


if __name__ == '__main__':
    run_task_1()
    run_task_2()
    print("Лабораторная работа успешно выполнена!")
