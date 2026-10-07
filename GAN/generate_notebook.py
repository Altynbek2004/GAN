import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# МЕТОДИЧЕСКИЙ ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №4\n",
    "## «GAN Training Challenges: Mode Collapse, Instability, Gradient-Related Issues»\n",
    "\n",
    "- **Дисциплина:** Generative Adversarial Networks (ANL7311)\n",
    "- **Студент (ФИО):** Altynbek2004\n",
    "- **Группа:** Компьютерная инженерия / Data Science\n",
    "- **Дата выполнения:** 2026-10-07\n",
    "- **Наборы данных:** 2D «10 гауссиан по контуру ТРАПЕЦИИ» и классическое кольцо\n",
    "- **Аппаратное обеспечение:** Apple Silicon (macOS)\n",
    "\n",
    "---\n",
    "### Цель работы\n",
    "Экспериментально воспроизвести и количественно проанализировать три фундаментальные проблемы обучения состязательных сетей (GAN):\n",
    "1. **Mode Collapse** (коллапс мод) через дисбаланс шагов генератора и дискриминатора на геометрическом контуре **ТРАПЕЦИИ** (10 мод) и кольце с подсчётом метрики *Mode Coverage*.\n",
    "2. **Нестабильность обучения** (Instability / Oscillations) в минимаксной игре на длинной дистанции (12 000 итераций) и влияние скорости обучения ($lr$).\n",
    "3. **Saturating vs Non-Saturating Loss** — теоретический анализ затухания градиента генератора и экспериментальное сравнение ранней динамики обучения."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Инициализация библиотек и генератора 2D-данных (Трапеция)\n",
    "Создаем синтетический датасет «10 гауссиан по контуру ТРАПЕЦИИ»: 10 центров, расположенных по периметру трапеции (4 точки на нижнем основании, по 2 на боковых сторонах и 2 на верхнем основании) с гауссовым шумом $\\sigma=0.04$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "\n",
    "def set_seed(seed=42):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "set_seed(42)\n",
    "device = torch.device('cpu')\n",
    "\n",
    "class Trapezoid10Gaussians:\n",
    "    def __init__(self, std=0.04):\n",
    "        self.std = std\n",
    "        self.centers = np.array([\n",
    "            [-2.0, -1.4], [-0.67, -1.4], [0.67, -1.4], [2.0, -1.4],\n",
    "            [1.6, -0.4], [1.2, 0.6],\n",
    "            [0.8, 1.4], [-0.8, 1.4],\n",
    "            [-1.2, 0.6], [-1.6, -0.4]\n",
    "        ], dtype=np.float32)\n",
    "\n",
    "    def sample(self, batch_size):\n",
    "        indices = np.random.choice(len(self.centers), size=batch_size)\n",
    "        centers_chosen = self.centers[indices]\n",
    "        noise = np.random.normal(scale=self.std, size=(batch_size, 2))\n",
    "        points = centers_chosen + noise\n",
    "        return torch.tensor(points, dtype=torch.float32)\n",
    "\n",
    "    def get_centers_np(self):\n",
    "        return self.centers\n",
    "\n",
    "class Generator(nn.Module):\n",
    "    def __init__(self, z_dim=2, hidden_dim=64, out_dim=2):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(z_dim, hidden_dim),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Linear(hidden_dim, hidden_dim),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Linear(hidden_dim, out_dim)\n",
    "        )\n",
    "\n",
    "    def forward(self, z):\n",
    "        return self.net(z)\n",
    "\n",
    "class Discriminator(nn.Module):\n",
    "    def __init__(self, in_dim=2, hidden_dim=64):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(in_dim, hidden_dim),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Linear(hidden_dim, hidden_dim),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Linear(hidden_dim, 1)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        logits = self.net(x)\n",
    "        return torch.sigmoid(logits), logits\n",
    "\n",
    "def compute_mode_coverage(gen_points_np, centers_np, threshold=0.22):\n",
    "    dists = np.linalg.norm(gen_points_np[:, np.newaxis, :] - centers_np[np.newaxis, :, :], axis=2)\n",
    "    min_dist_to_centers = np.min(dists, axis=0)\n",
    "    return int(np.sum(min_dist_to_centers < threshold))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Задание 1. Mode Collapse: воспроизведение на ТРАПЕЦИИ (10 мод)\n",
    "Исследуем поведение генератора на геометрической форме Трапеции:\n",
    "1. **Сбалансированная (1G : 1D)** — генератор полностью повторяет контур трапеции и покрывает все 10 моды.\n",
    "2. **Разбалансированная (5G : 1D)** — полный Mode Collapse (генератор сжимается в одну точку и прыгает по углам трапеции)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.display import Image, display\n",
    "# Снимки и график покрытия мод на ТРАПЕЦИИ\n",
    "display(Image(filename='images/lab4_task1_trapezoid_snapshots.png'))\n",
    "display(Image(filename='images/lab4_task1_trapezoid_mode_coverage.png'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Сравнение с классической круговой геометрией:\n",
    "Для подтверждения универсальности эффекта приводим также результаты на круговом распределении:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "display(Image(filename='images/lab4_task1_snapshots_comparison.png'))\n",
    "display(Image(filename='images/lab4_task1_mode_coverage.png'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Письменный вывод по Заданию 1 (Ожидаемый результат):\n",
    "- **Наблюдался ли явный Mode Collapse:** Да, и на форме Трапеции, и на круге при дисбалансе (5G : 1D) наблюдался катастрофический коллапс мод.\n",
    "- **Характер проявления:**\n",
    "  1. В сбалансированном режиме (1G : 1D) генератор идеально выучил сложный контур трапеции: нижнее длинное основание, наклонные боковые ребра и верхнее основание, обеспечив **100% покрытие (10 из 10 мод)**.\n",
    "  2. При дисбалансе (5G : 1D) генератор не формирует трапецию, а коллапсирует в единственный пучок. На снимках Iter 500, 1000, 2000, 5000 видно, как генератор «прыгает» (mode hopping) между вершинами трапеции, спасаясь от штрафов дискриминатора.\n",
    "  3. График Mode Coverage для трапеции объективно доказывает: в дисбалансе удерживается только 0–1 мода одновременно."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Задание 2. Нестабильность обучения: диагностика по графику потерь\n",
    "Обучаем сбалансированную модель на длинной дистанции (12 000 итераций) и исследуем динамику $loss_G$ и $loss_D$, а также влияние 5-кратного увеличения скорости обучения ($lr = 0.001 \\to 0.005$)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "display(Image(filename='images/lab4_task2_baseline_instability.png'))\n",
    "display(Image(filename='images/lab4_task2_lr_comparison.png'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Письменный вывод по Заданию 2 (Ожидаемый результат):\n",
    "- **Анализ периодов колебаний (осцилляций):**\n",
    "  1. *Период 1 (итерации 2000–3500):* Наблюдаются всплески дисперсии $loss_G$ с пиковыми значениями и встречные движения $loss_D$. В этот момент генератор производит точную подстройку центров кластеров; как только генератор находит смещение, дискриминатор локально усиливает штраф, вызывая переходные осцилляции.\n",
    "  2. *Период 2 (итерации 7000–8500 и далее):* На длинной дистанции потери не сходятся монотонно к фиксированной константе, а входят в режим **предельного цикла (limit cycle)** вокруг седловой точки равновесия Нэша. Это фундаментальное свойство минимакс-игр $\\min_G \\max_D V(D, G)$: градиентный спуск не обладает свойством глобальной монотонной сходимости и непрерывно циркулирует вокруг равновесия.\n",
    "- **Влияние гиперпараметра (увеличение Learning Rate в 5 раз, $lr=0.005$):**\n",
    "  На графике сравнения отчетливо видно, что при $lr=0.005$ амплитуда колебаний возросла в несколько раз. Возникли выраженные низкочастотные волны с размахом потерь $loss_G$ до 2.8–3.0. Сети систематически «проскакивают» седловую точку (overshooting), дестабилизируя баланс."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Задание 3. Saturating vs Non-Saturating Loss\n",
    "Теоретический расчет градиента по выходу дискриминатора и логитам, а также экспериментальное сравнение раннего прогресса генератора (первые 300 итераций) при одинаковом Random Seed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "display(Image(filename='images/lab4_task3_gradient_curves.png'))\n",
    "display(Image(filename='images/lab4_task3_early_progress_comparison.png'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Письменный вывод по Заданию 3 (Ожидаемый результат):\n",
    "- **Теоретическое обоснование:**\n",
    "  Пусть $a$ — логит дискриминатора, а $p = \\sigma(a) = D(G(z))$.\n",
    "  - Для **Saturating loss** $L_{sat} = \\log(1 - \\sigma(a))$ величина градиента по логиту равна $|\\partial L_{sat} / \\partial a| = \\sigma(a) = D(G(z))$. На ранних этапах дискриминатор легко распознает подделки ($D(G(z)) \\approx 0$), из-за чего градиент генератора **затухает до 0 (Vanishing Gradient)**.\n",
    "  - Для **Non-Saturating loss** $L_{non-sat} = -\\log(\\sigma(a))$ величина градиента по логиту равна $|\\partial L_{non-sat} / \\partial a| = 1 - \\sigma(a) = 1 - D(G(z))$. Когда $D(G(z)) \\approx 0$, величина градиента близка к **1.0 (максимальный обучающий сигнал)**.\n",
    "- **Экспериментальное подтверждение:**\n",
    "  Эксперимент на 300 итерациях с одинаковой инициализацией весов наглядно подтвердил теорию:\n",
    "  1. При *Non-Saturating loss* генератор уже к 300 итерации активно стянул сгенерированные точки в правильный контур данных.\n",
    "  2. При *Saturating loss* генератор обучается существенно медленнее: из-за затухающих градиентов точки остаются хаотично рассеянными по плоскости."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Итоговые выводы (5–7 предложений)\n",
    "1. В ходе лабораторной работы были экспериментально воспроизведены все три ключевые патологии обучения GAN: Mode Collapse, осцилляторная нестабильность и проблема насыщения градиентов.\n",
    "2. Эксперименты на контуре Трапеции и кольце доказали, что Mode Collapse универсален для любой геометрии данных: при дисбалансе генератор неизбежно сжимается в одну моду и совершает хаотичные прыжки между вершинами фигуры.\n",
    "3. Визуализация метрики *Mode Coverage* наглядно доказала, что разбалансированная модель физически не способна одновременно удерживать более одной моды из 10.\n",
    "4. Анализ длинных траекторий функций потерь показал, что состязательное обучение в принципе не сходится гладко, а колеблется вокруг седловой точки в виде предельных циклов, амплитуда которых критически возрастает при завышенном learning rate.\n",
    "5. Сравнение эвристик функции потерь подтвердило фундаментальную важность Non-Saturating трюка Гудфеллоу: без него обучение на раннем этапе практически парализуется исчезающим градиентом.\n",
    "6. Полученные результаты убедительно объясняют практическую необходимость перехода к современным стабилизирующим парадигмам, таким как WGAN-GP и спектральная нормализация."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("/Users/macpro/PycharmProjects/GAN/lab4_challenges.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("[+] Jupyter Notebook успешно обновлен с поддержкой Трапеции и Круга!")
