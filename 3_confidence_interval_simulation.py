import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

# 总体参数
mu = 100
sigma = 15
N = 1000


# 任务1：单次区间估计

n = 30
sample = np.random.normal(loc=mu, scale=sigma, size=n)

x_bar = np.mean(sample)
s = np.std(sample, ddof=1)

# 已知总体方差：z 区间
z_crit = stats.norm.ppf(0.975)
z_margin = z_crit * sigma / np.sqrt(n)
z_ci = (x_bar - z_margin, x_bar + z_margin)

# 未知总体方差：t 区间
t_crit = stats.t.ppf(0.975, df=n - 1)
t_margin = t_crit * s / np.sqrt(n)
t_ci = (x_bar - t_margin, x_bar + t_margin)

print("任务1：单次抽样结果")
print(f"样本均值 x_bar = {x_bar:.4f}")
print(f"样本标准差 s = {s:.4f}")
print(f"已知方差 z 区间：{z_ci[0]:.4f}, {z_ci[1]:.4f}")
print(f"未知方差 t 区间：{t_ci[0]:.4f}, {t_ci[1]:.4f}")
print(f"z 区间是否包含真实均值 100：{z_ci[0] <= mu <= z_ci[1]}")
print(f"t 区间是否包含真实均值 100：{t_ci[0] <= mu <= t_ci[1]}")


# =========================
# 任务2：不同置信水平覆盖率
# =========================

def t_confidence_interval(sample, confidence_level):
    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)

    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin = t_crit * s / np.sqrt(n)
    return x_bar - margin, x_bar + margin


confidence_levels = [0.90, 0.95, 0.99]
coverage_results = []

for level in confidence_levels:
    count = 0

    for i in range(N):
        sample = np.random.normal(loc=mu, scale=sigma, size=n)
        lower, upper = t_confidence_interval(sample, level)

        if lower <= mu <= upper:
            count += 1

    coverage = count / N
    coverage_results.append(coverage)

print("\n任务2：不同置信水平下的覆盖率")
print("名义置信水平\t实际覆盖率")
for level, coverage in zip(confidence_levels, coverage_results):
    print(f"{level:.0%}\t\t{coverage:.4f}")


# 图1：置信水平 vs 覆盖率
plt.figure(figsize=(7, 5))
plt.plot(
    [level * 100 for level in confidence_levels],
    [coverage * 100 for coverage in coverage_results],
    marker='o',
    label="实际覆盖率"
)
plt.plot(
    [level * 100 for level in confidence_levels],
    [level * 100 for level in confidence_levels],
    linestyle='--',
    label="名义置信水平"
)
plt.xlabel("名义置信水平 / %")
plt.ylabel("实际覆盖率 / %")
plt.title("不同置信水平下的实际覆盖率")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 任务3：样本量影响分析
# =========================

sample_sizes = [5, 10, 30, 50, 100]
coverage_by_n = []

for n in sample_sizes:
    count = 0

    for i in range(N):
        sample = np.random.normal(loc=mu, scale=sigma, size=n)
        lower, upper = t_confidence_interval(sample, 0.95)

        if lower <= mu <= upper:
            count += 1

    coverage = count / N
    coverage_by_n.append(coverage)

print("\n任务3：不同样本量下的 95% t 区间覆盖率")
print("样本量 n\t实际覆盖率")
for n, coverage in zip(sample_sizes, coverage_by_n):
    print(f"{n}\t\t{coverage:.4f}")


# 图2：样本量 vs 覆盖率
plt.figure(figsize=(7, 5))
plt.plot(
    sample_sizes,
    [coverage * 100 for coverage in coverage_by_n],
    marker='o',
    label="实际覆盖率"
)
plt.axhline(y=95, linestyle='--', label="名义置信水平 95%")
plt.xlabel("样本量 n")
plt.ylabel("实际覆盖率 / %")
plt.title("样本量对 95% t 区间覆盖率的影响")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 任务4：非正态总体下的稳健性
# =========================

def simulate_non_normal(distribution="exponential", n=30, level=0.95, N=1000):
    count = 0

    for i in range(N):
        if distribution == "exponential":
            # 指数分布均值为100
            sample = np.random.exponential(scale=100, size=n)
            true_mean = 100

        elif distribution == "uniform":
            # 均匀分布 U(70,130)，均值为100
            sample = np.random.uniform(low=70, high=130, size=n)
            true_mean = 100

        lower, upper = t_confidence_interval(sample, level)

        if lower <= true_mean <= upper:
            count += 1

    return count / N


exp_coverage = simulate_non_normal("exponential", n=30, level=0.95, N=N)
uni_coverage = simulate_non_normal("uniform", n=30, level=0.95, N=N)

print("\n任务4：非正态总体下的 95% t 区间覆盖率")
print(f"指数分布总体：{exp_coverage:.4f}")
print(f"均匀分布总体：{uni_coverage:.4f}")