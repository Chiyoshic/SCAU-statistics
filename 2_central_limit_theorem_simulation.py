import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 第一步：构造总体
N = 100000
population = np.arange(1, N + 1)  # 均匀分布 1~100000
mu = np.mean(population)
sigma2 = np.var(population)

print("总体均值:", mu)
print("总体方差:", sigma2)

# 第二步：设置参数
sample_sizes = [2, 4, 10, 15, 30]
m = 10000  # 抽样次数

results = {}

for n in sample_sizes:
    sample_means = []

    # 第三步：重复抽样
    for _ in range(m):
        sample = np.random.choice(population, size=n, replace=True)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # 第四步：计算均值与方差
    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means)

    results[n] = (mean_of_means, var_of_means)

    print(f"\nn = {n}")
    print("样本均值的均值:", mean_of_means)
    print("样本均值的方差:", var_of_means)
    print("理论方差 σ²/n:", sigma2 / n)

    # 第五步：画图
    plt.figure()
    plt.hist(sample_means, bins=50, density=True)
    plt.title(f"n = {n} 时样本均值分布")
    plt.xlabel("样本均值")
    plt.ylabel("密度")
    plt.show()