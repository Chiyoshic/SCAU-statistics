import pandas as pd
from scipy import stats

df = pd.read_csv("4_fitness_data.csv", encoding="gbk")

required_cols = [
    "group",
    "weight_before",
    "weight_after",
    "exercise_freq",
    "weight_loss"
]

df = df.dropna(subset=required_cols)

alpha = 0.05

df["d"] = df["weight_after"] - df["weight_before"]

t_stat_1, p_value_1 = stats.ttest_rel(
    df["weight_after"],
    df["weight_before"],
    alternative="two-sided"
)

print("【任务1：配对样本假设检验】")
print(f"统计量 = {t_stat_1:.6f}")
print(f"p值 = {p_value_1:.10f}")

if p_value_1 < alpha:
    print("结论：拒绝H0，体重前后变化显著")
else:
    print("结论：不拒绝H0，体重前后变化不显著")

print()

group_A_loss = df[df["group"] == "A"]["weight_loss"]
group_B_loss = df[df["group"] == "B"]["weight_loss"]

var_A_loss = group_A_loss.var(ddof=1)
var_B_loss = group_B_loss.var(ddof=1)

n_A = len(group_A_loss)
n_B = len(group_B_loss)

F_stat_2 = var_A_loss / var_B_loss

dfn_2 = n_A - 1
dfd_2 = n_B - 1

p_value_F_2 = 2 * min(
    stats.f.cdf(F_stat_2, dfn_2, dfd_2),
    1 - stats.f.cdf(F_stat_2, dfn_2, dfd_2)
)

print("【任务2：A组体重下降量是否大于B组】")
print("方差齐性检验：")
print(f"F统计量 = {F_stat_2:.6f}")
print(f"p值 = {p_value_F_2:.10f}")

if p_value_F_2 < alpha:
    equal_var = False
    print("方差齐性结论：拒绝H0，认为两组方差不相等，使用 Welch t 检验")
else:
    equal_var = True
    print("方差齐性结论：不拒绝H0，认为两组方差相等，使用方差齐性的独立样本 t 检验")

# 单侧独立样本 t 检验
t_stat_2, p_value_2 = stats.ttest_ind(
    group_A_loss,
    group_B_loss,
    equal_var=equal_var,
    alternative="greater"
)

print("独立双样本单侧 t 检验：")
print(f"统计量 = {t_stat_2:.6f}")
print(f"p值 = {p_value_2:.10f}")

if p_value_2 < alpha:
    print("结论：拒绝H0，A组体重下降量显著大于B组")
else:
    print("结论：不拒绝H0，不能认为A组体重下降量显著大于B组")

print()

group_A_exercise = df[df["group"] == "A"]["exercise_freq"]
group_B_exercise = df[df["group"] == "B"]["exercise_freq"]

var_A_exercise = group_A_exercise.var(ddof=1)
var_B_exercise = group_B_exercise.var(ddof=1)

n_A_exercise = len(group_A_exercise)
n_B_exercise = len(group_B_exercise)

F_stat_3 = var_A_exercise / var_B_exercise

dfn_3 = n_A_exercise - 1
dfd_3 = n_B_exercise - 1

p_value_F_3 = 2 * min(
    stats.f.cdf(F_stat_3, dfn_3, dfd_3),
    1 - stats.f.cdf(F_stat_3, dfn_3, dfd_3)
)

print("【任务3：运动频率方差齐性检验】")
print(f"F统计量 = {F_stat_3:.6f}")
print(f"p值 = {p_value_F_3:.10f}")

if p_value_F_3 < alpha:
    print("结论：拒绝H0，A组和B组运动频率方差不相等")
else:
    print("结论：不拒绝H0，A组和B组运动频率方差没有显著差异")