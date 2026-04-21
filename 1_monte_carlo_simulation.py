import random

def simulate(n=1000000):
    success = 0

    for _ in range(n):
        # 6个人抛硬币：0或1
        tosses = [random.randint(0, 1) for _ in range(6)]

        # 统计0和1的个数
        count_0 = tosses.count(0)
        count_1 = tosses.count(1)

        # 判断是否“单人出局”
        # 即：1个0 + 5个1 或 1个1 + 5个0
        if count_0 == 1 or count_1 == 1:
            success += 1

    # 概率估计
    return success / n


if __name__ == "__main__":
    probability = simulate(1000000)
    print(f"模拟得到的概率约为: {probability:.6f}")