from settings import EPSILON_END, EPSILON_START, EPSILON_STEPS


def epsilon(t):
    eps_frac = (EPSILON_END - EPSILON_START) / EPSILON_STEPS
    return max(EPSILON_END, EPSILON_START + t * eps_frac)


for i in range(0, 550000, 5000):
    print(f"i = {i} => epsilon = {epsilon(i)}")
