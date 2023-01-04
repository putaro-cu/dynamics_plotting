import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
from dual import *


def phi_i(S, A, m, f_i):  # セクハラの適応度コスト関数
    return 1 - (f_i * m * S) / ((1 - m) * S + A)


def f_density(X, f_s, f_a, b, m, d, c_s, c_a):  # 有性型, 無性型の密度に関する微分方程式
    # X=[density of S, A]

    dXdt = np.array([
        phi_i(X[0], X[1], m, f_s) * b * (1 - m) * X[0] - d * (X[0] + c_a * X[1]) * X[0],
        phi_i(X[0], X[1], m, f_a) * b * X[1] - d * (X[1] + c_s * X[0]) * X[1]
    ])
    return dXdt


def func(t, X, f_s, f_a, b, m, d, c_s, c_a):  # 微分方程式をsolve_ivp()の形式に変換
    return f_density(X, f_s, f_a, b, m, d, c_s, c_a)


def calc_trace_determinant(x, y):
    det = fx(x, y) * gy(x, y) - fy(x, y) * gx(x, y)
    tr = fx(x, y) + gy(x, y)
    return tr, det


def fx(x, y):
    return dual(f(dual(x, 1), y) - f(x, y)).im


def fy(x, y):
    return dual(f(x, dual(y, 1)) - f(x, y)).im


def gx(x, y):
    return dual(g(dual(x, 1), y) - g(x, y)).im


def gy(x, y):
    return dual(g(x, dual(y, 1)) - g(x, y)).im


def f():
    return 0


def g():
    return 0


def stable(x, y):
    tr, det = calc_trace_determinant(x, y)  # ヤコビ行列の行列式とトレースを計算

    return tr < 0 and det > 0


def fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, ttl):
    global f, g
    X0 = np.array([150, 150])

    def f(s, a):
        """
        dS/dt = f(x,y)
        """
        return f_density([s, a], f_s, f_a, b, m, d, c_s, c_a)[0]

    def g(s, a):
        """
        dA/dt = g(x,y)
        """
        return f_density([s, a], f_s, f_a, b, m, d, c_s, c_a)[1]

    eq = optimize.root(f_density, X0, args=(f_s, f_a, b, m, d, c_s, c_a), method='hybr')
    eq2 = optimize.root(f_density, np.array([50, 10]), args=(f_s, f_a, b, m, d, c_s, c_a), method='hybr')

    eqs = np.array([(1 - m - m * f_s) * b / d, 0])
    eqa = np.array([0, b / d])
    A, S = np.meshgrid(np.linspace(amin, amax, 1000), np.linspace(smin, smax, 1000))
    dS, dA = f_density([S, A], f_s, f_a, b, m, d, c_s, c_a)

    figure = plt.figure(figsize=(3, 3))  # 図のサイズ設定
    plt.axis([amin, amax, smin, smax])

    plt.streamplot(A, S, dA, dS, density=0.6)  # ベクトルの描画

    cntr_s = plt.contour(A, S, f_density([S, A], f_s, f_a, b, m, d, c_s, c_a)[0], [0], colors="r")  # ヌルクラインの描画
    cntr_a = plt.contour(A, S, f_density([S, A], f_s, f_a, b, m, d, c_s, c_a)[1], [0], colors="g")

    plt.plot(eq.x[1], eq.x[0], marker='.', markersize=15,
             color='black' if stable(eq.x[0], eq.x[1]) else 'white',
             markeredgecolor="black", clip_on=False)
    plt.plot(eq2.x[1], eq2.x[0], marker='.', markersize=15,
             color='white', markeredgecolor="black", clip_on=False)
    plt.plot(0, 0, marker='.', markersize=15,
             color='white', markeredgecolor="black", clip_on=False)
    plt.plot(eqs[1], eqs[0], marker='.', markersize=15, color='black' if stable(eqs[0], eqs[1]) else 'white',
             markeredgecolor="black", clip_on=False)
    plt.plot(eqa[1], eqa[0], marker='.', markersize=15, color='black' if stable(eqa[0], eqa[1]) else 'white',
             markeredgecolor="black", clip_on=False)

    print("(S*, A*)=", eq.x, "Stable?:", stable(eq.x[0], eq.x[1]))
    print("(S*, A*)=", eq2.x)
    print("(S*, A*)=", eqs, "Stable?:", stable(eqs[0], eqs[1]))
    print("(S*, A*)=", eqa, "Stable?:", stable(eqa[0], eqa[1]))

    plt.ylabel("density of sexual individuals [S]")
    plt.xlabel("density of asexual individuals [A]")

    handle_s, _ = cntr_s.legend_elements()
    handle_a, _ = cntr_a.legend_elements()
    handle_p = plt.plot(-20, -20, marker='.', markersize=15,
                        color='black', markeredgecolor="black")
    handle_up = plt.plot(-20, -20, marker='.', markersize=15,
                         color='white', markeredgecolor="black")

    plt.title(f"({ttl})  " + r"$m=$" + f"{m}, " + r"$c_s=$" + f"{c_s}, " + r"$c_a=$" + f"{c_a}", loc="left")

    plt.savefig(f"fig{ttl}.png", format="png", bbox_inches="tight", pad_inches=0.1)
    plt.show()

    return figure

def main():
    f_s = 0.1
    f_a = 0.9
    b = 2
    m = 0.3
    d = 0.01
    c_s = 0.3
    c_a = 0.3
    smin, smax = 0.1, 260
    amin, amax = 0.1, 260

    plta = fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, "a")
    c_s = 0.3
    c_a = 1
    pltb = fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, "b")
    c_s = 1
    c_a = 0.3
    smin, smax = 0.1, 160
    pltc = fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, "c")
    c_s = 0.3
    c_a = 0.3
    m = 0.6
    smin, smax = 0.1, 100
    pltd = fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, "d")
    c_s = 1
    c_a = 1
    m = 0.8
    plte = fig(f_s, f_a, b, m, d, c_s, c_a, smin, smax, amin, amax, "e")


main()
