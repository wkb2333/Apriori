"""Topic 3 讲义讲解：OLS 假设 A1 与 FWL 定理。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Topic 3 · OLS 与 FWL", page_icon="📘", layout="wide")

st.markdown("# Topic 3 讲义讲解")
st.caption("基于上传讲义 Topic3：在假设 A1 下，OLS 解的存在唯一性、关键矩阵，以及 FWL 定理。")

section = st.sidebar.radio(
    "章节导航",
    [
        "总览",
        "假设 A1",
        "简单 / 最简情形",
        "计算性质：OLS vs LAD",
        "关键表达式与矩阵",
        "FWL 定理",
        "小结",
    ],
)


def _rank_and_invertible(X: np.ndarray) -> tuple[int, bool]:
    rank = int(np.linalg.matrix_rank(X))
    k = X.shape[1]
    return rank, rank == k


if section == "总览":
    st.markdown(
        """
## 这节课在讲什么？

给定数据抽取 $\\{y, X\\}$，其中 $y$ 是 $S\\times 1$ 被解释变量向量，$X$ 是 $S\\times k$ 解释变量矩阵。
我们想用 $X$ 的列的线性组合去拟合 $y$，即求 OLS：

$$
\\hat{b}_{\\mathrm{OLS}}
= \\arg\\min_b\\ (y - Xb)^\\top (y - Xb)
= (X^\\top X)^{-1} X^\\top y
$$

本讲**只依赖假设 A1**，回答三件事：

1. **何时** OLS 解存在且唯一？
2. 在此条件下，拟合值、残差、投影矩阵有哪些**代数结构**？
3. 若只关心部分系数，Frisch–Waugh–Lovell（**FWL**）定理给出哪三种等价算法，以及它们如何体现“**其他条件不变**”？

> 注意：本节尚未讨论无偏性、一致性、渐近正态等统计性质；重点是代数与可计算性。
"""
    )
    st.info("请用左侧「章节导航」逐节阅读；若干节含可调参数的小实验。")

elif section == "假设 A1":
    st.markdown(
        """
## 假设 A1

**A1.** $X$ 是 $S\\times k$ 矩阵，且 $\\mathrm{rank}(X)=k<S$。

### 直觉
A1 排除了解释变量之间的**完全线性关系**（完全共线性）。

### 后果
- $X^\\top X$ 正定，因此 $(X^\\top X)^{-1}$ 存在；
- OLS 解存在且**唯一**；
- A1 是 $\\hat{b}_{\\mathrm{OLS}}$ 存在且唯一的**充分必要条件**。
"""
    )

    st.markdown("### 互动：完全共线性 vs 近似共线性")
    mode = st.radio(
        "构造解释变量矩阵 $X$（含截距列）",
        ["完全共线性（违反 A1）", "近似共线性（不违反 A1）", "正常变化（满足 A1）"],
        horizontal=True,
    )
    S = st.slider("样本量 S", min_value=5, max_value=30, value=10)

    ones = np.ones((S, 1))
    if mode.startswith("完全"):
        x2 = np.full((S, 1), 7.0)
        note = "两列满足 $x_{s1}=\\frac{1}{7}x_{s2}$，完全共线 → 违反 A1。"
    elif mode.startswith("近似"):
        x2 = np.array([[7.001], [6.999]] + [[7.0]] * (S - 2))
        note = "存在**近似**共线，但 $\\mathrm{rank}(X)=2$，不违反 A1。"
    else:
        rng = np.random.default_rng(0)
        x2 = rng.normal(loc=7.0, scale=1.5, size=(S, 1))
        note = "第二列有足够变异，样本方差 $>0$，满足 A1。"

    X = np.hstack([ones, x2])
    rank, ok = _rank_and_invertible(X)
    var_x2 = float(np.var(x2, ddof=1)) if S > 1 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("rank(X)", rank)
    c2.metric("A1 是否成立", "是" if ok else "否")
    c3.metric("sample var(x₂)", f"{var_x2:.4f}")

    st.write(pd.DataFrame(X, columns=["截距 (=1)", "x₂"]).head(8))
    st.markdown(note)

    if ok:
        b = np.linalg.inv(X.T @ X) @ X.T @ (np.arange(S, dtype=float))
        st.success(f"可求唯一 OLS（对演示用的 y=0,1,…,S-1）：b̂ = {np.round(b, 4)}")
    else:
        st.error("XᵀX 不可逆，OLS 公式 $(X^\\top X)^{-1}X^\\top y$ 不成立。")

elif section == "简单 / 最简情形":
    st.markdown(
        """
## “简单”情形（$k=2$，含截距）

$$
X=
\\begin{bmatrix}
1 & x_{12}\\\\
1 & x_{22}\\\\
\\vdots & \\vdots\\\\
1 & x_{S2}
\\end{bmatrix}
$$

此时 A1 要求 $S\\ge 3$，且等价于：

- 第二列样本方差 $>0$，或
- $\\min_s x_{s2} < \\max_s x_{s2}$。

若所有 $x_{s2}$ 都等于同一个常数（例如全是 7），则两列完全共线，A1 失败。

讲义提醒：即便 A1 成立、**可以**做 OLS，也不等于**应该**做 OLS（例如 Anscombe 四重奏第二组数据满足 A1，但线性拟合未必合适）。

---

## “最简”情形（仅截距，$k=1$）

$$
X=\\iota_S=
\\begin{bmatrix}1\\\\1\\\\\\vdots\\\\1\\end{bmatrix}
$$

A1 要求 $S\\ge 2$。此时：

$$
\\hat{b}_{\\mathrm{OLS}}=\\bar{y}=\\frac{1}{S}\\sum_{s=1}^{S} y_s
$$

而最小绝对偏差（LAD）给出：

$$
\\hat{b}_{\\mathrm{LAD}}=\\mathrm{median}(y)
$$
"""
    )

    st.markdown("### 互动：仅截距模型")
    raw = st.text_input("输入一串 y（逗号分隔）", "1, 2, 3, 100")
    try:
        y = np.array([float(v.strip()) for v in raw.split(",") if v.strip()], dtype=float)
    except ValueError:
        st.error("请输入合法数字。")
        st.stop()

    if y.size < 2:
        st.warning("最简情形下 A1 需要 S ≥ 2。")
    else:
        ols = float(np.mean(y))
        lad = float(np.median(y))
        c1, c2 = st.columns(2)
        c1.metric("OLS（样本均值）", f"{ols:.4f}")
        c2.metric("LAD（样本中位数）", f"{lad:.4f}")
        st.caption("异常值会强烈拉动均值，中位数更稳健——但计算更贵（见下一节）。")

elif section == "计算性质：OLS vs LAD":
    st.markdown(
        """
## 统计性质之外：计算性质也很关键

评价计量方法时，人们常谈无偏、一致、有效、渐近正态等**统计性质**。
但若没有可计算性，再漂亮的统计性质也难以落地。实用价值取决于二者的交织。

在仅截距模型中：

| 方法 | 目标函数 | 解 | 计算直觉 |
|------|----------|----|----------|
| OLS | $\\sum (y_s-b)^2$ | 样本均值 | 求和再除以 $S$ |
| LAD | $\\sum \\lvert y_s-b\\rvert$ | 样本中位数 | 需要排序找中间位置 |

结论：LAD 通常比 OLS（以及 LQD 等可微最小距离方法）**计算上难得多**。

### 简史（讲义时间线）
- **1950s–60s**：计量“石器时代”——滑尺手算回归（纪念 Richard Stone）。
- **1981**：个人电脑出现。
- **1990s 末**：多 megaflop；如今 gigaflop / 更高，量子计算仍偏理论。

硬件边界一直在重塑“什么方法算得出来”，从而影响计量方法论的前沿。
"""
    )

elif section == "关键表达式与矩阵":
    st.markdown(
        """
## A1 下的关键表达式

残差向量 $\\hat{e}(b)\\equiv y-\\hat{y}(b)$。在 OLS 解处：

$$
\\begin{aligned}
\\hat{b}_{\\mathrm{OLS}} &= (X^\\top X)^{-1}X^\\top y \\\\
\\hat{y}_{\\mathrm{OLS}} &= X\\hat{b}_{\\mathrm{OLS}} = X(X^\\top X)^{-1}X^\\top y \\\\
\\hat{e}_{\\mathrm{OLS}} &= y-\\hat{y}_{\\mathrm{OLS}} = \\big(I_S - X(X^\\top X)^{-1}X^\\top\\big)y
\\end{aligned}
$$

## 三个核心矩阵

| 矩阵 | 定义 | 作用 |
|------|------|------|
| $A_X$ | $(X^\\top X)^{-1}X^\\top$（$k\\times S$） | $\\hat{b}_{\\mathrm{OLS}}=A_X y$ |
| $P_X$ | $X(X^\\top X)^{-1}X^\\top$（$S\\times S$） | **投影矩阵**：$\\hat{y}=P_X y$ |
| $M_X$ | $I_S-P_X$（$S\\times S$） | **残差制造矩阵**：$\\hat{e}=M_X y$ |

### $P_X$ 与 $M_X$ 的性质
二者都是实对称、幂等矩阵：

- $P_X^\\top=P_X$，且 $P_X P_X=P_X$，且 $\\mathrm{rank}(P_X)=k$
- $M_X^\\top=M_X$，且 $M_X M_X=M_X$，且 $\\mathrm{rank}(M_X)=S-k$

$S-k$ 就是这次拟合的**自由度**。

几何图像：$P_X$ 把 $y$ 正交投影到 $X$ 的列空间；投影与 $y$ 之间的差向量正是残差 $M_X y$。

### 秩为何等于迹？
对称幂等矩阵的特征值只能是 0 或 1，故秩 = 非零特征值个数 = 迹：

$$
\\mathrm{rank}(P_X)=\\mathrm{tr}(P_X)=k,\\qquad
\\mathrm{rank}(M_X)=\\mathrm{tr}(M_X)=S-k.
$$

二者都是**秩亏**（奇异）的 $S\\times S$ 矩阵。
"""
    )

    st.markdown("### 互动：验证幂等与秩")
    S = st.slider("S", 6, 20, 10, key="mat_S")
    k = st.slider("k", 1, 4, 2, key="mat_k")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(S, k))
    # 保证满列秩
    while np.linalg.matrix_rank(X) < k:
        X = rng.normal(size=(S, k))

    XtX_inv = np.linalg.inv(X.T @ X)
    P = X @ XtX_inv @ X.T
    M = np.eye(S) - P
    y = rng.normal(size=S)
    yhat = P @ y
    ehat = M @ y

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("‖P²−P‖", f"{np.linalg.norm(P @ P - P):.2e}")
    m2.metric("‖M²−M‖", f"{np.linalg.norm(M @ M - M):.2e}")
    m3.metric("rank(P) / tr(P)", f"{np.linalg.matrix_rank(P)} / {np.trace(P):.2f}")
    m4.metric("rank(M) / tr(M)", f"{np.linalg.matrix_rank(M)} / {np.trace(M):.2f}")
    st.write("残差与拟合值应近似正交：", f"êᵀ ŷ = {float(ehat @ yhat):.2e}")

elif section == "FWL 定理":
    st.markdown(
        """
## Frisch–Waugh–Lovell（FWL）定理

把 $X=[X_A\\ \\cdots\\ X_B]$ 分块，维数 $k=k_A+k_B$。我们只关心子向量 $\\hat{b}_{A,\\mathrm{OLS}}$。
定义

$$
M_{X_B}=I_S-X_B(X_B^\\top X_B)^{-1}X_B^\\top.
$$

它对任意向量 $z$ 作用后，$M_{X_B}z$ 就是把 $z$ 对 $X_B$ 回归后的残差——即 $z$ 中**不能被 $X_B$ 解释**的部分。

### 三条等价路径

1. **蛮力**：用全部 $X_A,X_B$ 对 $y$ 做 OLS，再只读取 $A$ 块系数。  
2. **只清洗右侧**：用 $\\tilde{X}=M_{X_B}X_A$ 对 $y$ 回归。  
3. **左右都清洗**：用 $\\tilde{X}=M_{X_B}X_A$ 对 $\\tilde{y}=M_{X_B}y$ 回归。

三者给出同一个 $\\hat{b}_{A,\\mathrm{OLS}}$。这正是 OLS“**其他条件不变（ceteris paribus）**”的代数表达：要么显式控制 $X_B$，要么事先把 $X_B$ 的变动从变量中剔除。

### 关于 A1 的细节
$M_{X_B}$ 在 A1 下有定义，因为 $\\mathrm{rank}(X_B)=k_B$ 是 $\\mathrm{rank}(X)=k$ 的必要条件。
但 $\\mathrm{rank}(X_A)=k_A$ 且 $\\mathrm{rank}(X_B)=k_B$ **不足以**保证 A1：例如 $X_A$ 含截距、$X_B$ 是全 7 的常数列，则子块各自满秩，拼起来却共线。

### 实用价值
FWL 不只是计算技巧；在许多情形下 $\\tilde{y}$、$\\tilde{X}_A$ 有直接含义，不必显式求 $(X_B^\\top X_B)^{-1}$。常见应用：

1. 用季节虚拟变量做**去季节**；
2. 含截距时的**去均值**；
3. 时间趋势下的**去趋势**；
4. 面板个体截距下的 **within 变换**。
"""
    )

    st.markdown("### 互动：验证三种路径得到同一系数")
    S = st.slider("样本量 S", 20, 80, 40, key="fwl_S")
    rng = np.random.default_rng(7)
    xA = rng.normal(size=(S, 1))
    xB = rng.normal(size=(S, 1))
    # 制造相关：让 y 依赖两者
    y = 2.0 * xA[:, 0] - 1.5 * xB[:, 0] + rng.normal(scale=0.3, size=S)
    XA = np.hstack([np.ones((S, 1)), xA])
    XB = xB
    X = np.hstack([XA, XB])

    # Option 1
    b_full = np.linalg.inv(X.T @ X) @ X.T @ y
    bA_1 = b_full[: XA.shape[1]]

    # Option 2 & 3
    MXB = np.eye(S) - XB @ np.linalg.inv(XB.T @ XB) @ XB.T
    Xtilde = MXB @ XA
    ytilde = MXB @ y
    bA_2 = np.linalg.inv(Xtilde.T @ Xtilde) @ Xtilde.T @ y
    bA_3 = np.linalg.inv(Xtilde.T @ Xtilde) @ Xtilde.T @ ytilde

    df = pd.DataFrame(
        {
            "系数": ["截距", "xA"],
            "路径1 全回归": np.round(bA_1, 6),
            "路径2 只洗 RHS": np.round(bA_2, 6),
            "路径3 左右都洗": np.round(bA_3, 6),
        }
    )
    st.dataframe(df, use_container_width=True)
    st.success(
        f"最大两两差异 ≈ {max(np.max(np.abs(bA_1-bA_2)), np.max(np.abs(bA_1-bA_3)), np.max(np.abs(bA_2-bA_3))):.2e}"
    )

elif section == "小结":
    st.markdown(
        """
## 仅在 A1 下我们已经得到的结论

1. OLS 解 $\\hat{b}_{\\mathrm{OLS}}=A_X y$ **良定义**；
2. 拟合值 $\\hat{y}_{\\mathrm{OLS}}=P_X y$ **良定义**；
3. 残差 $\\hat{e}_{\\mathrm{OLS}}=M_X y$ **良定义**；
4. 最小残差平方和 $\\mathrm{RSS}=\\hat{e}^\\top\\hat{e}=y^\\top M_X y$ **良定义**；
5. 始终可以做分块回归：由 FWL，

$$
\\hat{b}_{\\mathrm{OLS}}
=
\\begin{bmatrix}
(X_A^\\top M_{X_B} X_A)^{-1} X_A^\\top M_{X_B} y \\\\
(X_B^\\top M_{X_A} X_B)^{-1} X_B^\\top M_{X_A} y
\\end{bmatrix}
=
\\begin{bmatrix}
\\hat{b}_{A,\\mathrm{OLS}} \\\\
\\hat{b}_{B,\\mathrm{OLS}}
\\end{bmatrix}.
$$

### 一句话记忆
**A1 = 满列秩 → 投影几何成立 → FWL 把“控制变量”变成“先残差化再回归”。**
"""
    )
    st.balloons()
