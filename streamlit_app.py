import streamlit as st

st.set_page_config(page_title="课程主页", page_icon="🏠", layout="wide")

st.markdown("# 课程主页")
st.markdown(
    """
欢迎。本应用以 **Topic 3 讲义讲解** 为主页内容入口，并保留原有的 Apriori 关联规则演示。

使用左侧边栏的页面切换：
"""
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 📘 Topic 3 · OLS 与 FWL")
    st.markdown(
        """
讲义核心（仅依赖假设 **A1**）：

- 何时 OLS 解存在且唯一  
- 投影矩阵 $P_X$、残差矩阵 $M_X$  
- Frisch–Waugh–Lovell 定理的三种等价算法  

请打开侧栏页面：**Topic3 讲义讲解**。
"""
    )

with c2:
    st.markdown("### 🛒 Apriori 关联规则（原应用）")
    st.markdown(
        """
基于 Agrawal & Srikant (1994) 的频繁项集 / 关联规则演示。

请打开侧栏页面：**Apriori 演示**（若尚未迁移，也可继续使用本仓库原交互逻辑）。
"""
    )

st.markdown("---")
st.markdown("## Topic 3 一分钟导读")

st.markdown(
    """
给定 $\\{y,X\\}$，$X$ 为 $S\\times k$。**假设 A1** 要求 $\\mathrm{rank}(X)=k<S$（无完全共线性）。
此时

$$
\\hat{b}_{\\mathrm{OLS}}=(X^\\top X)^{-1}X^\\top y
$$

存在且唯一；拟合与残差可分别写成 $\\hat{y}=P_X y$、$\\hat{e}=M_X y$。
若只关心部分系数，**FWL 定理**说明：全回归、只残差化右侧、左右都残差化——三条路得到同一个子向量系数。这就是回归“控制其他变量”的代数含义。

更完整的中文讲解、公式推导要点与可交互例子，见侧栏 **「Topic3 讲义讲解」**。
"""
)

st.sidebar.markdown(
    """
### 导航提示
Streamlit 多页面应用会在上方 / 侧栏列出 `pages/` 下的页面。

- **Topic3 讲义讲解**：本次上传讲义的主讲解  
- **Apriori 演示**：数据挖掘原示例
"""
)
