
---

```markdown
# 《PyTorch深度学习实践》课程及作业代码完整版

课程：B站 @刘二大人 【《PyTorch深度学习实践》完结合集】 https://www.bilibili.com/video/BV1Y7411d7Ys/?p=13&share_source=copy_web&vd_source=ca081a1c0d3d84091ecb1700100bec4f

记录本人在学习过程中写的代码，包括课程内容及作业的实现，均为纯手敲。
第一个完整学完的课程，如有疏漏欢迎指正！

## 环境

- Python 3.12
- PyTorch 2.14.0
- 需要的库：torch, numpy, matplotlib, pandas

## 目录

- [p02：线性模型](#p02线性模型)
- [p03：梯度下降算法](#p03梯度下降算法)
- [p04：反向传播](#p04反向传播)
- [p05：线性回归](#p05线性回归)
- [p06：逻辑斯蒂回归](#p06逻辑斯蒂回归)
- [p07：多维特征输入](#p07多维特征输入)
- [p08：加载数据集](#p08加载数据集)
- [p09：多分类问题](#p09多分类问题)
- [p10_1：卷积神经网络_基本用法](#p10_1卷积神经网络_基本用法)
- [p10_2：卷积神经网络_自定义卷积核](#p10_2卷积神经网络_自定义卷积核)
- [p10_3：卷积神经网络_手写数字识别](#p10_3卷积神经网络_手写数字识别)
- [p11：卷积神经网络_手写数字识别优化](#p11卷积神经网络_手写数字识别优化)
- [p12_1：循环神经网络_基本用法](#p12_1循环神经网络_基本用法)
- [p12_2：循环神经网络_手动循环](#p12_2循环神经网络_手动循环)
- [p12_3：循环神经网络_自动循环](#p12_3循环神经网络_自动循环)
- [p13：循环神经网络_人名国籍分类](#p13循环神经网络_人名国籍分类)

---

## 作业

| 文件 | 内容 |
|------|------|
| `p02homework.py` | 线性回归损失曲面绘制 |
| `p04homework.py` | 拟合二次函数 |
| `p05homework.py` | 各种优化器对比 |
| `p08homework.py` | 泰坦尼克号生存率 |
| `p09homework.py` | 9分类商品预测 |
| `p10homework.py` | 三层卷积+池化 |
| `p12homework.py` | 尝试GRU |
| `p13homework.py` | 电影评论情感分析 |

---

## 数据集

以下数据集需自行下载：

- **MNIST**
- **Titanic**：[Kaggle](https://www.kaggle.com/c/titanic)
- **Otto**：[Kaggle](https://www.kaggle.com/c/otto-group-product-classification-challenge)
- **Movie Reviews**：[Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

---

## 运行方式

```bash
python p01.py