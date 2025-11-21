import { CourseNode } from './types';

export const COURSE_DATA: CourseNode[] = [
  {
    id: 'phase-1',
    title: '第一阶段：机器学习基石',
    type: 'category',
    children: [
      {
        id: 'p1-intro',
        title: '第1章：环境与概述',
        type: 'category',
        children: [
          {
            id: 'p1-what-is-ml',
            title: '1.1 什么是机器学习',
            type: 'lesson',
            content: {
              description: '了解人工智能、机器学习与深度学习的关系，以及机器学习的基本工作流程：从数据中学习规律，而非手动编写规则。',
              keyConcepts: ['监督学习', '特征(Features)', '标签(Labels)', '泛化能力'],
              codeExample: `# 传统编程 vs 机器学习
def simple_rule(x):
    # 这是一个简单的规则模型示例
    # 机器学习的本质是寻找函数 f(x)
    if x > 50:
        return "High"
    else:
        return "Low"

# 模拟数据流
data_stream = [20, 45, 60, 80]
for x in data_stream:
    print(f"Input: {x}, Prediction: {simple_rule(x)}")`,
              explanation: '在传统编程中，我们不仅输入数据，还要输入规则（代码逻辑）。而在机器学习中，我们输入数据和答案（在监督学习中），机器通过算法自动学习出规则。'
            }
          },
          {
            id: 'p1-env',
            title: '1.2 环境搭建 (Anaconda & Pip)',
            type: 'lesson',
            content: {
              description: '工欲善其事，必先利其器。学习如何配置 Python 数据科学环境，管理虚拟环境及依赖包。',
              keyConcepts: ['VirtualEnv', 'Conda', 'Jupyter Notebook', 'Pip'],
              codeExample: `# 检查 Python 环境配置
import sys
import platform

print(f"Python 版本: {sys.version.split()[0]}")
print(f"操作系统: {platform.system()} {platform.release()}")

# 检查核心库是否安装
try:
    import numpy
    import pandas
    print("✅ NumPy & Pandas 已安装")
except ImportError as e:
    print(f"❌ 缺少库: {e.name}")`,
              explanation: '稳定的环境是实验的基础。在实际开发中，建议为每个项目创建独立的虚拟环境以避免版本冲突。'
            }
          }
        ]
      },
      {
        id: 'p1-math',
        title: '第2章：数学基础',
        type: 'category',
        children: [
            {
                id: 'p1-linear-alg',
                title: '2.1 线性代数核心',
                type: 'lesson',
                content: {
                    description: '向量和矩阵是机器学习的语言。掌握点积、矩阵乘法对于理解神经网络至关重要。',
                    keyConcepts: ['向量(Vector)', '矩阵(Matrix)', '点积(Dot Product)', '转置'],
                    codeExample: `import numpy as np

# 定义特征向量 (例如: [面积, 房间数])
features = np.array([120, 3])

# 定义权重向量 (模型学习到的参数)
weights = np.array([0.5, 10])

# 计算加权和 (点积)
# score = 120*0.5 + 3*10 = 60 + 30 = 90
score = np.dot(features, weights)

print(f"房屋评分: {score}")`,
                    explanation: '点积操作能够高效地计算输入特征与权重的加权和，这是几乎所有神经网络神经元计算的基础。'
                }
            },
            {
                id: 'p1-calc',
                title: '2.2 微积分与梯度下降',
                type: 'lesson',
                content: {
                    description: '梯度是函数上升最快的方向。在机器学习中，我们需要沿着梯度的反方向更新参数，以最小化损失函数。',
                    keyConcepts: ['导数', '偏导数', '梯度', '学习率'],
                    codeExample: `# 模拟简单的梯度下降
# 目标函数: y = x^2 (最小值在 x=0)
# 导数 (梯度): dy/dx = 2x

x = 10  # 初始猜测值
learning_rate = 0.1  # 学习率

print(f"初始 x: {x}")

for i in range(5):
    gradient = 2 * x
    x = x - learning_rate * gradient
    print(f"第 {i+1} 次迭代: x = {x:.4f}, gradient = {gradient:.4f}")`,
                    explanation: '通过不断减去梯度（乘以学习率），x 的值逐渐逼近抛物线 y=x^2 的最低点 0。这就是模型训练的核心机制。'
                }
            }
        ]
      },
      {
        id: 'p1-libs',
        title: '第3章：数据处理库',
        type: 'category',
        children: [
           {
            id: 'm1-numpy',
            title: '3.1 NumPy 数组操作',
            type: 'lesson',
            content: {
              description: 'NumPy 是 Python 数值计算的基石，提供了高性能的多维数组对象及广播机制。',
              keyConcepts: ['ndarray', '广播(Broadcasting)', '切片(Slicing)'],
              codeExample: `import numpy as np

# 创建矩阵
data = np.array([[1, 2, 3], 
                 [4, 5, 6]])

# 广播机制：直接对整个矩阵进行运算
print("原始数据:\n", data)
print("所有元素乘以 10:\n", data * 10)

# 统计计算
print(f"平均值: {np.mean(data)}")
print(f"每列最大值: {np.max(data, axis=0)}")`,
              explanation: 'NumPy 的底层使用 C 语言编写，其向量化操作比 Python 原生的循环快几个数量级，是处理大数据集的必备工具。'
            }
          },
          {
            id: 'm1-pandas',
            title: '3.2 Pandas 数据分析',
            type: 'lesson',
            content: {
              description: 'Pandas 提供了 DataFrame 结构，使得处理表格数据（如 Excel、CSV）变得异常简单。',
              keyConcepts: ['DataFrame', 'Series', 'GroupBy', '缺失值处理'],
              codeExample: `import pandas as pd

# 创建模拟数据集
df = pd.DataFrame({
    '城市': ['北京', '上海', '北京', '深圳', '上海'],
    '销售额': [100, 200, 150, 300, 220],
    '年份': [2023, 2023, 2024, 2024, 2024]
})

# 分组聚合分析
report = df.groupby('城市')['销售额'].sum()
print("城市销售总额排行榜：")
print(report.sort_values(ascending=False))`,
              explanation: 'Pandas 的 groupby 操作类似于 SQL 中的 GROUP BY，非常适合进行数据的探索性分析（EDA）。'
            }
          } 
        ]
      }
    ]
  },
  {
    id: 'phase-2',
    title: '第二阶段：监督学习 (Supervised)',
    type: 'category',
    children: [
      {
        id: 'p2-regression',
        title: '第4章：回归算法',
        type: 'category',
        children: [
            {
                id: 'm2-linear-reg',
                title: '4.1 线性回归原理',
                type: 'lesson',
                content: {
                  description: '回归用于预测连续值（如房价、气温）。线性回归试图拟合一条直线（或超平面）来最小化预测误差。',
                  keyConcepts: ['假设函数', '均方误差(MSE)', '拟合'],
                  codeExample: `from sklearn.linear_model import LinearRegression
import numpy as np

# 训练数据：[房屋面积] -> [价格(万)]
X = np.array([[50], [80], [100], [120]])
y = np.array([150, 240, 300, 350])

# 初始化并训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
new_house = [[90]]
pred = model.predict(new_house)
print(f"90平米房屋预测价格: {pred[0]:.2f} 万")
print(f"方程斜率 (每平米单价): {model.coef_[0]:.2f} 万")`,
                  explanation: 'Scikit-learn 封装了复杂的数学计算。模型通过 fit 方法学习数据中的线性关系，predict 方法用于推断未知数据。'
                }
              },
              {
                id: 'm2-poly-reg',
                title: '4.2 多项式回归',
                type: 'lesson',
                content: {
                    description: '当数据不是线性分布时，我们需要引入高次项（如 x²）来拟合曲线。',
                    keyConcepts: ['非线性', '特征转换', '过拟合风险'],
                    codeExample: `from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2], [3], [4]]) # 原始特征

# 转换为包含平方项的特征
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("原始特征 X:\n", X)
print("多项式特征 (1, x, x^2):\n", X_poly)`,
                    explanation: '通过增加特征的维度（例如添加 x的平方），线性模型可以被“欺骗”从而拟合出曲线形状的决策边界。'
                }
              }
        ]
      },
      {
        id: 'p2-classification',
        title: '第5章：分类算法',
        type: 'category',
        children: [
            {
                id: 'm2-c-logistic',
                title: '5.1 逻辑回归',
                type: 'lesson',
                content: {
                    description: '虽然名字叫回归，但它实际上是解决二分类问题的基石算法。',
                    keyConcepts: ['Sigmoid函数', '概率输出', '决策边界'],
                    codeExample: `import numpy as np

# Sigmoid 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 模拟模型输出值 z
z_values = [-5, 0, 5]

for z in z_values:
    prob = sigmoid(z)
    label = "正类" if prob >= 0.5 else "负类"
    print(f"输入: {z:>2} -> 概率: {prob:.4f} -> 预测: {label}")`,
                    explanation: 'Sigmoid 函数将任意实数压缩到 (0, 1) 之间，可以被解释为事件发生的概率。通常以 0.5 为阈值进行分类。'
                }
            },
            {
                id: 'm2-knn',
                title: '5.2 K-近邻算法 (KNN)',
                type: 'lesson',
                content: {
                    description: '“近朱者赤，近墨者黑”。KNN 是一种基于实例的惰性学习算法，通过查找最近的 K 个邻居来投票决定分类。',
                    keyConcepts: ['欧氏距离', 'K值选择', '非参数模型'],
                    codeExample: `from sklearn.neighbors import KNeighborsClassifier

# 特征: [甜度, 脆度]
# 标签: 0=苹果, 1=梨
X = [[8, 5], [9, 6], [3, 7], [2, 6]]
y = [0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

unknown_fruit = [[8.5, 5.5]]
print("预测结果 (0=苹果, 1=梨):", knn.predict(unknown_fruit))`,
                    explanation: 'KNN 没有显式的训练过程，它在预测时实时计算距离。这使得它简单直观，但在大数据集上计算量较大。'
                }
            },
            {
                id: 'm2-svm',
                title: '5.3 支持向量机 (SVM)',
                type: 'lesson',
                content: {
                    description: 'SVM 试图找到一个超平面，使得不同类别之间的间隔（Margin）最大化。',
                    keyConcepts: ['最大间隔', '核函数(Kernel)', '支持向量'],
                    codeExample: `# 概念演示代码
print("SVM 核心思想：")
print("1. 找到区分两类数据的线")
print("2. 确保这条线距离两边的最近点（支持向量）最远")
print("3. 如果线性的分不开，就用核函数投射到高维空间切分")`,
                    explanation: 'SVM 在高维空间和小样本数据集上表现优异，曾是深度学习流行之前最强大的分类器之一。'
                }
            }
        ]
      }
    ]
  },
  {
    id: 'phase-3',
    title: '第三阶段：无监督学习',
    type: 'category',
    children: [
      {
        id: 'p3-clustering',
        title: '第6章：聚类算法',
        type: 'category',
        children: [
           {
            id: 'm3-kmeans',
            title: '6.1 K-Means 聚类',
            type: 'lesson',
            content: {
                description: '将数据划分为 K 个簇，使簇内差异最小化，簇间差异最大化。',
                keyConcepts: ['质心(Centroid)', '迭代优化', '无标签数据'],
                codeExample: `from sklearn.cluster import KMeans
import numpy as np

# 模拟客户数据 [购买量, 访问频次]
X = np.array([[1, 1], [1, 2], [10, 10], [10, 12]])

# 聚成 2 类
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("每个样本的类别归属:", kmeans.labels_)
print("聚类中心坐标:\n", kmeans.cluster_centers_)`,
                explanation: '算法自动发现了两组截然不同的客户群体，而不需要我们预先告诉它谁属于哪一组。'
            }
           }
        ]
      },
      {
          id: 'p3-dim-red',
          title: '第7章：降维',
          type: 'category',
          children: [
              {
                  id: 'm3-pca',
                  title: '7.1 主成分分析 (PCA)',
                  type: 'lesson',
                  content: {
                      description: 'PCA 用于在保留数据主要特征的前提下减少数据的维度，常用于数据压缩或可视化。',
                      keyConcepts: ['方差', '投影', '特征值'],
                      codeExample: `# 伪代码概念
# PCA(n_components=2)
# 将 100 维的数据压缩成 2 维
# 丢弃噪声，保留主要信息`,
                      explanation: '想象给一个三维物体拍照片变成二维图片，PCA 就是在寻找那个能保留最多物体细节的拍摄角度。'
                  }
              }
          ]
      }
    ]
  },
  {
    id: 'phase-4',
    title: '第四阶段：深度学习入门',
    type: 'category',
    children: [
      {
        id: 'p4-neural-nets',
        title: '第8章：神经网络',
        type: 'category',
        children: [
            {
                id: 'm4-basic-nn',
                title: '8.1 感知机与神经网络',
                type: 'lesson',
                content: {
                    description: '受人脑神经元启发，通过层层堆叠的神经元来模拟复杂的非线性关系。',
                    keyConcepts: ['前向传播', '反向传播', '激活函数(ReLU)'],
                    codeExample: `import torch
import torch.nn as nn

# 定义一个简单的全连接网络
model = nn.Sequential(
    nn.Linear(10, 5),  # 输入层 -> 隐藏层
    nn.ReLU(),         # 非线性激活
    nn.Linear(5, 1)    # 隐藏层 -> 输出层
)

print(model)
# 深度学习框架自动处理了复杂的矩阵运算和求导`,
                    explanation: '现代深度学习框架（如 PyTorch, TensorFlow）使得搭建多层神经网络就像搭积木一样简单。'
                }
            }
        ]
      }
    ]
  }
];