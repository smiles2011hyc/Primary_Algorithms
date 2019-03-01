##### 1.机器学习的一些概念

- **有监督学习**：分类和回归问题属于有监督学习，其实因为这类算法必须知道预测什么，即目标变量的分类信息。

  > 分类：主要任务是将实例数据划分到合适的分类中。
  >
  > 回归：主要用于预测数值型数据；如：数据拟合曲线——通过给定数据点的最优拟合曲线。

- **无监督学习：**数据没有类别信息，也不会给定目标，在无监督学习中，将数据集合分成由类似的对象组成的多个类的过程称为**聚类**。将寻找描述数据统计值的过程称之为**密度估计**。


- **损失函数（loss Function)**:用来估量模型的预测值**f(x)**与真实值**Y**的不一致程度，其是一个**非负实数**函数，通常用**L(Y,f(x))**来表示。**损失函数越小，模型的鲁棒性（robustness）越好**。损失函数是经验风险函数的核心部分，也是结构风险函数的重要组成部分。统计学习常用的损失函数有以下4种：
  1. 0-1损失函数（0-1 loss function）

$$
L(y,f(x))=\begin{cases}1,& {y \neq f(x)} \\0,& {y=f(x)} \end{cases}
$$

​          2. 平方损失函数 （quadratic loss function)​      
$$
L(Y,f(x))=(Y-f(x))^2
$$
​          3. 绝对损失函数（absolute loss function）
$$
L(Y,f(x))=|Y-f(x)|
$$
​          4. 对数损失函数（logarithmic loss function）或对数似然损失函数（log-likelihood loss function）
$$
L(Y,P(Y|X))=-logP(Y|X)
$$

- **代价函数（Cost Function）**：损失函数是定义在单个样本的，而代价函数是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均。

$$
J(\theta) = \frac{1}{2}(y-h_\theta(x))^2
$$

- **目标函数（Objective Function）**：最终需要优化的函数，等于经验风险+结构风险（也就是代价函数+正则化）,其与代价函数最明显的区别是目标函数是最大化或者最小化，而后者是最小化。
- **过拟合 （Overfitting）**：通常在变量（特征）过多的时候，训练出的函数总能很好的拟合训练数据，即代价函数可能非常接近于0或者就是0，但是这样的函数很可能无法泛化到新的数据样本中。解决方法：
  1. 减少变量的个数，舍弃一些变量，保留更为重要的变量；
  2. 保留所有的变量，将不重要的特征的权值置为0或者变小使得特征的参数矩阵变得稀疏。
- **正则化（Regularization）**：模型选择的典型方法是正则化，正则化是结构风险最小化策略的实现，是在经验风险上加一个**正则化项（regularizer）**或**罚值（penalty term）**。正则化项一般是模型复杂度的单调递增函数，模型越复杂，正则化值就越大。
- **泛化能力 （generalization ability）**：是指由某一方法学习到的模型对未知数据的预测能力，是学习方法本质上重要的性质。现实中采用最多的方法是通过测试误差来评价学习方法的泛化能力。

主要内容**：关于在计算机上从数据中产生“模型”（model）的算法，即“学习算法”（learning algorithm）；有了学习算法，我们把经验数据提供给它，它就能基于这些数据产生模型；在面对新的情况时，模型会给我们提供相应的判断。更形式化的定义为：**假设用$P$来评估计算机程序在某任务类$T$上的性能，若一个程序通过利用经验$E$在$T$中任务上获得了性能改善，则我们就说关于$T$和$P$，该程序对$E$进行了学习。**

**假设空间**：**归纳**（induction）与**演绎**（deduction）是科学推理的两大基本手段；前者是从特殊到一般的“**泛化（generalization）过程**，即从具体的事实归结出一般性规律；后者则是从一般到特殊的**特化（specialization）过程**，即从基础原理推演出具体状况。

**归纳偏好**（inductive bias）：机器学习算法在学习过程中对某种类型假设的偏好，称为”归纳偏好“

![1545049491569](https://github.com/smiles2011hyc/Gallery/blob/master/1545049491569.png)

​        这里的每个训练样本是图中的一个点（$x,y$),要学得一个与训练集一致得模型，相当于找到一条穿过所有训练样本点得曲线；显然，对有限个样本点组成得训练集，存在着很多条曲线与其一致，我们得学习算法必须有某种偏好，才能长出它认为”正确“得模型。例如，若认为相似的样本应有相似的输出，则对应的学习算法可能偏好图中，比较**平滑**的曲线A而不是比较**崎岖**的曲线B。

​        **奥卡姆剃刀**（Occam's razor)是一种常用的、自然科学研究中最基本的原则，即**若有多个假设与观察一致，则选最简单的那个**。

##### 2.线性回归的原理

> 线性回归在假设特征满足线性关系，根据给定的训练数据训练一个模型，并用此模型进行预测。假定一个线性方程$Y=2x+1​$,$x​$变量为商品大小，$Y​$代表为销售量；当月份$x=5​$时，我们就能根据线性模型预测出$Y=11​$销量；则我们就可以粗略的将$Y=2x+1​$看做回归的模型。
>

##### 3.损失函数推导

- 均方误差（root mean  square error）（MSE）

$$
MSE=\frac{1}{m}\sum_{i=1}^n(y-\hat{y})^2
$$

- 均方根误差（root mean square error）（RMSE）

$$
RMSE=\sqrt{\frac{1}{m}\sum_{i=1}^m(y_i-\hat{y_i})^2}
$$

- 平均绝对误差（Mean Absolute Error）（MAE）

$$
MAE=\frac{1}{m}\sum_{i=1}^m|y-\hat{y}|
$$

- R-Squared（$R^2$)

  > 利用数据拟合一个模型，则此模型肯定是存在误差的，那么回归方程对观测值拟合的如何，就叫做拟合优度，$R^2$就是度量拟合优度的一个统计量，也叫做可决系数，其计算方法是：
  > $$
  > R^2=1-\sum_i\frac{(y-\hat{y})^2}{(y-\overline{y})^2}
  > $$
  > 上式用1减去$y$对回归方程的方差（未解释离差）与$y$的总方差的比值，分子是残差，即拟合方程中不能解释的部分，则$R^2$解释了因变量变动的百分比的多少，则$R^2$越大越好，

##### 4.梯度下降法（gradient  descent）

> 加入$f(x)$在$x_o$的梯度就是$f(x)$变化最快的方向，梯度下降法就是一个最优化算法，通常称为**最速下降法**，梯度下降不一定能够找到全局的最优解，有可能是一个局部最优解。当然，如果损失函数是**凸函数**，梯度下降法得到的解就一定是全局最优解。

##### 5. 牛顿法（Newton method）

> 牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数$f(x)$的泰勒级数的前面几项来寻找方程$f(x)=0$的根。牛顿法最大的特点给就在于它的收敛速度很快。具体步骤：
>
> ​    首先，选择一个接近函数$f(x)$零点的$x_0$,计算相应的$f(x_0)$和切线斜率$f\prime(x_0)$(这里$f\prime$表示函数$f$的导数）。然后我们计算穿过点（$x_0$,$f(x_o)$)并且斜率为$f\prime(x_0)$的直线和$x$轴的交点的$x$坐标，也就是求如下解：
> $$
> x.f\prime(x_0)+f(x_0)-x_0.f\prime(x_0)=0
> $$
> ​    我们将新求得的点的$x$坐标命名为$x_1$,通常$x_1$会比$x_0$更接近方程$f(x)=0$的解，因此可以使用$x_1$开始下一轮迭代。迭代公式如下：
> $$
> x_{n+1}=x_n-\frac{f(x_n)}{f\prime(x_n)}
> $$
>

##### 6.拟牛顿法（quasi Newton method）

> 拟牛顿法的本质思想是改善牛顿法每次需要求解复杂Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂的。

##### 6.sklearn参数详解

​    由于时间关系只是copy和运行了一下别人的代码：https://blog.csdn.net/linxid/article/details/79104130

    如需认真学习，可参考：
    https://blog.csdn.net/u014248127/article/details/78885180
    https://blog.csdn.net/linxid/article/details/79104130

```Python
##安装scikit-learn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit_learn
##鸢尾花数据集
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
iris_X = iris.data
iris_Y =iris.target
X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size =0.3)
print(Y_train)
[2 0 1 0 2 1 0 0 1 2 0 2 1 2 1 2 2 2 1 0 1 2 1 2 0 1 1 1 2 2 0 1 1 1 1 2 0
 0 0 1 2 1 0 2 1 0 1 0 1 2 2 0 2 0 0 1 2 1 1 2 1 2 0 0 2 2 1 0 0 2 2 1 1 0
 0 1 1 0 1 0 1 0 2 0 2 0 0 2 2 0 0 2 2 1 1 2 2 0 0 0 2 0 1 1 0]
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')
print(knn.predict(X_test))
[2 0 1 2 1 1 0 1 2 2 2 2 1 2 2 0 0 1 1 0 1 0 2 1 2 2 2 2 0 2 0 2 1 2 0 0 0
 0 2 0 1 1 1 0 1]
print(Y_test)
[2 0 1 2 1 1 0 1 2 2 2 2 1 2 1 0 0 1 1 0 1 0 2 1 2 2 2 2 0 2 0 2 1 2 0 0 0
 0 2 0 1 1 1 0 1]
##波士顿房价数据集
import matplotlib.pyplot as plt
load_data = datasets.load_boston()
data_X = load_data.data
data_Y = load_data.target
model = LinearRegression()
model.fit(data_X,data_Y)
##LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
print(model.predict(data_X[:4,:]))
##[30.00384338 25.02556238 30.56759672 28.60703649]
print(data_Y[:4])
[24.  21.6 34.7 33.4]
```


