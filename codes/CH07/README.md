# SVM-Iris

### 文件说明

- `utils.py` 数据处理
- `svm_smo.py` 自己实现的SVM方法对数据集进行分类
- `svm_sklearn.py` 调用了sklearn中SVM方法
- `compare.py` 对两种方法进行准确率和时间效率的对比
- `data` 数据集目录



### 数据处理

- 数据来源为鸢尾花数据集的变体
- 特征选择时选取了全部4个特征
- 并且将全部150条数据的25%作为测试集



### 解释

- 代码`svm_smo.py`中`SVM`的`fit`训方法运用了`SMO`算法来更新`alpha`的值

```python
alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
alpha[j] = max(alpha[j], L)
alpha[j] = min(alpha[j], H)
alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
```

- SMO算法

> 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),\dots, (x_N,y_N)}$，其中$x_i\in\mathcal X=\bf R^n, y_i\in\mathcal Y=\{-1,+1\}, i=1,2,\dots,N$,精度$\epsilon$
>
> 输出：近似解$\hat\alpha$
>
> 1. 取初值$\alpha_0=0$，令$k=0$
>
> 1. **选取**优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$，解析求解两个变量的最优化问题，求得最优解$\alpha_1^{(k+1)},\alpha_2^{(k+1)}$，更新$\alpha$为$\alpha^{k+1}$
>
> 1. 若在精度$\epsilon$范围内满足停机条件
>    $$
>    \sum_{i=1}^{N}\alpha_iy_i=0\\
>    0\leqslant\alpha_i\leqslant C,i=1,2,\dots,N\\
>    y_i\cdot g(x_i)=
>    \begin{cases}
>    \geqslant1,\{x_i|\alpha_i=0\}\\
>    =1,\{x_i|0<\alpha_i<C\}\\
>    \leqslant1,\{x_i|\alpha_i=C\}
>    \end{cases}\\
>    g(x_i)=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)+b
>    $$
>    则转4,否则，$k=k+1$转2
>
> 1. 取$\hat\alpha=\alpha^{(k+1)}$



### 结果

- 运行结果

```
Number of training samples: 113
Number of test samples: 37
Predictions:
[ 1 -1  1 -1 -1 -1 -1 -1  1 -1 -1  1  1 -1 -1 -1 -1 -1 -1  1  1 -1 -1  1
  1 -1 -1 -1 -1  1  1 -1 -1 -1 -1  1  1]
Accuracy: 0.8108108108108109
Number of Support Vectors: 67
Coef(weights): [-16.18832215  -4.48051598 -37.79549257 -20.42952706]
Intercept(bias): 271.8895018507325
```

- 性能对比

```
# first
Accuracy comparison: 0.8108108108108109 and 0.972972972972973
Time consumption: 0.07299971580505371s and 0.0009999275207519531s
# second
Accuracy comparison: 0.7567567567567568 and 0.972972972972973
Time consumption: 0.035001277923583984s and 0.0010006427764892578s
```

