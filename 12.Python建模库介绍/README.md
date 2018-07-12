

```python
import pandas as pd
```


```python
import numpy as np
```


```python
import patsy
```


```python
import statsmodels.api as sm
```


```python
import statsmodels.formula.api as smf
```


```python
from sklearn.linear_model import LogisticRegression
```


```python
from sklearn.linear_model import LogisticRegressionCV
```


```python
from sklearn.model_selection import cross_val_score
```


```python
# 1.pandas与模型代码的接口
```


```python
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]})
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.columns
```




    Index(['x0', 'x1', 'y'], dtype='object')




```python
data.values
```




    array([[ 1.  ,  0.01, -1.5 ],
           [ 2.  , -0.01,  0.  ],
           [ 3.  ,  0.25,  3.6 ],
           [ 4.  , -4.1 ,  1.3 ],
           [ 5.  ,  0.  , -2.  ]])




```python
df2 = pd.DataFrame(data.values, columns=['one', 'two', 'three'])
```


```python
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = data.copy()
```


```python
df3['strings'] = ['a', 'b', 'c', 'd', 'e']
```


```python
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>strings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.values
```




    array([[1, 0.01, -1.5, 'a'],
           [2, -0.01, 0.0, 'b'],
           [3, 0.25, 3.6, 'c'],
           [4, -4.1, 1.3, 'd'],
           [5, 0.0, -2.0, 'e']], dtype=object)




```python
model_cols = ['x0', 'x1']
```


```python
data.loc[:, model_cols].values
```




    array([[ 1.  ,  0.01],
           [ 2.  , -0.01],
           [ 3.  ,  0.25],
           [ 4.  , -4.1 ],
           [ 5.  ,  0.  ]])




```python
data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'], categories=['a', 'b'])
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 替换category列为虚变量，
# 创建虚变量，
dummies = pd.get_dummies(data.category, prefix='category')
# 删除category列，然后添加到结果
data_with_dummies = data.drop('category', axis=1).join(dummies)
```


```python
data_with_dummies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category_a</th>
      <th>category_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2.用Patsy创建模型描述
```


```python
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]})
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('y ~ x0 + x1', data)
```


```python
y
```




    DesignMatrix with shape (5, 1)
         y
      -1.5
       0.0
       3.6
       1.3
      -2.0
      Terms:
        'y' (column 0)




```python
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0     x1
              1   1   0.01
              1   2  -0.01
              1   3   0.25
              1   4  -4.10
              1   5   0.00
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'x1' (column 2)




```python
np.asarray(y)
```




    array([[-1.5],
           [ 0. ],
           [ 3.6],
           [ 1.3],
           [-2. ]])




```python
np.asarray(X)
```




    array([[ 1.  ,  1.  ,  0.01],
           [ 1.  ,  2.  , -0.01],
           [ 1.  ,  3.  ,  0.25],
           [ 1.  ,  4.  , -4.1 ],
           [ 1.  ,  5.  ,  0.  ]])




```python
patsy.dmatrices('y ~ x0 + x1 + 0', data)[1]
```




    DesignMatrix with shape (5, 2)
      x0     x1
       1   0.01
       2  -0.01
       3   0.25
       4  -4.10
       5   0.00
      Terms:
        'x0' (column 0)
        'x1' (column 1)




```python
# 最小二乘回归
coef, resid, _, _ = np.linalg.lstsq(X, y)
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      
    


```python
coef
```




    array([[ 0.31290976],
           [-0.07910564],
           [-0.26546384]])




```python
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
```


```python
coef
```




    Intercept    0.312910
    x0          -0.079106
    x1          -0.265464
    dtype: float64




```python
# 用 Patsy公式进行数据转换
```


```python
y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
```


```python
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0  np.log(np.abs(x1) + 1)
              1   1                 0.00995
              1   2                 0.00995
              1   3                 0.22314
              1   4                 1.62924
              1   5                 0.00000
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'np.log(np.abs(x1) + 1)' (column 2)




```python
y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
```


```python
X
```




    DesignMatrix with shape (5, 3)
      Intercept  standardize(x0)  center(x1)
              1         -1.41421        0.78
              1         -0.70711        0.76
              1          0.00000        1.02
              1          0.70711       -3.33
              1          1.41421        0.77
      Terms:
        'Intercept' (column 0)
        'standardize(x0)' (column 1)
        'center(x1)' (column 2)




```python
# 作为建模的一步，你可能拟合模型到一个数据集，然后用另一个数据集评估模型。
# 另一个数据集可能是剩余的部分或是新数据。
# 当执行中心化和标准化转变，用新数据进行预测要格外小心。
# 因为你必须使用平均值或标准差转换新数据集，这也称作状态转换
```


```python
# patsy.build_design_matrices函数可以应用于转换新数据，使用原始样本数据集的保存信息
```


```python
new_data = pd.DataFrame({
    'x0': [6, 7, 8, 9],
    'x1': [3.1, -0.5, 0, 2.3],
    'y': [1, 2, 3, 4]})
```


```python
 new_X = patsy.build_design_matrices([X.design_info], new_data)
```


```python
new_X
```




    [DesignMatrix with shape (4, 3)
       Intercept  standardize(x0)  center(x1)
               1          2.12132        3.87
               1          2.82843        0.27
               1          3.53553        0.77
               1          4.24264        3.07
       Terms:
         'Intercept' (column 0)
         'standardize(x0)' (column 1)
         'center(x1)' (column 2)]




```python
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
```


```python
X
```




    DesignMatrix with shape (5, 2)
      Intercept  I(x0 + x1)
              1        1.01
              1        1.99
              1        3.25
              1       -0.10
              1        5.00
      Terms:
        'Intercept' (column 0)
        'I(x0 + x1)' (column 1)




```python
# 分类数据和Patsy
```


```python
data = pd.DataFrame({
    'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
    'key2': [0, 1, 0, 1, 0, 1, 0, 0],
    'v1': [1, 2, 3, 4, 5, 6, 7, 8],
    'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]
})
```


```python
y, X = patsy.dmatrices('v2 ~ key1', data)
```


```python
X
```




    DesignMatrix with shape (8, 2)
      Intercept  key1[T.b]
              1          0
              1          0
              1          1
              1          1
              1          0
              1          1
              1          0
              1          1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)




```python
y, X = patsy.dmatrices('v2 ~ key1 + 0', data)
```


```python
X
```




    DesignMatrix with shape (8, 2)
      key1[a]  key1[b]
            1        0
            1        0
            0        1
            0        1
            1        0
            0        1
            1        0
            0        1
      Terms:
        'key1' (columns 0:2)




```python
y, X = patsy.dmatrices('v2 ~ C(key2)', data)
```


```python
X
```




    DesignMatrix with shape (8, 2)
      Intercept  C(key2)[T.1]
              1             0
              1             1
              1             0
              1             1
              1             0
              1             1
              1             0
              1             0
      Terms:
        'Intercept' (column 0)
        'C(key2)' (column 1)




```python
data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>zero</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>one</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>zero</td>
      <td>3</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>one</td>
      <td>4</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>zero</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>one</td>
      <td>6</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>zero</td>
      <td>7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>zero</td>
      <td>8</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('v2 ~ key1 + key2', data)
```


```python
X
```




    DesignMatrix with shape (8, 3)
      Intercept  key1[T.b]  key2[T.zero]
              1          0             1
              1          0             0
              1          1             1
              1          1             0
              1          0             1
              1          1             0
              1          0             1
              1          1             1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)




```python
y, X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data)
```


```python
X
```




    DesignMatrix with shape (8, 4)
      Intercept  key1[T.b]  key2[T.zero]  key1[T.b]:key2[T.zero]
              1          0             1                       0
              1          0             0                       0
              1          1             1                       1
              1          1             0                       0
              1          0             1                       0
              1          1             0                       0
              1          0             1                       0
              1          1             1                       1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)
        'key1:key2' (column 3)




```python
# 3.statsmodels介绍
```


```python
# statsmodels是Python进行拟合多种统计模型、进行统计试验和数据探索可视化的库。
# Statsmodels包含许多经典的统计方法，但没有贝叶斯方法和机器学习模型。
# statsmodels包含的模型有：
#     线性模型，广义线性模型和健壮线性模型
#     线性混合效应模型
#     方差（ANOVA）方法分析
#     时间序列过程和状态空间模型
#     广义矩估计
```


```python
# 估计线性模型
```


```python
# 从随机数据生成一个线性模型：
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * np.random.randn(*size)

# For reproducibility
np.random.seed(12345)

N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps
```


```python
X[:5]
```




    array([[-0.12946849, -1.21275292,  0.50422488],
           [ 0.30291036, -0.43574176, -0.25417986],
           [-0.32852189, -0.02530153,  0.13835097],
           [-0.35147471, -0.71960511, -0.25821463],
           [ 1.2432688 , -0.37379916, -0.52262905]])




```python
y[:5]
```




    array([ 0.42786349, -0.67348041, -0.09087764, -0.48949442, -0.12894109])




```python
X_model = sm.add_constant(X)
```


```python
X_model[:5]
```




    array([[ 1.        , -0.12946849, -1.21275292,  0.50422488],
           [ 1.        ,  0.30291036, -0.43574176, -0.25417986],
           [ 1.        , -0.32852189, -0.02530153,  0.13835097],
           [ 1.        , -0.35147471, -0.71960511, -0.25821463],
           [ 1.        ,  1.2432688 , -0.37379916, -0.52262905]])




```python
# sm.OLS类可以拟合一个普通最小二乘回归
model = sm.OLS(y, X)
```


```python
results = model.fit()
```


```python
results.params
```




    array([0.17826108, 0.22303962, 0.50095093])




```python
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.430
    Model:                            OLS   Adj. R-squared:                  0.413
    Method:                 Least Squares   F-statistic:                     24.42
    Date:                Thu, 12 Jul 2018   Prob (F-statistic):           7.44e-12
    Time:                        22:01:13   Log-Likelihood:                -34.305
    No. Observations:                 100   AIC:                             74.61
    Df Residuals:                      97   BIC:                             82.42
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.1783      0.053      3.364      0.001       0.073       0.283
    x2             0.2230      0.046      4.818      0.000       0.131       0.315
    x3             0.5010      0.080      6.237      0.000       0.342       0.660
    ==============================================================================
    Omnibus:                        4.662   Durbin-Watson:                   2.201
    Prob(Omnibus):                  0.097   Jarque-Bera (JB):                4.098
    Skew:                           0.481   Prob(JB):                        0.129
    Kurtosis:                       3.243   Cond. No.                         1.74
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
```


```python
data['y'] = y
```


```python
data[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.129468</td>
      <td>-1.212753</td>
      <td>0.504225</td>
      <td>0.427863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.302910</td>
      <td>-0.435742</td>
      <td>-0.254180</td>
      <td>-0.673480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.328522</td>
      <td>-0.025302</td>
      <td>0.138351</td>
      <td>-0.090878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.351475</td>
      <td>-0.719605</td>
      <td>-0.258215</td>
      <td>-0.489494</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.243269</td>
      <td>-0.373799</td>
      <td>-0.522629</td>
      <td>-0.128941</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
```


```python
results.params
```




    Intercept    0.033559
    col0         0.176149
    col1         0.224826
    col2         0.514808
    dtype: float64




```python
results.tvalues
```




    Intercept    0.952188
    col0         3.319754
    col1         4.850730
    col2         6.303971
    dtype: float64




```python
results.predict(data[:5])
```




    0   -0.002327
    1   -0.141904
    2    0.041226
    3   -0.323070
    4   -0.100535
    dtype: float64




```python
# 估计时间序列过程
```


```python
init_x = 4

import random
values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_x)
```


```python
MAXLAGS = 5
```


```python
model = sm.tsa.AR(values)
```


```python
results = model.fit(MAXLAGS)
```


```python
results.params
```




    array([-0.00616093,  0.78446347, -0.40847891, -0.01364148,  0.01496872,
            0.01429462])




```python
# 4.scikit-learn介绍
```


```python
# Kaggle竞赛的经典数据集，关于泰坦尼克号乘客的生还率
train = pd.read_csv('datasets/titanic/train.csv')
```


```python
test = pd.read_csv('datasets/titanic/test.csv')
```


```python
train[:4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64




```python
# 用训练数据集的中位数补全两个表的空值
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)
```


```python
# 增加一个列IsFemale，作为“Sex”列的编码
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
```


```python
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
```


```python
X_train[:5]
```




    array([[ 3.,  0., 22.],
           [ 1.,  1., 38.],
           [ 3.,  1., 26.],
           [ 1.,  1., 35.],
           [ 3.,  0., 35.]])




```python
y_train[:5]
```




    array([0, 1, 1, 1, 0], dtype=int64)




```python
# 用scikit-learn的LogisticRegression模型，创建一个模型实例
model = LogisticRegression()
```


```python
# 用模型的fit方法，将它拟合到训练数据
```


```python
model.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
# 用model.predict，对测试数据进行预测
```


```python
y_predict = model.predict(X_test)
```


```python
y_predict[:10]
```




    array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=int64)




```python
# 在实际中，模型训练经常有许多额外的复杂因素。
# 许多模型有可以调节的参数，有些方法（比如交叉验证）可以用来进行参数调节，
# 避免对训练数据过拟合。这通常可以提高预测性或对新数据的健壮性。

# 交叉验证通过分割训练数据来模拟样本外预测。
# 基于模型的精度得分（比如均方差），可以对模型参数进行网格搜索。
# 有些模型，如logistic回归，有内置的交叉验证的估计类。
```


```python
# logisticregressioncv类可以用一个参数指定网格搜索对模型的正则化参数C的粒度
```


```python
model_cv = LogisticRegressionCV(10)
```


```python
model_cv.fit(X_train, y_train)
```




    LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)




```python
model = LogisticRegression(C=10)
```


```python
scores = cross_val_score(model, X_train, y_train, cv=4)
```


```python
scores
```




    array([0.77232143, 0.80269058, 0.77027027, 0.78828829])


