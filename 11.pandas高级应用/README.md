

```python
import numpy as np
```


```python
import pandas as pd
```


```python
# 分类数据
```


```python
# 背景和目的
```


```python
values = pd.Series(['apple', 'orange', 'apple', 'apple'] * 2)
```


```python
values
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    dtype: object




```python
pd.unique(values)
```




    array(['apple', 'orange'], dtype=object)




```python
pd.value_counts(values)
```




    apple     6
    orange    2
    dtype: int64




```python
values = pd.Series([0, 1, 0, 0] * 2)
```


```python
dim = pd.Series(['apple', 'orange'])
```


```python
values
```




    0    0
    1    1
    2    0
    3    0
    4    0
    5    1
    6    0
    7    0
    dtype: int64




```python
dim
```




    0     apple
    1    orange
    dtype: object




```python
dim.take(values)
```




    0     apple
    1    orange
    0     apple
    0     apple
    0     apple
    1    orange
    0     apple
    0     apple
    dtype: object




```python
# pandas的分类类型
```


```python
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
```


```python
N = len(fruits)
```


```python
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': np.random.randint(3, 15, size=N),
                   'weight': np.random.uniform(0, 4, size=N)},
                   columns=['basket_id', 'fruit', 'count', 'weight'])
```


```python
df
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
      <th>basket_id</th>
      <th>fruit</th>
      <th>count</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>apple</td>
      <td>14</td>
      <td>0.639642</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>orange</td>
      <td>11</td>
      <td>1.113841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>apple</td>
      <td>5</td>
      <td>2.590931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>apple</td>
      <td>6</td>
      <td>1.280041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>apple</td>
      <td>8</td>
      <td>2.473221</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>orange</td>
      <td>12</td>
      <td>2.391871</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>apple</td>
      <td>10</td>
      <td>2.156193</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>apple</td>
      <td>6</td>
      <td>1.454934</td>
    </tr>
  </tbody>
</table>
</div>




```python
fruit_cat = df['fruit'].astype('category')
```


```python
fruit_cat
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    Name: fruit, dtype: category
    Categories (2, object): [apple, orange]




```python
c = fruit_cat.values
```


```python
type(c)
```




    pandas.core.arrays.categorical.Categorical




```python
c.categories
```




    Index(['apple', 'orange'], dtype='object')




```python
c.codes
```




    array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int8)




```python
# 将DataFrame的列通过分配转换结果，转换为分类
df['fruit'] = df['fruit'].astype('category')
```


```python
df.fruit
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    Name: fruit, dtype: category
    Categories (2, object): [apple, orange]




```python
# 从其它Python序列直接创建pandas.Categorical
my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
```


```python
my_categories
```




    [foo, bar, baz, foo, bar]
    Categories (3, object): [bar, baz, foo]




```python
categories = ['foo', 'bar', 'baz']
```


```python
codes = [0, 1, 2, 0, 0, 1]
```


```python
my_cats_2 = pd.Categorical.from_codes(codes, categories)
```


```python
my_cats_2
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo, bar, baz]




```python
ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)
```


```python
ordered_cat
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo < bar < baz]




```python
my_cats_2.as_ordered()
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo < bar < baz]




```python
# 用分类进行计算
```


```python
np.random.seed(12345)
```


```python
draws = np.random.randn(1000)
```


```python
draws[:5]
```




    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057])




```python
# 计算这个数据的分位面元，提取一些统计信息：
bins = pd.qcut(draws, 4)
```


```python
bins
```




    [(-0.684, -0.0101], (-0.0101, 0.63], (-0.684, -0.0101], (-0.684, -0.0101], (0.63, 3.928], ..., (-0.0101, 0.63], (-0.684, -0.0101], (-2.9499999999999997, -0.684], (-0.0101, 0.63], (0.63, 3.928]]
    Length: 1000
    Categories (4, interval[float64]): [(-2.9499999999999997, -0.684] < (-0.684, -0.0101] < (-0.0101, 0.63] < (0.63, 3.928]]




```python
bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```


```python
bins
```




    [Q2, Q3, Q2, Q2, Q4, ..., Q3, Q2, Q1, Q3, Q4]
    Length: 1000
    Categories (4, object): [Q1 < Q2 < Q3 < Q4]




```python
bins.codes[:10]
```




    array([1, 2, 1, 1, 3, 3, 2, 2, 3, 3], dtype=int8)




```python
# 使用groupby提取一些汇总信息
bins = pd.Series(bins, name='quartile')
```


```python
results = (pd.Series(draws).groupby(bins).agg(['count', 'min', 'max']).reset_index())
```


```python
results
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
      <th>quartile</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q1</td>
      <td>250</td>
      <td>-2.949343</td>
      <td>-0.685484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q2</td>
      <td>250</td>
      <td>-0.683066</td>
      <td>-0.010115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Q3</td>
      <td>250</td>
      <td>-0.010032</td>
      <td>0.628894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q4</td>
      <td>250</td>
      <td>0.634238</td>
      <td>3.927528</td>
    </tr>
  </tbody>
</table>
</div>




```python
results['quartile']
```




    0    Q1
    1    Q2
    2    Q3
    3    Q4
    Name: quartile, dtype: category
    Categories (4, object): [Q1 < Q2 < Q3 < Q4]




```python
# 用分类提高性能
```


```python
N = 10000000
```


```python
draws = pd.Series(np.random.randn(N))
```


```python
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))
```


```python
categories = labels.astype('category')
```


```python
# 标签使用的内存远比分类多
labels.memory_usage()
```




    80000080




```python
categories.memory_usage()
```




    10000272




```python
# 转换为分类的一次性的代价
%time _ = labels.astype('category')
```

    Wall time: 600 ms
    


```python
# 分类方法
```


```python
s = pd.Series(['a', 'b', 'c', 'd'] * 2)
```


```python
cat_s = s.astype('category')
```


```python
cat_s
```




    0    a
    1    b
    2    c
    3    d
    4    a
    5    b
    6    c
    7    d
    dtype: category
    Categories (4, object): [a, b, c, d]




```python
cat_s.cat.codes
```




    0    0
    1    1
    2    2
    3    3
    4    0
    5    1
    6    2
    7    3
    dtype: int8




```python
cat_s.cat.categories
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
actual_categories = ['a', 'b', 'c', 'd', 'e']
```


```python
cat_s2 = cat_s.cat.set_categories(actual_categories)
```


```python
cat_s2
```




    0    a
    1    b
    2    c
    3    d
    4    a
    5    b
    6    c
    7    d
    dtype: category
    Categories (5, object): [a, b, c, d, e]




```python
cat_s.value_counts()
```




    d    2
    c    2
    b    2
    a    2
    dtype: int64




```python
cat_s2.value_counts()
```




    d    2
    c    2
    b    2
    a    2
    e    0
    dtype: int64




```python
cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
```


```python
cat_s3
```




    0    a
    1    b
    4    a
    5    b
    dtype: category
    Categories (4, object): [a, b, c, d]




```python
cat_s3.cat.remove_unused_categories()
```




    0    a
    1    b
    4    a
    5    b
    dtype: category
    Categories (2, object): [a, b]




```python
# 为建模创建虚拟变量
```


```python
# 使用统计或机器学习工具时，通常会将分类数据转换为虚拟变量，也称为one-hot编码
```


```python
cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')
```


```python
pd.get_dummies(cat_s)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GroupBy高级应用
```


```python
# 分组转换和“解封”GroupBy
```


```python
# transform方法，与apply很像，但是对使用的函数有一定限制：
# 可以产生向分组形状广播标量值
# 可以产生一个和输入组形状相同的对象
# 不能修改输入
```


```python
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                   'value': np.arange(12.)})
```


```python
df
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
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = df.groupby('key').value
```


```python
g.mean()
```




    key
    a    4.5
    b    5.5
    c    6.5
    Name: value, dtype: float64




```python
g.transform(lambda x: x.mean())
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64




```python
g.transform('mean')
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64




```python
g.transform(lambda x: x * 2)
```




    0      0.0
    1      2.0
    2      4.0
    3      6.0
    4      8.0
    5     10.0
    6     12.0
    7     14.0
    8     16.0
    9     18.0
    10    20.0
    11    22.0
    Name: value, dtype: float64




```python
g.transform(lambda x: x.rank(ascending=False))
```




    0     4.0
    1     4.0
    2     4.0
    3     3.0
    4     3.0
    5     3.0
    6     2.0
    7     2.0
    8     2.0
    9     1.0
    10    1.0
    11    1.0
    Name: value, dtype: float64




```python
def normalize(x):
    return (x - x.mean()) / x.std()
```


```python
g.transform(normalize)
```




    0    -1.161895
    1    -1.161895
    2    -1.161895
    3    -0.387298
    4    -0.387298
    5    -0.387298
    6     0.387298
    7     0.387298
    8     0.387298
    9     1.161895
    10    1.161895
    11    1.161895
    Name: value, dtype: float64




```python
g.apply(normalize)
```




    0    -1.161895
    1    -1.161895
    2    -1.161895
    3    -0.387298
    4    -0.387298
    5    -0.387298
    6     0.387298
    7     0.387298
    8     0.387298
    9     1.161895
    10    1.161895
    11    1.161895
    Name: value, dtype: float64




```python
g.transform('mean')
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64




```python
# 解封（unwrapped）分组操作
normalized = (df['value'] - g.transform('mean')) / g.transform('std')
```


```python
normalized
```




    0    -1.161895
    1    -1.161895
    2    -1.161895
    3    -0.387298
    4    -0.387298
    5    -0.387298
    6     0.387298
    7     0.387298
    8     0.387298
    9     1.161895
    10    1.161895
    11    1.161895
    Name: value, dtype: float64




```python
# 分组的时间重采样
```


```python
N = 15
```


```python
times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
```


```python
df = pd.DataFrame({'time': times, 'value': np.arange(N)})
```


```python
df
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
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-05-20 00:01:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-05-20 00:02:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-05-20 00:03:00</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-05-20 00:04:00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-05-20 00:05:00</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-05-20 00:06:00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-05-20 00:07:00</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-05-20 00:08:00</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-05-20 00:09:00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-05-20 00:10:00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-05-20 00:11:00</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-05-20 00:12:00</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-05-20 00:13:00</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2017-05-20 00:14:00</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用time作为索引，重采样
df.set_index('time').resample('5min').count()
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
      <th>value</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-05-20 00:00:00</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2017-05-20 00:05:00</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2017-05-20 00:10:00</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a', 'b', 'c'], N),
                    'value': np.arange(N * 3.)})
```


```python
df2[:7]
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
      <th>time</th>
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-20 00:00:00</td>
      <td>a</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-05-20 00:00:00</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-05-20 00:00:00</td>
      <td>c</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-05-20 00:01:00</td>
      <td>a</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-05-20 00:01:00</td>
      <td>b</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-05-20 00:01:00</td>
      <td>c</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-05-20 00:02:00</td>
      <td>a</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
time_key = pd.TimeGrouper('5min')
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: pd.TimeGrouper is deprecated and will be removed; Please use pd.Grouper(freq=...)
      """Entry point for launching an IPython kernel.
    


```python
resampled = (df2.set_index('time').groupby(['key', time_key]).sum())
```


```python
resampled
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
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>key</th>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">a</th>
      <th>2017-05-20 00:00:00</th>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:05:00</th>
      <td>105.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:10:00</th>
      <td>180.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">b</th>
      <th>2017-05-20 00:00:00</th>
      <td>35.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:05:00</th>
      <td>110.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:10:00</th>
      <td>185.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">c</th>
      <th>2017-05-20 00:00:00</th>
      <td>40.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:05:00</th>
      <td>115.0</td>
    </tr>
    <tr>
      <th>2017-05-20 00:10:00</th>
      <td>190.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
resampled.reset_index()
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
      <th>key</th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>2017-05-20 00:00:00</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>2017-05-20 00:05:00</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2017-05-20 00:10:00</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>2017-05-20 00:00:00</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>2017-05-20 00:05:00</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>2017-05-20 00:10:00</td>
      <td>185.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c</td>
      <td>2017-05-20 00:00:00</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>c</td>
      <td>2017-05-20 00:05:00</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c</td>
      <td>2017-05-20 00:10:00</td>
      <td>190.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 链式编程技术
```


```python
# df = load_data()
# df2 = df[df['col2'] < 0]
# df2['col1_demeaned'] = df2['col1'] - df2['col1'].mean()
# result = df2.groupby('key').col1_demeaned.std()

# Usual non-functional way
# df2 = df.copy()
# df2['k'] = v
# DataFrame.assign方法是一个df[k] = v形式的函数式的列分配方法。
# 它不是就地修改对象，而是返回新的修改过的DataFrame。
# Functional assign way
# df2 = df.assign(k=v)
# 分配可能会比assign快，但是assign可以方便地进行链式编程
# result = (df2.assign(col1_demeaned=df2.col1 - df2.col2.mean()).groupby('key').col1_demeaned.std())
# df = (load_data()
#       [lambda x: x['col2'] < 0])
# 把整个过程写为一个单链表达式：

# result = (load_data()
#           [lambda x: x.col2 < 0]
#           .assign(col1_demeaned=lambda x: x.col1 - x.col1.mean())
#           .groupby('key')
#           .col1_demeaned.std())
```


```python
# 管道方法
```


```python
a = f(df, arg1=v1)
b = g(a, v2, arg3=v3)
c = h(b, arg4=v4)
# #### #
result = (df.pipe(f, arg1=v1)
          .pipe(g, v2, arg3=v3)
          .pipe(h, arg4=v4))
# # #
g = df.groupby(['key1', 'key2'])
df['col1'] = df['col1'] - g.transform('mean')

def group_demean(df, by, cols):
    result = df.copy()
    g = df.groupby(by)
    for c in cols:
        result[c] = df[c] - g[c].transform('mean')
    return result

result = (df[df.col1 < 0]
          .pipe(group_demean, ['key1', 'key2'], ['col1']))
```
