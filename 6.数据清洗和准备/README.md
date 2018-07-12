

```python
import pandas as pd
```


```python
import numpy as np
```


```python
from numpy import nan as NA
```


```python
import re
```


```python
# 处理缺失数据
```


```python
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
```


```python
string_data
```




    0     aardvark
    1    artichoke
    2          NaN
    3      avocado
    dtype: object




```python
string_data.isnull()
```




    0    False
    1    False
    2     True
    3    False
    dtype: bool




```python
string_data[0] = None
```


```python
string_data.isnull()
```




    0     True
    1    False
    2     True
    3    False
    dtype: bool




```python
# dropna  根据各标签的值中是否存在缺失数据对轴标签进行过滤 可通过阈值调节对缺失值的容忍度
# fillna  用指定值或插值方法填充缺失数据
# isnull  返回一个含有布尔值的对象 这些布尔值表示那些值是缺失值/NA 该对象的类型与元类型一样
# notnull  isnull 的否定式
```


```python
# 滤除缺失数据
```


```python
data = pd.Series([1, NA, 3.5, NA, 7])
```


```python
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
data[data.notnull()]
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
data[[True, True, False, False, True]]
```




    0    1.0
    1    NaN
    4    7.0
    dtype: float64




```python
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3.]])
```


```python
cleaned = data.dropna()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleaned
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[4] = NA
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(axis=1, how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame(np.random.randn(7, 3))
```


```python
df.iloc[:4, 1] = NA
```


```python
df.iloc[:2, 2] = NA
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.147915</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.381507</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.489601</td>
      <td>NaN</td>
      <td>1.347094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.480540</td>
      <td>NaN</td>
      <td>2.328446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1.489601</td>
      <td>NaN</td>
      <td>1.347094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.480540</td>
      <td>NaN</td>
      <td>2.328446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 填充缺失数据
```


```python
df.fillna(0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.147915</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.381507</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.489601</td>
      <td>0.000000</td>
      <td>1.347094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.480540</td>
      <td>0.000000</td>
      <td>2.328446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna({1: 0.5, 2: 0})
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.147915</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.381507</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.489601</td>
      <td>0.500000</td>
      <td>1.347094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.480540</td>
      <td>0.500000</td>
      <td>2.328446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = df.fillna(0, inplace=True)  # fillna默认会返回新对象，但也可以对现有对象进行就地修改
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.147915</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.381507</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.489601</td>
      <td>0.000000</td>
      <td>1.347094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.480540</td>
      <td>0.000000</td>
      <td>2.328446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.175996</td>
      <td>2.288694</td>
      <td>-0.764707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.397975</td>
      <td>-0.984505</td>
      <td>-0.352848</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.974693</td>
      <td>-2.086794</td>
      <td>-0.151966</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对reindexing有效的插值方法也可用于fillna
df = pd.DataFrame(np.random.randn(6, 3))
```


```python
df.iloc[2:, 1] = NA
```


```python
df.iloc[4:, 2] = NA
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.219523</td>
      <td>1.032883</td>
      <td>0.480712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.670486</td>
      <td>-0.277100</td>
      <td>-0.981239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.340329</td>
      <td>NaN</td>
      <td>0.752606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.885756</td>
      <td>NaN</td>
      <td>0.610340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.198402</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.160998</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.219523</td>
      <td>1.032883</td>
      <td>0.480712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.670486</td>
      <td>-0.277100</td>
      <td>-0.981239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.340329</td>
      <td>-0.277100</td>
      <td>0.752606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.885756</td>
      <td>-0.277100</td>
      <td>0.610340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.198402</td>
      <td>-0.277100</td>
      <td>0.610340</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.160998</td>
      <td>-0.277100</td>
      <td>0.610340</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill', limit=2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.219523</td>
      <td>1.032883</td>
      <td>0.480712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.670486</td>
      <td>-0.277100</td>
      <td>-0.981239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.340329</td>
      <td>-0.277100</td>
      <td>0.752606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.885756</td>
      <td>-0.277100</td>
      <td>0.610340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.198402</td>
      <td>NaN</td>
      <td>0.610340</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.160998</td>
      <td>NaN</td>
      <td>0.610340</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.Series([1., NA, 3.5, NA, 7])
```


```python
data.fillna(data.mean())
```




    0    1.000000
    1    3.833333
    2    3.500000
    3    3.833333
    4    7.000000
    dtype: float64




```python
# fillna
# value 用于填充缺失值的标量值或字典对象
# method 插值方法 如果函数调用时未指定其它参数的话 默认为'ffill'
# axis  待填充的轴, 默认axis=0
# inplace 修改调用者对象而不产生副本
# limit  (对于前向和后向填充) 可以连续填充的最大数量
```


```python
# 数据转换
```


```python
# 移除重复数据
```


```python
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.duplicated()  # 返回布尔型 Series前面是否出现过
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.drop_duplicates()  # 返回 DataFrame 重复数组标记为 False
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['v1'] = range(7)
```


```python
data.drop_duplicates(['k1'])
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# duplicated和drop_duplicates默认保留的是第一个出现的值组合。
# 传入keep='last'则保留最后一个
```


```python
# 利用函数或映射进行数据转换
```


```python
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                              'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
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
      <th>food</th>
      <th>ounces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
```


```python
lowercased = data['food'].str.lower()  # 使用str.lower 将各个值转换为小写
```


```python
lowercased
```




    0          bacon
    1    pulled pork
    2          bacon
    3       pastrami
    4    corned beef
    5          bacon
    6       pastrami
    7      honey ham
    8       nova lox
    Name: food, dtype: object




```python
data['animal'] = lowercased.map(meat_to_animal)
# map方法可以接受一个函数或含有映射关系的字典型对象
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
      <th>food</th>
      <th>ounces</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
      <td>salmon</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用map是一种实现元素级转换以及其他数据清理工作的便捷方式
data['food'].map(lambda x: meat_to_animal[x.lower()])
```




    0       pig
    1       pig
    2       pig
    3       cow
    4       cow
    5       pig
    6       cow
    7       pig
    8    salmon
    Name: food, dtype: object




```python
# 替换值
```


```python
data = pd.Series([1., -999., 2., -999., -1000., 3.])
```


```python
data
```




    0       1.0
    1    -999.0
    2       2.0
    3    -999.0
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace(-999, np.nan)  # 替换-999
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace([-999, -1000], np.nan)  # 替换-999 , -1000为nan
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    NaN
    5    3.0
    dtype: float64




```python
data.replace([-999, -1000], [np.nan, 0])  # 每个值替换不同值
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
data.replace({-999: np.nan, -1000: 0})  # 传入字典
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
# 重命名轴索引
```


```python
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
```


```python
transform = lambda x: x[:4].upper()
```


```python
data.index.map(transform)
```




    Index(['OHIO', 'COLO', 'NEW '], dtype='object')




```python
data.index = data.index.map(transform)
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 创建数据集的转换版（而不是修改原始数据）
data.rename(index=str.title, columns=str.upper)
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
      <th>ONE</th>
      <th>TWO</th>
      <th>THREE</th>
      <th>FOUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colo</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'})
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
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
```


```python
# 离散化和面元划分
```


```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
```


```python
bins = [18, 25, 35, 60, 100]
```


```python
cats = pd.cut(ages, bins)
```


```python
cats  # ages 所在 bins的区间
```




    [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]




```python
cats.codes  # ages 所在区间索引
```




    array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)




```python
cats.categories
```




    IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]]
                  closed='right',
                  dtype='interval[int64]')




```python
pd.value_counts(cats)  # 面元计数
```




    (18, 25]     5
    (35, 60]     3
    (25, 35]     3
    (60, 100]    1
    dtype: int64




```python
# right 默认True 是否包含右边最后一个值 (闭端)
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
```




    [[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
    Length: 12
    Categories (4, interval[int64]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]




```python
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
```


```python
pd.cut(ages, bins, labels=group_names)  # 设置面元名称
```




    [Youth, Youth, Youth, YoungAdult, Youth, ..., YoungAdult, Senior, MiddleAged, MiddleAged, YoungAdult]
    Length: 12
    Categories (4, object): [Youth < YoungAdult < MiddleAged < Senior]




```python
data = np.random.rand(20)
```


```python
pd.cut(data, 4, precision=2)  # 传入分区个数  4
```




    [(0.26, 0.5], (0.75, 0.99], (0.75, 0.99], (0.26, 0.5], (0.015, 0.26], ..., (0.26, 0.5], (0.015, 0.26], (0.015, 0.26], (0.5, 0.75], (0.26, 0.5]]
    Length: 20
    Categories (4, interval[float64]): [(0.015, 0.26] < (0.26, 0.5] < (0.5, 0.75] < (0.75, 0.99]]




```python
data = np.random.random(1000)
```


```python
cats = pd.qcut(data, 4)  # 根据样本分位数对数据进行面元划分
```


```python
cats
```




    [(0.755, 1.0], (0.515, 0.755], (0.00608, 0.27], (0.27, 0.515], (0.00608, 0.27], ..., (0.27, 0.515], (0.515, 0.755], (0.27, 0.515], (0.515, 0.755], (0.755, 1.0]]
    Length: 1000
    Categories (4, interval[float64]): [(0.00608, 0.27] < (0.27, 0.515] < (0.515, 0.755] < (0.755, 1.0]]




```python
pd.value_counts(cats)
```




    (0.755, 1.0]       250
    (0.515, 0.755]     250
    (0.27, 0.515]      250
    (0.00608, 0.27]    250
    dtype: int64




```python
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
```




    [(0.912, 1.0], (0.515, 0.912], (0.00608, 0.119], (0.119, 0.515], (0.00608, 0.119], ..., (0.119, 0.515], (0.515, 0.912], (0.119, 0.515], (0.515, 0.912], (0.912, 1.0]]
    Length: 1000
    Categories (4, interval[float64]): [(0.00608, 0.119] < (0.119, 0.515] < (0.515, 0.912] < (0.912, 1.0]]




```python
# 检测和过滤异常值
```


```python
data = pd.DataFrame(np.random.randn(1000, 4))
```


```python
data.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.011805</td>
      <td>0.010797</td>
      <td>-0.002624</td>
      <td>0.005651</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.988832</td>
      <td>1.000983</td>
      <td>1.000650</td>
      <td>0.972372</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.947519</td>
      <td>-3.649215</td>
      <td>-2.715677</td>
      <td>-3.941866</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.709498</td>
      <td>-0.679148</td>
      <td>-0.688437</td>
      <td>-0.653982</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.005146</td>
      <td>0.011318</td>
      <td>-0.023763</td>
      <td>0.045819</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.712202</td>
      <td>0.721964</td>
      <td>0.685699</td>
      <td>0.703592</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.065713</td>
      <td>3.181983</td>
      <td>3.967068</td>
      <td>2.884123</td>
    </tr>
  </tbody>
</table>
</div>




```python
col = data[2]
```


```python
col[np.abs(col) > 3]  # 找出某列中绝对值大小超过3的值
```




    181    3.744970
    913    3.967068
    Name: 2, dtype: float64




```python
data[(np.abs(data) > 3).any(1)]  # 选出全部含有“超过3或－3的值”的行
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>129</th>
      <td>3.065713</td>
      <td>-0.360436</td>
      <td>-1.921476</td>
      <td>0.496130</td>
    </tr>
    <tr>
      <th>178</th>
      <td>0.226706</td>
      <td>3.181983</td>
      <td>0.304138</td>
      <td>-0.720813</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.083472</td>
      <td>-1.161882</td>
      <td>3.744970</td>
      <td>-0.045521</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0.574429</td>
      <td>-3.649215</td>
      <td>-0.246139</td>
      <td>-0.207614</td>
    </tr>
    <tr>
      <th>452</th>
      <td>-0.859680</td>
      <td>-3.563828</td>
      <td>0.103302</td>
      <td>-0.793543</td>
    </tr>
    <tr>
      <th>716</th>
      <td>-0.516102</td>
      <td>0.270339</td>
      <td>1.079534</td>
      <td>-3.941866</td>
    </tr>
    <tr>
      <th>913</th>
      <td>-1.794112</td>
      <td>0.982514</td>
      <td>3.967068</td>
      <td>-1.741257</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[np.abs(data) > 3] = np.sign(data) * 3  # 将值限制在区间－3到3以内
```


```python
data.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.011870</td>
      <td>0.011828</td>
      <td>-0.004336</td>
      <td>0.006593</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.988630</td>
      <td>0.996389</td>
      <td>0.994744</td>
      <td>0.968995</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.947519</td>
      <td>-3.000000</td>
      <td>-2.715677</td>
      <td>-3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.709498</td>
      <td>-0.679148</td>
      <td>-0.688437</td>
      <td>-0.653982</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.005146</td>
      <td>0.011318</td>
      <td>-0.023763</td>
      <td>0.045819</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.712202</td>
      <td>0.721964</td>
      <td>0.685699</td>
      <td>0.703592</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.884123</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.sign(data).head()  # 根据数据的值是正还是负，生成1和-1
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 排列和随机采样
```


```python
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
```


```python
sampler = np.random.permutation(5)  # permuting，随机重排序
```


```python
sampler
```




    array([3, 2, 4, 0, 1])




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.take(sampler)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=3)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
choices = pd.Series([5, 7, -1, 6, 4])
```


```python
draws = choices.sample(n=10, replace=True) 
# 要通过替换的方式产生样本（允许重复选择），可以传递replace=True到sample
```


```python
draws
```




    1    7
    2   -1
    2   -1
    4    4
    1    7
    2   -1
    2   -1
    1    7
    0    5
    3    6
    dtype: int64




```python
# 计算指标/哑变量
```


```python
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
```


```python
pd.get_dummies(df['key'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies = pd.get_dummies(df['key'], prefix='key')
```


```python
df_with_dummy = df[['data1']].join(dummies)
```


```python
df_with_dummy
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
      <th>data1</th>
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mnames = ['movie_id', 'title', 'genres']
```


```python
movies = pd.read_table('examples/movielens/movies.dat', sep='::', header=None, names=mnames)
```

    c:\program files\python36\lib\site-packages\ipykernel_launcher.py:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      """Entry point for launching an IPython kernel.
    


```python
movies[:10]
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
      <th>movie_id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck (1995)</td>
      <td>Adventure|Children's</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death (1995)</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_genres = []
```


```python
for x in movies.genres:
    all_genres.extend(x.split('|'))
```


```python
genres = pd.unique(all_genres)
```


```python
genres
```




    array(['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy',
           'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror',
           'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir',
           'Western'], dtype=object)




```python
zero_matrix = np.zeros((len(movies), len(genres)))
```


```python
dummies = pd.DataFrame(zero_matrix, columns=genres)
```


```python
gen = movies.genres[0]
```


```python
gen.split('|')
```




    ['Animation', "Children's", 'Comedy']




```python
dummies.columns.get_indexer(gen.split('|'))
```




    array([0, 1, 2], dtype=int64)




```python
for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1
```


```python
movies_windic = movies.join(dummies.add_prefix('Genre_'))
```


```python
movies_windic.iloc[0]
```




    movie_id                                       1
    title                           Toy Story (1995)
    genres               Animation|Children's|Comedy
    Genre_Animation                                1
    Genre_Children's                               1
    Genre_Comedy                                   1
    Genre_Adventure                                0
    Genre_Fantasy                                  0
    Genre_Romance                                  0
    Genre_Drama                                    0
    Genre_Action                                   0
    Genre_Crime                                    0
    Genre_Thriller                                 0
    Genre_Horror                                   0
    Genre_Sci-Fi                                   0
    Genre_Documentary                              0
    Genre_War                                      0
    Genre_Musical                                  0
    Genre_Mystery                                  0
    Genre_Film-Noir                                0
    Genre_Western                                  0
    Name: 0, dtype: object




```python
np.random.seed(12345)
```


```python
values = np.random.rand(10)
```


```python
values
```




    array([0.92961609, 0.31637555, 0.18391881, 0.20456028, 0.56772503,
           0.5955447 , 0.96451452, 0.6531771 , 0.74890664, 0.65356987])




```python
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
```


```python
pd.get_dummies(pd.cut(values, bins))
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
      <th>(0.0, 0.2]</th>
      <th>(0.2, 0.4]</th>
      <th>(0.4, 0.6]</th>
      <th>(0.6, 0.8]</th>
      <th>(0.8, 1.0]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 字符串操作
```


```python
# 字符串对象方法
```


```python
val = 'a,b, guido'
```


```python
val.split(',')
```




    ['a', 'b', ' guido']




```python
pieces = [x.strip() for x in val.split(',')]
```


```python
pieces
```




    ['a', 'b', 'guido']




```python
first, second, third = pieces
```


```python
first + '::' + second + '::' + third
```




    'a::b::guido'




```python
'::'.join(pieces)
```




    'a::b::guido'




```python
'guido' in val
```




    True




```python
val.index(',')
```




    1




```python
val.find(':')
```




    -1




```python
val.index(':')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-154-2c016e7367ac> in <module>()
    ----> 1 val.index(':')
    

    ValueError: substring not found



```python
val.count(',')
```




    2




```python
val.replace(',', '::')
```




    'a::b:: guido'




```python
val.replace(',', '')
```




    'ab guido'




```python
# 正则表达式
```


```python
text = 'foo    bar\t baz  \tqux'
```


```python
re.split('\s+', text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex = re.compile('\s+')
```


```python
regex.split(text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex.findall(text)
```




    ['    ', '\t ', '  \t']




```python
text = '''Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com'''
```


```python
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
```


```python
regex = re.compile(pattern, flags=re.IGNORECASE)
```


```python
regex.findall(text)
```




    ['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', 'ryan@yahoo.com']




```python
m = regex.search(text)
```


```python
m
```




    <_sre.SRE_Match object; span=(5, 20), match='dave@google.com'>




```python
text[m.start():m.end()]
```




    'dave@google.com'




```python
print(regex.match(text))
```

    None
    


```python
print(regex.sub('REDACTED', text))
```

    Dave REDACTED
    Steve REDACTED
    Rob REDACTED
    Ryan REDACTED
    


```python
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
```


```python
regex = re.compile(pattern, flags=re.IGNORECASE)
```


```python
m = regex.match('wesm@bright.net')
```


```python
m.groups()
```




    ('wesm', 'bright', 'net')




```python
regex.findall(text)
```




    [('dave', 'google', 'com'),
     ('steve', 'gmail', 'com'),
     ('rob', 'gmail', 'com'),
     ('ryan', 'yahoo', 'com')]




```python
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
```

    Dave Username: dave, Domain: google, Suffix: com
    Steve Username: steve, Domain: gmail, Suffix: com
    Rob Username: rob, Domain: gmail, Suffix: com
    Ryan Username: ryan, Domain: yahoo, Suffix: com
    


```python
# pandas的矢量化字符串函数
```


```python
data = {'Dave': 'dave@google.com',
        'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com',
        'Wes': np.nan}
```


```python
data = pd.Series(data)
```


```python
data
```




    Dave     dave@google.com
    Steve    steve@gmail.com
    Rob        rob@gmail.com
    Wes                  NaN
    dtype: object




```python
data.isnull()
```




    Dave     False
    Steve    False
    Rob      False
    Wes       True
    dtype: bool




```python
# 通过data.map，所有字符串和正则表达式方法都能被应用于（传入lambda表达式或其他函数）各个值
# 但是如果存在NA（null）就会报错
```


```python
data.str.contains('gmail')
```




    Dave     False
    Steve     True
    Rob       True
    Wes        NaN
    dtype: object




```python
pattern
```




    '([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})'




```python
data.str.findall(pattern, flags=re.IGNORECASE)
```




    Dave     [(dave, google, com)]
    Steve    [(steve, gmail, com)]
    Rob        [(rob, gmail, com)]
    Wes                        NaN
    dtype: object




```python
matches = data.str.match(pattern, flags=re.IGNORECASE)
```


```python
matches
```




    Dave     True
    Steve    True
    Rob      True
    Wes       NaN
    dtype: object




```python
matches.str.get(1)
```




    Dave    NaN
    Steve   NaN
    Rob     NaN
    Wes     NaN
    dtype: float64




```python
matches.str[0]
```




    Dave    NaN
    Steve   NaN
    Rob     NaN
    Wes     NaN
    dtype: float64




```python
data.str[:5]
```




    Dave     dave@
    Steve    steve
    Rob      rob@g
    Wes        NaN
    dtype: object




```python
# pandas 对象方法
# cat  实现元素级的字符串连接操作, 可指定分隔符
# count  返回表示个字符串是否含有指定模式的布尔型数组
# extract  使用带分组的正则表达式从字符串Series提取一个或多个字符串, 结果是一个DataFrame, 每组有一列
# endswith  相对于每个元素执行 x.endswith(pattern)
# startswith  相对于每个元素执行 x.startswith(pattern)
# findall  计算各个字符串的模式列表
# get  获取各个元素的第i个字符串
# isalnum  相当于内置的 str.alnum
# isalpha  相当于内置的 str.isalpha
# isdecimal  相当于内置的 str.isdecimal
# isdigit  相当于内置的 str.isdigit
# islower  相当于内置的 str.islower
# isnumeric  相当于内置的 str.isnumeric
# isupper   相当于内置的 str.isupper
# join  根据指定的分隔符将 Series中各个元素的字符串连接起来
# len  计算各个字符串的长度
# lower, upper 转换大小写 相当于对各个元素执行 x.lower()或 x.upper()
# match  根据指定的正则表达式对各个元素执行re.mathc, 返回匹配的组为列表
# pad  在字符串的左边, 右边或两边添加空白符
# center  相当于 pad(side='both')
# repeat  重复值.
# replace  用指定字符串替换找到的模式
# slice  对Series中的各个子串进行子串截取
# split  根据分割符或正则表达式对字符串进行拆分
# strip  去除两边的空白符, 包括新行
# rstrip  去除右边空白符
# lstrip  去除左边空白符
```
