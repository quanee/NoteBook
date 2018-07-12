

```python
import pandas as pd
```


```python
import numpy as np
```


```python
import statsmodels.api as sm
```


```python
# GroupBy机制
```


```python
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
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
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>-0.971903</td>
      <td>-0.123304</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>-1.008096</td>
      <td>2.824124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>1.619547</td>
      <td>-0.688791</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>-1.505993</td>
      <td>0.031858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>1.001912</td>
      <td>0.180842</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = df['data1'].groupby(df['key1'])
```


```python
grouped
```




    <pandas.core.groupby.groupby.SeriesGroupBy object at 0x0000029E15458F60>




```python
grouped.mean()  # 分组平均值
```




    key1
    a   -0.326029
    b    0.056777
    Name: data1, dtype: float64




```python
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
```


```python
means
```




    key1  key2
    a     one     0.015004
          two    -1.008096
    b     one     1.619547
          two    -1.505993
    Name: data1, dtype: float64




```python
means.unstack()
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
      <th>key2</th>
      <th>one</th>
      <th>two</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.015004</td>
      <td>-1.008096</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.619547</td>
      <td>-1.505993</td>
    </tr>
  </tbody>
</table>
</div>




```python
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
```


```python
years = np.array([2005, 2005, 2006, 2005, 2006])
```


```python
df['data1'].groupby([states, years]).mean()
```




    California  2005   -1.008096
                2006    1.619547
    Ohio        2005   -1.238948
                2006    1.001912
    Name: data1, dtype: float64




```python
df.groupby('key1').mean()
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
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.326029</td>
      <td>0.960554</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.056777</td>
      <td>-0.328467</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['key1', 'key2']).mean()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>one</th>
      <td>0.015004</td>
      <td>0.028769</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-1.008096</td>
      <td>2.824124</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>one</th>
      <td>1.619547</td>
      <td>-0.688791</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-1.505993</td>
      <td>0.031858</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['key1', 'key2']).size()
```




    key1  key2
    a     one     2
          two     1
    b     one     1
          two     1
    dtype: int64




```python
# 对分组进行迭代
```


```python
for name, group in df.groupby('key1'):
    print(name)
    print(group)
```

    a
      key1 key2     data1     data2
    0    a  one -0.971903 -0.123304
    1    a  two -1.008096  2.824124
    4    a  one  1.001912  0.180842
    b
      key1 key2     data1     data2
    2    b  one  1.619547 -0.688791
    3    b  two -1.505993  0.031858
    


```python
for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)
```

    ('a', 'one')
      key1 key2     data1     data2
    0    a  one -0.971903 -0.123304
    4    a  one  1.001912  0.180842
    ('a', 'two')
      key1 key2     data1     data2
    1    a  two -1.008096  2.824124
    ('b', 'one')
      key1 key2     data1     data2
    2    b  one  1.619547 -0.688791
    ('b', 'two')
      key1 key2     data1     data2
    3    b  two -1.505993  0.031858
    


```python
pieces = dict(list(df.groupby('key1')))
```


```python
pieces['b']
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>1.619547</td>
      <td>-0.688791</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>-1.505993</td>
      <td>0.031858</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    key1      object
    key2      object
    data1    float64
    data2    float64
    dtype: object




```python
grouped = df.groupby(df.dtypes, axis=1)
```


```python
for dtype, group in grouped:
    print(dtype)
    print(group)
```

    float64
          data1     data2
    0 -0.971903 -0.123304
    1 -1.008096  2.824124
    2  1.619547 -0.688791
    3 -1.505993  0.031858
    4  1.001912  0.180842
    object
      key1 key2
    0    a  one
    1    a  two
    2    b  one
    3    b  two
    4    a  one
    


```python
# 选取一列或列的子集
```


```python
df.groupby('key1')['data1']
```




    <pandas.core.groupby.groupby.SeriesGroupBy object at 0x0000029E1549BE48>




```python
df['data1'].groupby(df['key1'])  # 语法糖
```




    <pandas.core.groupby.groupby.SeriesGroupBy object at 0x0000029E1549B748>




```python
s_grouped = df.groupby(['key1', 'key2'])['data2']
```


```python
s_grouped
```




    <pandas.core.groupby.groupby.SeriesGroupBy object at 0x0000029E154BC128>




```python
s_grouped.mean()
```




    key1  key2
    a     one     0.028769
          two     2.824124
    b     one    -0.688791
          two     0.031858
    Name: data2, dtype: float64




```python
# 通过字典或Series进行分组
```


```python
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
```


```python
people.iloc[2:3, [1, 2]] = np.nan  # 添加 NA值
```


```python
people
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
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Joe</th>
      <td>-1.180347</td>
      <td>-0.991097</td>
      <td>0.159158</td>
      <td>-1.589048</td>
      <td>-1.737739</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>0.974448</td>
      <td>0.884664</td>
      <td>0.482729</td>
      <td>1.124062</td>
      <td>-0.743893</td>
    </tr>
    <tr>
      <th>Wes</th>
      <td>-1.232817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.178278</td>
      <td>-0.008922</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>-0.794304</td>
      <td>1.408775</td>
      <td>0.106913</td>
      <td>0.432857</td>
      <td>-0.641837</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>1.662231</td>
      <td>-0.361089</td>
      <td>-0.762428</td>
      <td>0.165612</td>
      <td>0.114125</td>
    </tr>
  </tbody>
</table>
</div>




```python
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}
```


```python
by_column = people.groupby(mapping, axis=1)
```


```python
by_column.sum()
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
      <th>blue</th>
      <th>red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Joe</th>
      <td>-1.429890</td>
      <td>-3.909182</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>1.606791</td>
      <td>1.115219</td>
    </tr>
    <tr>
      <th>Wes</th>
      <td>0.178278</td>
      <td>-1.241738</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>0.539770</td>
      <td>-0.027367</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>-0.596816</td>
      <td>1.415267</td>
    </tr>
  </tbody>
</table>
</div>




```python
map_series = pd.Series(mapping)
```


```python
map_series
```




    a       red
    b       red
    c      blue
    d      blue
    e       red
    f    orange
    dtype: object




```python
people.groupby(map_series, axis=1).count()
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
      <th>blue</th>
      <th>red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Joe</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Wes</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 通过函数进行分组
```


```python
people.groupby(len).sum()
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
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>-3.207467</td>
      <td>0.417678</td>
      <td>0.266071</td>
      <td>-0.977913</td>
      <td>-2.388498</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.974448</td>
      <td>0.884664</td>
      <td>0.482729</td>
      <td>1.124062</td>
      <td>-0.743893</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.662231</td>
      <td>-0.361089</td>
      <td>-0.762428</td>
      <td>0.165612</td>
      <td>0.114125</td>
    </tr>
  </tbody>
</table>
</div>




```python
key_list = ['one', 'one', 'one', 'two', 'two']
```


```python
people.groupby([len, key_list]).min()
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>one</th>
      <td>-1.232817</td>
      <td>-0.991097</td>
      <td>0.159158</td>
      <td>-1.589048</td>
      <td>-1.737739</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.794304</td>
      <td>1.408775</td>
      <td>0.106913</td>
      <td>0.432857</td>
      <td>-0.641837</td>
    </tr>
    <tr>
      <th>5</th>
      <th>one</th>
      <td>0.974448</td>
      <td>0.884664</td>
      <td>0.482729</td>
      <td>1.124062</td>
      <td>-0.743893</td>
    </tr>
    <tr>
      <th>6</th>
      <th>two</th>
      <td>1.662231</td>
      <td>-0.361089</td>
      <td>-0.762428</td>
      <td>0.165612</td>
      <td>0.114125</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 根据索引级别分组
```


```python
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                     [1, 3, 5, 1, 3]],
                                     names=['cty', 'tenor'])
```


```python
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
```


```python
hier_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>cty</th>
      <th colspan="3" halign="left">US</th>
      <th colspan="2" halign="left">JP</th>
    </tr>
    <tr>
      <th>tenor</th>
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>1</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.077423</td>
      <td>-0.806452</td>
      <td>-0.691562</td>
      <td>1.144406</td>
      <td>2.560567</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.614354</td>
      <td>0.101818</td>
      <td>-1.274585</td>
      <td>-1.137360</td>
      <td>-1.213281</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.039988</td>
      <td>0.776125</td>
      <td>-1.743783</td>
      <td>1.158421</td>
      <td>0.426746</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.087590</td>
      <td>0.167167</td>
      <td>1.785513</td>
      <td>-0.094097</td>
      <td>-0.192900</td>
    </tr>
  </tbody>
</table>
</div>




```python
hier_df.groupby(level='cty', axis=1).count()
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
      <th>cty</th>
      <th>JP</th>
      <th>US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据聚集
```


```python
# 经过优化的groupby方法
# count  分组中的非NA值得数量
# sum  非 NA值的和
# mean  非 NA值的平均值
# median  非 NA值的算术中位数
# std, var  无偏(分母为n-1)标准差和方差
# min, max  非 NA值的最大值和最小值
# prod  非 NA值的积
# first, last  第一个和最后一个非 NA值
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
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>-0.971903</td>
      <td>-0.123304</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>-1.008096</td>
      <td>2.824124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>1.619547</td>
      <td>-0.688791</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>-1.505993</td>
      <td>0.031858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>1.001912</td>
      <td>0.180842</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = df.groupby('key1')
```


```python
grouped['data1'].quantile(0.9)  # quantile样本分位数
```




    key1
    a    0.607149
    b    1.306993
    Name: data1, dtype: float64




```python
def peak_to_peak(arr):
    return arr.max() - arr.min()
```


```python
grouped.agg(peak_to_peak)
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
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.010008</td>
      <td>2.947427</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.125541</td>
      <td>0.720649</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped.aggregate(peak_to_peak)  # 自定义聚合函数 比优化过的函数慢
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
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.010008</td>
      <td>2.947427</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.125541</td>
      <td>0.720649</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">data1</th>
      <th colspan="8" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>3.0</td>
      <td>-0.326029</td>
      <td>1.150173</td>
      <td>-1.008096</td>
      <td>-0.989999</td>
      <td>-0.971903</td>
      <td>0.015004</td>
      <td>1.001912</td>
      <td>3.0</td>
      <td>0.960554</td>
      <td>1.621047</td>
      <td>-0.123304</td>
      <td>0.028769</td>
      <td>0.180842</td>
      <td>1.502483</td>
      <td>2.824124</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>0.056777</td>
      <td>2.210091</td>
      <td>-1.505993</td>
      <td>-0.724608</td>
      <td>0.056777</td>
      <td>0.838162</td>
      <td>1.619547</td>
      <td>2.0</td>
      <td>-0.328467</td>
      <td>0.509576</td>
      <td>-0.688791</td>
      <td>-0.508629</td>
      <td>-0.328467</td>
      <td>-0.148304</td>
      <td>0.031858</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 面向列的多函数应用
```


```python
tips = pd.read_csv('examples/tips.csv')
```


```python
tips['tip_pct'] = tips['tip'] / tips['total_bill']
```


```python
tips[:6]
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.059447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.160542</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.166587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.139780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.146808</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25.29</td>
      <td>4.71</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.186240</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = tips.groupby(['day', 'smoker'])
```


```python
grouped_pct = grouped['tip_pct']
```


```python
grouped_pct.agg('mean')
```




    day   smoker
    Fri   No        0.151650
          Yes       0.174783
    Sat   No        0.158048
          Yes       0.147906
    Sun   No        0.160113
          Yes       0.187250
    Thur  No        0.160298
          Yes       0.163863
    Name: tip_pct, dtype: float64




```python
# 传入一组函数或函数名 得到的DataFrame的列就会以相应的函数命名
grouped_pct.agg(['mean', 'std', peak_to_peak])
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
      <th>mean</th>
      <th>std</th>
      <th>peak_to_peak</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.151650</td>
      <td>0.028123</td>
      <td>0.067349</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.174783</td>
      <td>0.051293</td>
      <td>0.159925</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.158048</td>
      <td>0.039767</td>
      <td>0.235193</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.147906</td>
      <td>0.061375</td>
      <td>0.290095</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.160113</td>
      <td>0.042347</td>
      <td>0.193226</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.187250</td>
      <td>0.154134</td>
      <td>0.644685</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.160298</td>
      <td>0.038774</td>
      <td>0.193350</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.163863</td>
      <td>0.039389</td>
      <td>0.151240</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果传入的是一个由(name,function)元组组成的列表，
# 则各元组的第一个元素就会被用作DataFrame的列名
# 可以将这种二元元组列表看做一个有序映射
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])
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
      <th>foo</th>
      <th>bar</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.151650</td>
      <td>0.028123</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.174783</td>
      <td>0.051293</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.158048</td>
      <td>0.039767</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.147906</td>
      <td>0.061375</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.160113</td>
      <td>0.042347</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.187250</td>
      <td>0.154134</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.160298</td>
      <td>0.038774</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.163863</td>
      <td>0.039389</td>
    </tr>
  </tbody>
</table>
</div>




```python
functions = ['count', 'mean', 'max']  # 函数列表
```


```python
result = grouped['tip_pct', 'total_bill'].agg(functions)  # 对多个列各进行聚合
```


```python
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">tip_pct</th>
      <th colspan="3" halign="left">total_bill</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>4</td>
      <td>0.151650</td>
      <td>0.187735</td>
      <td>4</td>
      <td>18.420000</td>
      <td>22.75</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>15</td>
      <td>0.174783</td>
      <td>0.263480</td>
      <td>15</td>
      <td>16.813333</td>
      <td>40.17</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>45</td>
      <td>0.158048</td>
      <td>0.291990</td>
      <td>45</td>
      <td>19.661778</td>
      <td>48.33</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>42</td>
      <td>0.147906</td>
      <td>0.325733</td>
      <td>42</td>
      <td>21.276667</td>
      <td>50.81</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>57</td>
      <td>0.160113</td>
      <td>0.252672</td>
      <td>57</td>
      <td>20.506667</td>
      <td>48.17</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>19</td>
      <td>0.187250</td>
      <td>0.710345</td>
      <td>19</td>
      <td>24.120000</td>
      <td>45.35</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>45</td>
      <td>0.160298</td>
      <td>0.266312</td>
      <td>45</td>
      <td>17.113111</td>
      <td>41.19</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>17</td>
      <td>0.163863</td>
      <td>0.241255</td>
      <td>17</td>
      <td>19.190588</td>
      <td>43.11</td>
    </tr>
  </tbody>
</table>
</div>




```python
result['tip_pct']
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
      <th>count</th>
      <th>mean</th>
      <th>max</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>4</td>
      <td>0.151650</td>
      <td>0.187735</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>15</td>
      <td>0.174783</td>
      <td>0.263480</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>45</td>
      <td>0.158048</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>42</td>
      <td>0.147906</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>57</td>
      <td>0.160113</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>19</td>
      <td>0.187250</td>
      <td>0.710345</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>45</td>
      <td>0.160298</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>17</td>
      <td>0.163863</td>
      <td>0.241255</td>
    </tr>
  </tbody>
</table>
</div>




```python
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
```


```python
grouped['tip_pct', 'total_bill'].agg(ftuples)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">tip_pct</th>
      <th colspan="2" halign="left">total_bill</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Durchschnitt</th>
      <th>Abweichung</th>
      <th>Durchschnitt</th>
      <th>Abweichung</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.151650</td>
      <td>0.000791</td>
      <td>18.420000</td>
      <td>25.596333</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.174783</td>
      <td>0.002631</td>
      <td>16.813333</td>
      <td>82.562438</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.158048</td>
      <td>0.001581</td>
      <td>19.661778</td>
      <td>79.908965</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.147906</td>
      <td>0.003767</td>
      <td>21.276667</td>
      <td>101.387535</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.160113</td>
      <td>0.001793</td>
      <td>20.506667</td>
      <td>66.099980</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.187250</td>
      <td>0.023757</td>
      <td>24.120000</td>
      <td>109.046044</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.160298</td>
      <td>0.001503</td>
      <td>17.113111</td>
      <td>59.625081</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.163863</td>
      <td>0.001551</td>
      <td>19.190588</td>
      <td>69.808518</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped.agg({'tip' : np.max, 'size' : 'sum'})  # 对一个列或不同的列应用不同的函数
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
      <th>tip</th>
      <th>size</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>3.50</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>4.73</td>
      <td>31</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>9.00</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>10.00</td>
      <td>104</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>6.00</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>6.50</td>
      <td>49</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>6.70</td>
      <td>112</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>5.00</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 只有将多个函数应用到至少一列时，DataFrame才会拥有层次化的列。
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'], 'size' : 'sum'})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">tip_pct</th>
      <th>size</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.120385</td>
      <td>0.187735</td>
      <td>0.151650</td>
      <td>0.028123</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.103555</td>
      <td>0.263480</td>
      <td>0.174783</td>
      <td>0.051293</td>
      <td>31</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.056797</td>
      <td>0.291990</td>
      <td>0.158048</td>
      <td>0.039767</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.035638</td>
      <td>0.325733</td>
      <td>0.147906</td>
      <td>0.061375</td>
      <td>104</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.059447</td>
      <td>0.252672</td>
      <td>0.160113</td>
      <td>0.042347</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.065660</td>
      <td>0.710345</td>
      <td>0.187250</td>
      <td>0.154134</td>
      <td>49</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.072961</td>
      <td>0.266312</td>
      <td>0.160298</td>
      <td>0.038774</td>
      <td>112</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.090014</td>
      <td>0.241255</td>
      <td>0.163863</td>
      <td>0.039389</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 以“没有行索引”的形式返回聚合数据
```


```python
# 对结果调用reset_index也能得到这种形式的结果
tips.groupby(['day', 'smoker'], as_index=False).mean()
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
      <th>day</th>
      <th>smoker</th>
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>No</td>
      <td>18.420000</td>
      <td>2.812500</td>
      <td>2.250000</td>
      <td>0.151650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Yes</td>
      <td>16.813333</td>
      <td>2.714000</td>
      <td>2.066667</td>
      <td>0.174783</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>No</td>
      <td>19.661778</td>
      <td>3.102889</td>
      <td>2.555556</td>
      <td>0.158048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Yes</td>
      <td>21.276667</td>
      <td>2.875476</td>
      <td>2.476190</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>No</td>
      <td>20.506667</td>
      <td>3.167895</td>
      <td>2.929825</td>
      <td>0.160113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Yes</td>
      <td>24.120000</td>
      <td>3.516842</td>
      <td>2.578947</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>No</td>
      <td>17.113111</td>
      <td>2.673778</td>
      <td>2.488889</td>
      <td>0.160298</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Yes</td>
      <td>19.190588</td>
      <td>3.030000</td>
      <td>2.352941</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>




```python
# apply：一般性的“拆分－应用－合并”
```


```python
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]
```


```python
top(tips, n=6)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>109</th>
      <td>14.31</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.279525</td>
    </tr>
    <tr>
      <th>183</th>
      <td>23.17</td>
      <td>6.50</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.280535</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.groupby('smoker').apply(top)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
    <tr>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">No</th>
      <th>88</th>
      <td>24.71</td>
      <td>5.85</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.236746</td>
    </tr>
    <tr>
      <th>185</th>
      <td>20.69</td>
      <td>5.00</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>5</td>
      <td>0.241663</td>
    </tr>
    <tr>
      <th>51</th>
      <td>10.29</td>
      <td>2.60</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th>149</th>
      <td>7.51</td>
      <td>2.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Yes</th>
      <th>109</th>
      <td>14.31</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.279525</td>
    </tr>
    <tr>
      <th>183</th>
      <td>23.17</td>
      <td>6.50</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.280535</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
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
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
    <tr>
      <th>smoker</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">No</th>
      <th>Fri</th>
      <th>94</th>
      <td>22.75</td>
      <td>3.25</td>
      <td>No</td>
      <td>Fri</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Sat</th>
      <th>212</th>
      <td>48.33</td>
      <td>9.00</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.186220</td>
    </tr>
    <tr>
      <th>Sun</th>
      <th>156</th>
      <td>48.17</td>
      <td>5.00</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>6</td>
      <td>0.103799</td>
    </tr>
    <tr>
      <th>Thur</th>
      <th>142</th>
      <td>41.19</td>
      <td>5.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>5</td>
      <td>0.121389</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Yes</th>
      <th>Fri</th>
      <th>95</th>
      <td>40.17</td>
      <td>4.73</td>
      <td>Yes</td>
      <td>Fri</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.117750</td>
    </tr>
    <tr>
      <th>Sat</th>
      <th>170</th>
      <td>50.81</td>
      <td>10.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.196812</td>
    </tr>
    <tr>
      <th>Sun</th>
      <th>182</th>
      <td>45.35</td>
      <td>3.50</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.077178</td>
    </tr>
    <tr>
      <th>Thur</th>
      <th>197</th>
      <td>43.11</td>
      <td>5.00</td>
      <td>Yes</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>4</td>
      <td>0.115982</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = tips.groupby('smoker')['tip_pct'].describe()
```


```python
result
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>151.0</td>
      <td>0.159328</td>
      <td>0.039910</td>
      <td>0.056797</td>
      <td>0.136906</td>
      <td>0.155625</td>
      <td>0.185014</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>93.0</td>
      <td>0.163196</td>
      <td>0.085119</td>
      <td>0.035638</td>
      <td>0.106771</td>
      <td>0.153846</td>
      <td>0.195059</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.unstack('smoker')
```




           smoker
    count  No        151.000000
           Yes        93.000000
    mean   No          0.159328
           Yes         0.163196
    std    No          0.039910
           Yes         0.085119
    min    No          0.056797
           Yes         0.035638
    25%    No          0.136906
           Yes         0.106771
    50%    No          0.155625
           Yes         0.153846
    75%    No          0.185014
           Yes         0.195059
    max    No          0.291990
           Yes         0.710345
    dtype: float64




```python
f = lambda x: x.describe()
grouped.apply(f)
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
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="16" valign="top">Fri</th>
      <th rowspan="8" valign="top">No</th>
      <th>count</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>18.420000</td>
      <td>2.812500</td>
      <td>2.250000</td>
      <td>0.151650</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.059282</td>
      <td>0.898494</td>
      <td>0.500000</td>
      <td>0.028123</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.460000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>0.120385</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.100000</td>
      <td>2.625000</td>
      <td>2.000000</td>
      <td>0.137239</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19.235000</td>
      <td>3.125000</td>
      <td>2.000000</td>
      <td>0.149241</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.555000</td>
      <td>3.312500</td>
      <td>2.250000</td>
      <td>0.163652</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22.750000</td>
      <td>3.500000</td>
      <td>3.000000</td>
      <td>0.187735</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Yes</th>
      <th>count</th>
      <td>15.000000</td>
      <td>15.000000</td>
      <td>15.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.813333</td>
      <td>2.714000</td>
      <td>2.066667</td>
      <td>0.174783</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.086388</td>
      <td>1.077668</td>
      <td>0.593617</td>
      <td>0.051293</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.103555</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.690000</td>
      <td>1.960000</td>
      <td>2.000000</td>
      <td>0.133739</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.420000</td>
      <td>2.500000</td>
      <td>2.000000</td>
      <td>0.173913</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.665000</td>
      <td>3.240000</td>
      <td>2.000000</td>
      <td>0.209240</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40.170000</td>
      <td>4.730000</td>
      <td>4.000000</td>
      <td>0.263480</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Sat</th>
      <th rowspan="8" valign="top">No</th>
      <th>count</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19.661778</td>
      <td>3.102889</td>
      <td>2.555556</td>
      <td>0.158048</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.939181</td>
      <td>1.642088</td>
      <td>0.784960</td>
      <td>0.039767</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.250000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.056797</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>14.730000</td>
      <td>2.010000</td>
      <td>2.000000</td>
      <td>0.136240</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17.820000</td>
      <td>2.750000</td>
      <td>2.000000</td>
      <td>0.150152</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20.650000</td>
      <td>3.390000</td>
      <td>3.000000</td>
      <td>0.183915</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48.330000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Yes</th>
      <th>count</th>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.276667</td>
      <td>2.875476</td>
      <td>2.476190</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.069138</td>
      <td>1.630580</td>
      <td>0.862161</td>
      <td>0.061375</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.070000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.035638</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.405000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.091797</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20.390000</td>
      <td>2.690000</td>
      <td>2.000000</td>
      <td>0.153624</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Sun</th>
      <th rowspan="6" valign="top">No</th>
      <th>std</th>
      <td>8.130189</td>
      <td>1.224785</td>
      <td>1.032674</td>
      <td>0.042347</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.770000</td>
      <td>1.010000</td>
      <td>2.000000</td>
      <td>0.059447</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>14.780000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.139780</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18.430000</td>
      <td>3.020000</td>
      <td>3.000000</td>
      <td>0.161665</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.000000</td>
      <td>3.920000</td>
      <td>4.000000</td>
      <td>0.185185</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48.170000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Yes</th>
      <th>count</th>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.120000</td>
      <td>3.516842</td>
      <td>2.578947</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.442511</td>
      <td>1.261151</td>
      <td>0.901591</td>
      <td>0.154134</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.250000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>0.065660</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.165000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.097723</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.100000</td>
      <td>3.500000</td>
      <td>2.000000</td>
      <td>0.138122</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.375000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>0.215325</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45.350000</td>
      <td>6.500000</td>
      <td>5.000000</td>
      <td>0.710345</td>
    </tr>
    <tr>
      <th rowspan="16" valign="top">Thur</th>
      <th rowspan="8" valign="top">No</th>
      <th>count</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.113111</td>
      <td>2.673778</td>
      <td>2.488889</td>
      <td>0.160298</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.721728</td>
      <td>1.282964</td>
      <td>1.179796</td>
      <td>0.038774</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.510000</td>
      <td>1.250000</td>
      <td>1.000000</td>
      <td>0.072961</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.690000</td>
      <td>1.800000</td>
      <td>2.000000</td>
      <td>0.137741</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15.950000</td>
      <td>2.180000</td>
      <td>2.000000</td>
      <td>0.153492</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20.270000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.184843</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41.190000</td>
      <td>6.700000</td>
      <td>6.000000</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Yes</th>
      <th>count</th>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19.190588</td>
      <td>3.030000</td>
      <td>2.352941</td>
      <td>0.163863</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.355149</td>
      <td>1.113491</td>
      <td>0.701888</td>
      <td>0.039389</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.340000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.090014</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.510000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.148038</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.470000</td>
      <td>2.560000</td>
      <td>2.000000</td>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>19.810000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>0.194837</td>
    </tr>
    <tr>
      <th>max</th>
      <td>43.110000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>0.241255</td>
    </tr>
  </tbody>
</table>
<p>64 rows × 4 columns</p>
</div>




```python
# 禁止分组键
```


```python
tips.groupby('smoker', group_keys=False).apply(top)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>88</th>
      <td>24.71</td>
      <td>5.85</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.236746</td>
    </tr>
    <tr>
      <th>185</th>
      <td>20.69</td>
      <td>5.00</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>5</td>
      <td>0.241663</td>
    </tr>
    <tr>
      <th>51</th>
      <td>10.29</td>
      <td>2.60</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th>149</th>
      <td>7.51</td>
      <td>2.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>109</th>
      <td>14.31</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.279525</td>
    </tr>
    <tr>
      <th>183</th>
      <td>23.17</td>
      <td>6.50</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.280535</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 分位数和桶分析
```


```python
frame = pd.DataFrame({'data1': np.random.randn(1000), 'data2': np.random.randn(1000)})
```


```python
quartiles = pd.cut(frame.data1, 4)
```


```python
quartiles[:10]
```




    0     (0.0694, 1.717]
    1    (-1.578, 0.0694]
    2     (0.0694, 1.717]
    3     (0.0694, 1.717]
    4     (0.0694, 1.717]
    5     (0.0694, 1.717]
    6    (-1.578, 0.0694]
    7    (-1.578, 0.0694]
    8     (0.0694, 1.717]
    9     (0.0694, 1.717]
    Name: data1, dtype: category
    Categories (4, interval[float64]): [(-3.232, -1.578] < (-1.578, 0.0694] < (0.0694, 1.717] < (1.717, 3.364]]




```python
def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
```


```python
grouped = frame.data2.groupby(quartiles)
```


```python
grouped.apply(get_stats).unstack()
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
      <th>count</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
    </tr>
    <tr>
      <th>data1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(-3.232, -1.578]</th>
      <td>70.0</td>
      <td>2.199183</td>
      <td>-0.063975</td>
      <td>-2.438700</td>
    </tr>
    <tr>
      <th>(-1.578, 0.0694]</th>
      <td>435.0</td>
      <td>2.502537</td>
      <td>-0.002962</td>
      <td>-2.330817</td>
    </tr>
    <tr>
      <th>(0.0694, 1.717]</th>
      <td>454.0</td>
      <td>2.802182</td>
      <td>0.010430</td>
      <td>-3.389642</td>
    </tr>
    <tr>
      <th>(1.717, 3.364]</th>
      <td>41.0</td>
      <td>1.432583</td>
      <td>0.056975</td>
      <td>-2.038616</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 根据样本分位数得到大小相等的桶
# labels=False 只获取分位数的编号
grouping = pd.qcut(frame.data1, 10, labels=False)
```


```python
grouped = frame.data2.groupby(grouping)
```


```python
grouped.apply(get_stats).unstack()
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
      <th>count</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
    </tr>
    <tr>
      <th>data1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>2.199183</td>
      <td>-0.010641</td>
      <td>-2.438700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100.0</td>
      <td>2.451880</td>
      <td>0.028010</td>
      <td>-2.304453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.0</td>
      <td>1.923003</td>
      <td>0.013248</td>
      <td>-2.279909</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.0</td>
      <td>2.108032</td>
      <td>-0.102051</td>
      <td>-2.289198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>2.502537</td>
      <td>0.005952</td>
      <td>-2.330817</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>2.146883</td>
      <td>-0.016103</td>
      <td>-2.454888</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100.0</td>
      <td>2.802182</td>
      <td>0.045884</td>
      <td>-2.426720</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100.0</td>
      <td>2.597919</td>
      <td>-0.018496</td>
      <td>-2.340837</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100.0</td>
      <td>2.753333</td>
      <td>0.021250</td>
      <td>-2.539414</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100.0</td>
      <td>1.839721</td>
      <td>0.045989</td>
      <td>-3.389642</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 示例：用特定于分组的值填充缺失值
```


```python
s = pd.Series(np.random.randn(6))
```


```python
s[::2] = np.nan
```


```python
s
```




    0         NaN
    1    0.602677
    2         NaN
    3   -1.099396
    4         NaN
    5   -0.044948
    dtype: float64




```python
s.fillna(s.mean())
```




    0   -0.180556
    1    0.602677
    2   -0.180556
    3   -1.099396
    4   -0.180556
    5   -0.044948
    dtype: float64




```python
states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
```


```python
group_key = ['East'] * 4 + ['West'] * 4
```


```python
data = pd.Series(np.random.randn(8), index=states)
```


```python
data
```




    Ohio          0.439139
    New York     -0.562333
    Vermont      -0.785135
    Florida       1.381403
    Oregon        0.226499
    Nevada       -0.483901
    California    0.813845
    Idaho         0.580253
    dtype: float64




```python
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
```


```python
data
```




    Ohio          0.439139
    New York     -0.562333
    Vermont            NaN
    Florida       1.381403
    Oregon        0.226499
    Nevada             NaN
    California    0.813845
    Idaho              NaN
    dtype: float64




```python
data.groupby(group_key).mean()
```




    East    0.419403
    West    0.520172
    dtype: float64




```python
fill_mean = lambda g: g.fillna(g.mean())
```


```python
data.groupby(group_key).apply(fill_mean)
```




    Ohio          0.439139
    New York     -0.562333
    Vermont       0.419403
    Florida       1.381403
    Oregon        0.226499
    Nevada        0.520172
    California    0.813845
    Idaho         0.520172
    dtype: float64




```python
fill_values = {'East': 0.5, 'West': -1}
```


```python
fill_func = lambda g: g.fillna(fill_values[g.name])
```


```python
data.groupby(group_key).apply(fill_func)
```




    Ohio          0.439139
    New York     -0.562333
    Vermont       0.500000
    Florida       1.381403
    Oregon        0.226499
    Nevada       -1.000000
    California    0.813845
    Idaho        -1.000000
    dtype: float64




```python
# 示例：随机采样和排列
# 假设你想要从一个大数据集中随机抽取（进行替换或不替换）样本以进行蒙特卡罗模拟（Monte Carlo simulation）或其他分析工作
```


```python
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)
```


```python
deck[:13]
```




    AH      1
    2H      2
    3H      3
    4H      4
    5H      5
    6H      6
    7H      7
    8H      8
    9H      9
    10H    10
    JH     10
    KH     10
    QH     10
    dtype: int64




```python
def draw(deck, n=5):
    return deck.sample(n)
```


```python
get_suit = lambda card: card[-1]  # last letter is suit
```


```python
deck.groupby(get_suit).apply(draw, n=2)
```




    C  JC     10
       6C      6
    D  10D    10
       6D      6
    H  AH      1
       6H      6
    S  AS      1
       7S      7
    dtype: int64




```python
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)
```




    10C    10
    7C      7
    KD     10
    2D      2
    JH     10
    3H      3
    JS     10
    8S      8
    dtype: int64




```python
# 示例：分组加权平均数和相关系数
```


```python
df = pd.DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                   'data': np.random.randn(8),
                   'weights': np.random.rand(8)})
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
      <th>category</th>
      <th>data</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.130173</td>
      <td>0.292165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>-0.346761</td>
      <td>0.269762</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>-0.663549</td>
      <td>0.752251</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>-0.052279</td>
      <td>0.330457</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>2.273957</td>
      <td>0.298347</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>-0.156782</td>
      <td>0.694747</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>0.899369</td>
      <td>0.496069</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>0.270720</td>
      <td>0.817164</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = df.groupby('category')
```


```python
 get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
```


```python
grouped.apply(get_wavg)
```




    category
    a   -0.347762
    b    0.536297
    dtype: float64




```python
close_px = pd.read_csv('examples/stock_px_2.csv', parse_dates=True, index_col=0)
```


```python
close_px.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2214 entries, 2003-01-02 to 2011-10-14
    Data columns (total 4 columns):
    AAPL    2214 non-null float64
    MSFT    2214 non-null float64
    XOM     2214 non-null float64
    SPX     2214 non-null float64
    dtypes: float64(4)
    memory usage: 86.5 KB
    


```python
close_px[-4:]
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
      <th>AAPL</th>
      <th>MSFT</th>
      <th>XOM</th>
      <th>SPX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-10-11</th>
      <td>400.29</td>
      <td>27.00</td>
      <td>76.27</td>
      <td>1195.54</td>
    </tr>
    <tr>
      <th>2011-10-12</th>
      <td>402.19</td>
      <td>26.96</td>
      <td>77.16</td>
      <td>1207.25</td>
    </tr>
    <tr>
      <th>2011-10-13</th>
      <td>408.43</td>
      <td>27.18</td>
      <td>76.37</td>
      <td>1203.66</td>
    </tr>
    <tr>
      <th>2011-10-14</th>
      <td>422.00</td>
      <td>27.27</td>
      <td>78.11</td>
      <td>1224.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 计算一个由日收益率（通过百分数变化计算）与SPX之间的年度相关系数组成的DataFrame
spx_corr = lambda x: x.corrwith(x['SPX'])
```


```python
rets = close_px.pct_change().dropna()
```


```python
get_year = lambda x: x.year
```


```python
by_year = rets.groupby(get_year)
```


```python
by_year.apply(spx_corr)
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
      <th>AAPL</th>
      <th>MSFT</th>
      <th>XOM</th>
      <th>SPX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003</th>
      <td>0.541124</td>
      <td>0.745174</td>
      <td>0.661265</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>0.374283</td>
      <td>0.588531</td>
      <td>0.557742</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>0.467540</td>
      <td>0.562374</td>
      <td>0.631010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>0.428267</td>
      <td>0.406126</td>
      <td>0.518514</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>0.508118</td>
      <td>0.658770</td>
      <td>0.786264</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>0.681434</td>
      <td>0.804626</td>
      <td>0.828303</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>0.707103</td>
      <td>0.654902</td>
      <td>0.797921</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>0.710105</td>
      <td>0.730118</td>
      <td>0.839057</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>0.691931</td>
      <td>0.800996</td>
      <td>0.859975</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))
```




    2003    0.480868
    2004    0.259024
    2005    0.300093
    2006    0.161735
    2007    0.417738
    2008    0.611901
    2009    0.432738
    2010    0.571946
    2011    0.581987
    dtype: float64




```python
# 示例：组级别的线性回归
```


```python
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params
```


```python
by_year.apply(regress, 'AAPL', ['SPX'])
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
      <th>SPX</th>
      <th>intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003</th>
      <td>1.195406</td>
      <td>0.000710</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>1.363463</td>
      <td>0.004201</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1.766415</td>
      <td>0.003246</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>1.645496</td>
      <td>0.000080</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.198761</td>
      <td>0.003438</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>0.968016</td>
      <td>-0.001110</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>0.879103</td>
      <td>0.002954</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1.052608</td>
      <td>0.001261</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>0.806605</td>
      <td>0.001514</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 透视表和交叉表
# 透视表（ pivot table）是各种电子表格程序和其他数据分析软件中一种常见的数据汇总工具。
# 它根据一个或多个键对数据进行聚合，并根据行和列上的分组键将数据分配到各个矩形区域中。
# 在 Python和 pandas中，可以通过本章所介绍的 groupby功能以及（能够利用层次化索引的）重塑
# 运算制作透视表。DataFrame有一个pivot_table方法，
# 此外还有一个顶级的 pandas.pivot_table函数。
# 除能为groupby提供便利之外，pivot_table还可以添加分项小计，也叫做 margins。
```


```python
tips.pivot_table(index=['day', 'smoker'])
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
      <th>size</th>
      <th>tip</th>
      <th>tip_pct</th>
      <th>total_bill</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>2.250000</td>
      <td>2.812500</td>
      <td>0.151650</td>
      <td>18.420000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.066667</td>
      <td>2.714000</td>
      <td>0.174783</td>
      <td>16.813333</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>2.555556</td>
      <td>3.102889</td>
      <td>0.158048</td>
      <td>19.661778</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.476190</td>
      <td>2.875476</td>
      <td>0.147906</td>
      <td>21.276667</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>2.929825</td>
      <td>3.167895</td>
      <td>0.160113</td>
      <td>20.506667</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.578947</td>
      <td>3.516842</td>
      <td>0.187250</td>
      <td>24.120000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>2.488889</td>
      <td>2.673778</td>
      <td>0.160298</td>
      <td>17.113111</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.352941</td>
      <td>3.030000</td>
      <td>0.163863</td>
      <td>19.190588</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'], columns='smoker')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">size</th>
      <th colspan="2" halign="left">tip_pct</th>
    </tr>
    <tr>
      <th></th>
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>2.000000</td>
      <td>2.222222</td>
      <td>0.139622</td>
      <td>0.165347</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>2.555556</td>
      <td>2.476190</td>
      <td>0.158048</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>2.929825</td>
      <td>2.578947</td>
      <td>0.160113</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>0.159744</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>3.000000</td>
      <td>1.833333</td>
      <td>0.187735</td>
      <td>0.188937</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.500000</td>
      <td>2.352941</td>
      <td>0.160311</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'], columns='smoker', margins=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">tip_pct</th>
    </tr>
    <tr>
      <th></th>
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>2.000000</td>
      <td>2.222222</td>
      <td>2.166667</td>
      <td>0.139622</td>
      <td>0.165347</td>
      <td>0.158916</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>2.555556</td>
      <td>2.476190</td>
      <td>2.517241</td>
      <td>0.158048</td>
      <td>0.147906</td>
      <td>0.153152</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>2.929825</td>
      <td>2.578947</td>
      <td>2.842105</td>
      <td>0.160113</td>
      <td>0.187250</td>
      <td>0.166897</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>0.159744</td>
      <td>NaN</td>
      <td>0.159744</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>3.000000</td>
      <td>1.833333</td>
      <td>2.000000</td>
      <td>0.187735</td>
      <td>0.188937</td>
      <td>0.188765</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.500000</td>
      <td>2.352941</td>
      <td>2.459016</td>
      <td>0.160311</td>
      <td>0.163863</td>
      <td>0.161301</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>2.668874</td>
      <td>2.408602</td>
      <td>2.569672</td>
      <td>0.159328</td>
      <td>0.163196</td>
      <td>0.160803</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.pivot_table('tip_pct', index=['time', 'smoker'], columns='day', aggfunc=len, margins=True)
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
      <th>day</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
      <th>Thur</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Dinner</th>
      <th>No</th>
      <td>3.0</td>
      <td>45.0</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>9.0</td>
      <td>42.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>No</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>19.0</td>
      <td>87.0</td>
      <td>76.0</td>
      <td>62.0</td>
      <td>244.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.pivot_table('tip_pct', index=['time', 'size', 'smoker'], columns='day', aggfunc='mean', fill_value=0)
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
      <th>day</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
      <th>Thur</th>
    </tr>
    <tr>
      <th>time</th>
      <th>size</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">Dinner</th>
      <th rowspan="2" valign="top">1</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.137931</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.000000</td>
      <td>0.325733</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>No</th>
      <td>0.139622</td>
      <td>0.162705</td>
      <td>0.168859</td>
      <td>0.159744</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.171297</td>
      <td>0.148668</td>
      <td>0.207893</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.154661</td>
      <td>0.152663</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.000000</td>
      <td>0.144995</td>
      <td>0.152660</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.150096</td>
      <td>0.148143</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.117750</td>
      <td>0.124515</td>
      <td>0.193370</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">5</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.206928</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.000000</td>
      <td>0.106572</td>
      <td>0.065660</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.103799</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">Lunch</th>
      <th rowspan="2" valign="top">1</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181728</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.223776</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166005</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.181969</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.158843</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>No</th>
      <td>0.187735</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.084246</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.204952</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.138919</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.155410</td>
    </tr>
    <tr>
      <th>5</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.121389</td>
    </tr>
    <tr>
      <th>6</th>
      <th>No</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.173706</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 交叉表：crosstab
```


```python
data
```




    Ohio          0.439139
    New York     -0.562333
    Vermont            NaN
    Florida       1.381403
    Oregon        0.226499
    Nevada             NaN
    California    0.813845
    Idaho              NaN
    dtype: float64




```python
data = pd.DataFrame({'Sample': [i + 1 for i in range(10)],
             'Nationality': ['USA', 'Japan', 'USA', 'Japan', 'Japan', 
                             'Japan', 'USA', 'USA', 'Japan', 'USA'],
             'Handedness': ['Right-handed', 'Left-handed', 'Right-handed', 
                            'Right-handed', 'Left-handed', 'Right-handed', 
                            'Right-handed', 'Left-handed', 'Right-handed', 
                            'Right-handed']})
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
      <th>Sample</th>
      <th>Nationality</th>
      <th>Handedness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Japan</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Japan</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>USA</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(data.Nationality, data.Handedness, margins=True)
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
      <th>Handedness</th>
      <th>Left-handed</th>
      <th>Right-handed</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Nationality</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Japan</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>All</th>
      <td>3</td>
      <td>7</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)
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
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>3</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>45</td>
      <td>42</td>
      <td>87</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>57</td>
      <td>19</td>
      <td>76</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>1</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>44</td>
      <td>17</td>
      <td>61</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>151</td>
      <td>93</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>


