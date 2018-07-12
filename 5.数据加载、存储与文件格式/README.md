

```python
import pandas as pd
```


```python
import sys
```


```python
import numpy as np
```


```python
import csv
```


```python
import json
```


```python
from lxml import objectify
```


```python
from io import StringIO
```


```python
import requests
```


```python
import sqlite3
```


```python
import sqlalchemy as sqla
```


```python
# 输入输出通常可以划分为几个大类：
# 读取文本文件和其他更高效的磁盘存储格式，
# 加载数据库中的数据，利用Web API操作网络资源。
```


```python
# 读写文本格式数据
```


```python
# 将表格型数据读取为DataFrame对象的函数
# read_csv  从文件, URL, 文件型对象中加载带分隔符的数据. 默认分隔符逗号
# read_table  从文件, URL, 文件型对象中加载带分隔符的数据. 默认分隔符制表符('\t')
# read_fwf  读取定宽格式数据(没有分隔符)
# read_clipboard  读取剪贴板中的数据
# read_excel  从Excel XLS或XLSX file读取表格数据
# read_hdf  读取pandas写的HDF5文件
# read_html  读取HTML文档中的所有表格
# read_json  读取JSON(JavaScript Object Notation)字符串中的数据
# read_msgpack  二进制编码的pandas数据
# read_pickle  读取Python pickle格式中存储的任意对象
# read_sas  读取存储于SAS系统自定义存储格式的SAS数据集
# read_sql  (使用SQLAlchemy) 读取SQL查询结果为pandas的DataFrame
# read_stata  读取Stata文件格式的数据集
# read_feather 读取Feather二进制文件格式
# 函数的选项可以划分为以下几个大类：
# 索引：将一个或多个列当做返回的DataFrame处理，以及是否从文件、用户获取列名。
# 类型推断和数据转换：包括用户定义值的转换、和自定义的缺失值标记列表等。
# 日期解析：包括组合功能，比如将分散在多个列中的日期时间信息组合成结果中的单个列。
# 迭代：支持对大文件进行逐块迭代。
# 不规整数据问题：跳过一些行、页脚、注释或其他一些不重要的东西（比如由成千上万个逗号隔开的数值数据）。
```


```python
!type examples\ex1.csv
```

    a,b,c,d,message
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
df = pd.read_csv('examples/ex1.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_table('examples/ex1.csv', sep=',')
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\ex2.csv
```

    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
pd.read_csv('examples/ex2.csv', header=None)  # 没有标题行
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
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv('examples/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
names=['a', 'b', 'c', 'd', 'message']
```


```python
pd.read_csv('examples/ex2.csv', names=names, index_col='message')
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
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\csv_mindex.csv
```

    key1,key2,value1,value2
    one,a,1,2
    one,b,3,4
    one,c,5,6
    one,d,7,8
    two,a,9,10
    two,b,11,12
    two,c,13,14
    two,d,15,16
    


```python
parsed = pd.read_csv('examples/csv_mindex.csv', index_col=['key1', 'key2'])  # 将多个列做成一个层次化索引
```


```python
parsed
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
      <th>value1</th>
      <th>value2</th>
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
      <th rowspan="4" valign="top">one</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>a</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>c</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <td>15</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(open('examples/ex3.txt'))
```




    ['            A         B         C\n',
     'aaa -0.264438 -1.026059 -0.619500\n',
     'bbb  0.927272  0.302904 -0.032399\n',
     'ccc -0.264273 -0.386314 -0.217601\n',
     'ddd -0.871858 -0.348382  1.100491\n']




```python
# 传递一个正则表达式作为read_table的分隔符。可以用正则表达式表达为\s+
result = pd.read_table('examples/ex3.txt', sep='\s+')
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aaa</th>
      <td>-0.264438</td>
      <td>-1.026059</td>
      <td>-0.619500</td>
    </tr>
    <tr>
      <th>bbb</th>
      <td>0.927272</td>
      <td>0.302904</td>
      <td>-0.032399</td>
    </tr>
    <tr>
      <th>ccc</th>
      <td>-0.264273</td>
      <td>-0.386314</td>
      <td>-0.217601</td>
    </tr>
    <tr>
      <th>ddd</th>
      <td>-0.871858</td>
      <td>-0.348382</td>
      <td>1.100491</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\ex4.csv
```

    # hey!
    a,b,c,d,message
    # just wanted to make things more difficult for you
    # who reads CSV files with computers, anyway?
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
pd.read_csv('examples/ex4.csv', skiprows=[0, 2, 3])  # 用skiprows跳过文件的第一行、第三行和第四行
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.read_csv('examples/ex5.csv')
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.isnull(result)
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])  # na_values可以用一个列表或集合的字符串表示缺失值
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
```


```python
pd.read_csv('examples/ex5.csv', na_values=sentinels)  # 字典的各列可以使用不同的NA标记值
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# path  文件系统位置
# sep或delimiter
# header
# index_col
# names
# skiprows
# na_values
# comment
# parse_dates
# keep_date_col
# converters
# dayfirst
# date_parser
# nrows
# iterator
# chunksize
# skip_footer
# verbose
# encoding
# squeeze
# thousands
```


```python
# 逐块读取文本文件
```


```python
pd.options.display.max_rows = 10  # 设置pandas显示地更紧
```


```python
result = pd.read_csv('examples/ex6.csv')
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>2.311896</td>
      <td>-0.417070</td>
      <td>-1.409599</td>
      <td>-0.515821</td>
      <td>L</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>-0.479893</td>
      <td>-0.650419</td>
      <td>0.745152</td>
      <td>-0.646038</td>
      <td>E</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.523331</td>
      <td>0.787112</td>
      <td>0.486066</td>
      <td>1.093156</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>-0.362559</td>
      <td>0.598894</td>
      <td>-1.843201</td>
      <td>0.887292</td>
      <td>G</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>-0.096376</td>
      <td>-1.012999</td>
      <td>-0.657431</td>
      <td>-0.573315</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
pd.read_csv('examples/ex6.csv', nrows=5)  # 只想读取几行（避免读取整个文件），通过nrows进行指定
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
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
chunker= pd.read_csv('examples/ex6.csv', chunksize=1000)  # 逐块读取文件，可以指定chunksize（行数）
# 迭代处理ex6.csv，将值计数聚合到"key"列中
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
```


```python
chunker
```




    <pandas.io.parsers.TextFileReader at 0x213a65eb4a8>




```python
tot[:10]
```




    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
    M    338.0
    J    337.0
    F    335.0
    K    334.0
    H    330.0
    dtype: float64




```python
# 将数据写到文本格式
data = pd.read_csv('examples/ex5.csv')
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.to_csv('examples/out.csv')
```


```python
!type examples\out.csv
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,
    1,two,5,6,,8,world
    2,three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, sep='|')  # 以'|' 写入
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo
    


```python
data.to_csv(sys.stdout, na_rep='$$$')  # 缺失值在输出结果中表示为别的标记值
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,$$$
    1,two,5,6,$$$,8,world
    2,three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, header=False)
```

    one,1,2,3.0,4,
    two,5,6,,8,world
    three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0
    


```python
dates = pd.date_range('1/1/2000', periods=7)
```


```python
ts = pd.Series(np.arange(7), index=dates)
```


```python
ts.to_csv('examples/tseries.csv')
```


```python
!type examples\tseries.csv
```

    2000-01-01,0
    2000-01-02,1
    2000-01-03,2
    2000-01-04,3
    2000-01-05,4
    2000-01-06,5
    2000-01-07,6
    


```python
# 处理分隔符格式
```


```python
!type examples\ex7.csv
```

    "a","b","c"
    "1","2","3"
    "1","2","3"
    


```python
f = open('examples/ex7.csv')
```


```python
# 对于任何单字符分隔符文件，可以直接使用Python内置的csv模块。
# 将任意已打开的文件或文件型的对象传给csv.reader
reader = csv.reader(f)
```


```python
for line in reader:
    print(line)
```

    ['a', 'b', 'c']
    ['1', '2', '3']
    ['1', '2', '3']
    


```python
with open('examples/ex7.csv') as f:
    lines = list(csv.reader(f))
```


```python
header, values = lines[0], lines[1:]
```


```python
data_dict = {h: v for h, v in zip(header, zip(*values))}
```


```python
data_dict
```




    {'a': ('1', '1'), 'b': ('2', '2'), 'c': ('3', '3')}




```python
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL

reader = csv.reader(f, dialect=my_dialect)
```


```python
reader = csv.reader(f, delimiter='|')
```


```python
# 手工输出分隔符文件，可以使用csv.writer。
# 接受一个已打开且可写的文件对象以及跟csv.reader相同的那些语支和格式化选项
```


```python
with open('my_data.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
```


```python
# JSON数据
```


```python
obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38, "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
```


```python
result = json.loads(obj)
```


```python
result
```




    {'name': 'Wes',
     'places_lived': ['United States', 'Spain', 'Germany'],
     'pet': None,
     'siblings': [{'name': 'Scott', 'age': 30, 'pets': ['Zeus', 'Zuko']},
      {'name': 'Katie', 'age': 38, 'pets': ['Sixes', 'Stache', 'Cisco']}]}




```python
asjson = json.dumps(result)
```


```python
# 向DataFrame构造器传入一个字典的列表（就是原先的JSON对象），并选取数据字段的子集
siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
```


```python
siblings
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scott</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Katie</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\example.json
```

    [{"a": 1, "b": 2, "c": 3},
     {"a": 4, "b": 5, "c": 6},
     {"a": 7, "b": 8, "c": 9}]
    


```python
# pandas.read_json的默认选项假设JSON数组中的每个对象是表格中的一行
data = pd.read_json('examples/example.json')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将数据从pandas输出到JSON
print(data.to_json())
```

    {"a":{"0":1,"1":4,"2":7},"b":{"0":2,"1":5,"2":8},"c":{"0":3,"1":6,"2":9}}
    


```python
print(data.to_json(orient='records'))
```

    [{"a":1,"b":2,"c":3},{"a":4,"b":5,"c":6},{"a":7,"b":8,"c":9}]
    


```python
# XML和HTML：Web信息收集
```


```python
tables = pd.read_html('examples/fdic_failed_bank_list.html')
```


```python
len(tables)
```




    1




```python
failures = tables[0]
```


```python
failures.head()
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
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
      <th>Updated Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allied Bank</td>
      <td>Mulberry</td>
      <td>AR</td>
      <td>91</td>
      <td>Today's Bank</td>
      <td>September 23, 2016</td>
      <td>November 17, 2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Woodbury Banking Company</td>
      <td>Woodbury</td>
      <td>GA</td>
      <td>11297</td>
      <td>United Bank</td>
      <td>August 19, 2016</td>
      <td>November 17, 2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First CornerStone Bank</td>
      <td>King of Prussia</td>
      <td>PA</td>
      <td>35312</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 6, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trust Company Bank</td>
      <td>Memphis</td>
      <td>TN</td>
      <td>9956</td>
      <td>The Bank of Fayette County</td>
      <td>April 29, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North Milwaukee State Bank</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>20364</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>March 11, 2016</td>
      <td>June 16, 2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 做数据清洗和分析，计算按年份计算倒闭的银行数
```


```python
close_timestamps = pd.to_datetime(failures['Closing Date'])
```


```python
close_timestamps.dt.year.value_counts()
```




    2010    157
    2009    140
    2011     92
    2012     51
    2008     25
           ... 
    2004      4
    2001      4
    2007      3
    2003      3
    2000      2
    Name: Closing Date, Length: 15, dtype: int64




```python
# 利用lxml.objectify解析XML
```


```python
path = 'examples/mta_perf/Performance_MNR.xml'
```


```python
parsed = objectify.parse(open(path))
```


```python
root = parsed.getroot()
```


```python
data = []

skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
               'DESIRED_CHANGE', 'DECIMAL_PLACES']
# root.INDICATOR返回一个用于产生各个<INDICATOR>XML元素的生成器
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)
```


```python
perf = pd.DataFrame(data)  # 将字典转化为DataFrame
```


```python
perf.head()
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
      <th>AGENCY_NAME</th>
      <th>CATEGORY</th>
      <th>DESCRIPTION</th>
      <th>FREQUENCY</th>
      <th>INDICATOR_NAME</th>
      <th>INDICATOR_UNIT</th>
      <th>MONTHLY_ACTUAL</th>
      <th>MONTHLY_TARGET</th>
      <th>PERIOD_MONTH</th>
      <th>PERIOD_YEAR</th>
      <th>YTD_ACTUAL</th>
      <th>YTD_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Metro-North Railroad</td>
      <td>Service Indicators</td>
      <td>Percent of commuter trains that arrive at thei...</td>
      <td>M</td>
      <td>On-Time Performance (West of Hudson)</td>
      <td>%</td>
      <td>96.9</td>
      <td>95</td>
      <td>1</td>
      <td>2008</td>
      <td>96.9</td>
      <td>95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Metro-North Railroad</td>
      <td>Service Indicators</td>
      <td>Percent of commuter trains that arrive at thei...</td>
      <td>M</td>
      <td>On-Time Performance (West of Hudson)</td>
      <td>%</td>
      <td>95</td>
      <td>95</td>
      <td>2</td>
      <td>2008</td>
      <td>96</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Metro-North Railroad</td>
      <td>Service Indicators</td>
      <td>Percent of commuter trains that arrive at thei...</td>
      <td>M</td>
      <td>On-Time Performance (West of Hudson)</td>
      <td>%</td>
      <td>96.9</td>
      <td>95</td>
      <td>3</td>
      <td>2008</td>
      <td>96.3</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Metro-North Railroad</td>
      <td>Service Indicators</td>
      <td>Percent of commuter trains that arrive at thei...</td>
      <td>M</td>
      <td>On-Time Performance (West of Hudson)</td>
      <td>%</td>
      <td>98.3</td>
      <td>95</td>
      <td>4</td>
      <td>2008</td>
      <td>96.8</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Metro-North Railroad</td>
      <td>Service Indicators</td>
      <td>Percent of commuter trains that arrive at thei...</td>
      <td>M</td>
      <td>On-Time Performance (West of Hudson)</td>
      <td>%</td>
      <td>95.8</td>
      <td>95</td>
      <td>5</td>
      <td>2008</td>
      <td>96.6</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
tag = '<a href="http://www.google.com">Google</a>'
```


```python
root = objectify.parse(StringIO(tag)).getroot()
```


```python
root
```




    <Element a at 0x213a6597c88>




```python
root.get('href')
```




    'http://www.google.com'




```python
root.text
```




    'Google'




```python
# 二进制数据格式
```


```python
frame = pd.read_csv('examples/ex1.csv')
```


```python
frame
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.to_pickle('examples/frame_pickle')  # 将数据以pickle格式保存到磁盘上
```


```python
pd.read_pickle('examples/frame_pickle')  # 读取pickle化的数据
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame = pd.DataFrame({'a': np.random.randn(100)})
```


```python
store = pd.HDFStore('mydata.h5')
```


```python
store['obj1'] = frame
```


```python
store['obj1_col']  =  frame['a']
```


```python
store
```




    <class 'pandas.io.pytables.HDFStore'>
    File path: mydata.h5




```python
store['obj1']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.052403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.334901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.739852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.691742</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.363188</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.126409</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.878570</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.288966</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-0.032129</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-1.118655</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>




```python
store.put('obj2', frame, format='table')
```


```python
store.select('obj2', where=['index >= 10 and index <= 15'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.872572</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.042847</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.501108</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.805839</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.669630</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.039934</td>
    </tr>
  </tbody>
</table>
</div>




```python
store.close()
```


```python
frame.to_hdf('mydata.h5', 'obj3', format='table')
```


```python
pd.read_hdf('mydata.h5', 'obj3', where=['index < 5'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.052403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.334901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.739852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.691742</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.363188</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 读取Microsoft Excel文件
```


```python
xlsx = pd.ExcelFile('examples/ex1.xlsx')
```


```python
pd.read_excel(xlsx, 'Sheet1')
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame = pd.read_excel('examples/ex1.xlsx', 'Sheet1')
```


```python
frame
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
writer = pd.ExcelWriter('examples/ex2.xlsx')  # 将pandas数据写入为Excel格式
```


```python
frame.to_excel(writer, 'Sheet1')
```


```python
writer.save()
```


```python
frame.to_excel('examples/ex2.xlsx')
```


```python
# Web APIs交互
```


```python
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
```


```python
resp = requests.get(url)
```


```python
resp
```




    <Response [200]>




```python
data = resp.json()
```


```python
data[0]['title']
```




    'pd.to_timedelta not parsing iso-formatted strings'




```python
issues = pd.DataFrame(data, columns=['number', 'title', 'labels', 'state'])
```


```python
issues
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
      <th>number</th>
      <th>title</th>
      <th>labels</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21877</td>
      <td>pd.to_timedelta not parsing iso-formatted strings</td>
      <td>[]</td>
      <td>open</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21874</td>
      <td>BUG: Align Series.str.zfill() with str.zfill()</td>
      <td>[{'id': 76811, 'node_id': 'MDU6TGFiZWw3NjgxMQ=...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21873</td>
      <td>TST: Parameterize more tests</td>
      <td>[{'id': 211029535, 'node_id': 'MDU6TGFiZWwyMTE...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21872</td>
      <td>[REF] Move comparison methods to EAMixins, sha...</td>
      <td>[{'id': 211029535, 'node_id': 'MDU6TGFiZWwyMTE...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21871</td>
      <td>API: Add DataFrame.droplevel</td>
      <td>[{'id': 35818298, 'node_id': 'MDU6TGFiZWwzNTgx...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>21839</td>
      <td>PERF: better memory footprint for intna</td>
      <td>[{'id': 31404521, 'node_id': 'MDU6TGFiZWwzMTQw...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>26</th>
      <td>21838</td>
      <td>CLN/DEPR: Undocumented, old helper functions</td>
      <td>[{'id': 211029535, 'node_id': 'MDU6TGFiZWwyMTE...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>27</th>
      <td>21835</td>
      <td>Expose ExcelWriter as part of the Generated API</td>
      <td>[{'id': 134699, 'node_id': 'MDU6TGFiZWwxMzQ2OT...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>28</th>
      <td>21831</td>
      <td>SIGSEGV when calling .copy()</td>
      <td>[{'id': 685114413, 'node_id': 'MDU6TGFiZWw2ODU...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>29</th>
      <td>21829</td>
      <td>Increasing minimum cython version silently bre...</td>
      <td>[{'id': 129350, 'node_id': 'MDU6TGFiZWwxMjkzNT...</td>
      <td>open</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 4 columns</p>
</div>




```python
# 数据库交互
```


```python
query = """CREATE TABLE test (a VARCHAR(20), b VARCHAR(20), c REAL, d INTEGER );"""
```


```python
con = sqlite3.connect('mydata.sqlite')
```


```python
con.execute(query)
```




    <sqlite3.Cursor at 0x213a8464260>




```python
con.commit()
```


```python
data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
```


```python
stmt = "INSERT INTO test VALUES(?, ?, ? ,?)"
```


```python
con.executemany(stmt, data)
```




    <sqlite3.Cursor at 0x213a83ba5e0>




```python
cursor = con.execute('SELECT * FROM test')
```


```python
rows = cursor.fetchall()
```


```python
rows
```




    [('Atlanta', 'Georgia', 1.25, 6),
     ('Tallahassee', 'Florida', 2.6, 3),
     ('Sacramento', 'California', 1.7, 5)]




```python
cursor.description
```




    (('a', None, None, None, None, None, None),
     ('b', None, None, None, None, None, None),
     ('c', None, None, None, None, None, None),
     ('d', None, None, None, None, None, None))




```python
pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
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
      <td>Atlanta</td>
      <td>Georgia</td>
      <td>1.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tallahassee</td>
      <td>Florida</td>
      <td>2.60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sacramento</td>
      <td>California</td>
      <td>1.70</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
db = sqla.create_engine('sqlit:///mydata.sqlite')
```


```python
pd.read_sql('select * from test', db)
```
