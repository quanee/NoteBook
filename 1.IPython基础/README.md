

```python
import datetime
```


```python
datetime.datetime
```




    datetime.datetime




```python
b = [1, 2, 3]
```


```python
b?
```


```python
b?
```


```python
print?
```


```python
b?
```


```python
print('hello')
```

    hello
    


```python
def add_numbers(a, b):
    """
    Add two numbers together

    Returns
    -------
    the_sum : type of arguments
    """
    return a + b
```


```python
add_numbers?
```


```python
add_numbers??
```


```python
%run ipython_script_test.py
```


```python
c
```




    7.5




```python
result
```




    1.4666666666666666




```python
# %load ipython_script_test.py


# In[1]:


def f(x, y, z):
    return (x + y) / z

a = 5
b = 6
c = 7.5

result = f(a, b, c)


```


```python
# %paste
```


```python
# %pastebin
```


```python
%debug?
```


```python
import numpy as np
```


```python
a = np.random.randn(100, 100)
```


```python
%timeit np.dot(a, a)
```

    124 µs ± 12.5 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    


```python
%pwd
```




    'E:\\我的坚果云\\Note\\Python\\Jupyter\\利用Python进行数据分析'




```python
%quickref  # 显示IPython的快速参考
```


```python
%magic  # 显示所有魔术命令的详细文档
```


```python
# %debug  # 在出现异常的语句进入调试模式
```


```python
%hist  # 打印命令的输入历史
```


```python
# %pdb  # 出现异常时自动进入调试
```


```python
# %rest  # 删除所有命名空间的变量和名字
```


```python
# %page OBJECT  # 美化打印对象 分页显示
```


```python
%run test.ipynb  # 运行代码
```

    [[[[]]]]
    


```python
%prun print('hello')  # 用CProfile运行代码, 并报告分析器输出
```

    hello
     


```python
%time print('hello')  # 报告单条语句的执行时间
```

    hello
    Wall time: 0 ns
    


```python
%timeit print('hello')  # 多次运行一条语句, 计算平均执行时间(适合执行时间短的代码)
```

    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    ...
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    hello
    151 µs ± 17.3 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    


```python
%who  # 显示命名空间中的变量
```

    No variables match your requested type.
    


```python
%who_ls  # 显示命名空间中的变量
```




    []




```python
%whos  # 显示命名空间中的变量
```

    No variables match your requested type.
    


```python
%xdel a  # 删除一个变量, 并清空任何对它的引用
```

    NameError: name 'a # 删除一个变量, 并清空任何对它的引用' is not defined
    


```python
%matplotlib
```

    Using matplotlib backend: TkAgg
    


```python
%matplotlib inline
```


```python
import matplotlib.pyplot as plt
plt.plot(np.random.randn(50).cumsum())
```




    [<matplotlib.lines.Line2D at 0x1fe3aed0e80>]




![png](output_39_1.png)



```python
from datetime import datetime, date, time
```


```python
dt = datetime(2018, 7, 2, 9, 30, 59)
```


```python
dt.day
```




    2




```python
dt.minute
```




    30




```python
dt.date()
```




    datetime.date(2018, 7, 2)




```python
dt.strftime('%m/%d/%Y %H:%M')  # 将datetime格式化为字符串
```




    '07/02/2018 09:30'




```python
datetime.strptime('20180701', '%Y%m%d')  # 将字符串转换成datetime对象
```




    datetime.datetime(2018, 7, 1, 0, 0)




```python
dt.replace(minute=0, second=0)
```




    datetime.datetime(2018, 7, 2, 9, 0)




```python
dt2 = datetime(2018, 8, 2, 8, 0)
```


```python
delta = dt2 - dt  # 两个datetime对象的差会产生一个datetime.timedelta类型
```


```python
delta
```




    datetime.timedelta(30, 80941)




```python
type(delta)
```




    datetime.timedelta


