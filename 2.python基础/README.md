

```python
tup = 4, 5, 6
```


```python
tup
```




    (4, 5, 6)




```python
nested_tup = (4, 5, 6), (7, 8)
```


```python
nested_tup
```




    ((4, 5, 6), (7, 8))




```python
tuple([4, 0, 2])
```




    (4, 0, 2)




```python
tup = tuple('string')
```


```python
tup
```




    ('s', 't', 'r', 'i', 'n', 'g')




```python
tup[0]
```




    's'




```python
tup = tuple(['foo', [1, 2], True])
```


```python
tup[2] = False
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-b89d0c4ae599> in <module>()
    ----> 1 tup[2] = False
    

    TypeError: 'tuple' object does not support item assignment



```python
tup[1].append(3)
```


```python
tup
```




    ('foo', [1, 2, 3], True)




```python
(4, None, 'foo') + (6, 0) + ('bar', )  # 用加号运算符将元组串联起来
```




    (4, None, 'foo', 6, 0, 'bar')




```python
tup = (4, 5, 6)  # 拆分元组
```


```python
a, b, c = tup
```


```python
b
```




    5




```python
tup = 4, 5, (6, 7)
```


```python
a, b, (c, d) = tup
```


```python
d
```




    7




```python
a, b = 1, 2
```


```python
a
```




    1




```python
b
```




    2




```python
a, b = b, a
```


```python
a
```




    2




```python
b
```




    1




```python
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
```


```python
for a, b, c in seq:
    print('a={}, b={}, c={}'.format(a, b, c))
```

    a=1, b=2, c=3
    a=4, b=5, c=6
    a=7, b=8, c=9
    


```python
values = 1, 2, 3, 4, 5
```


```python
a, b, *rest = values
```


```python
a, b
```




    (1, 2)




```python
values
```




    (1, 2, 3, 4, 5)




```python
type(values)
```




    tuple




```python
a, b, *_ = values
```


```python
_
```




    [3, 4, 5]




```python
a = (1, 2, 2, 2, 3, 4, 2)
```


```python
a.count(2)
```




    4




```python
# 列表
```


```python
a_list = [2, 3, 7, None]
```


```python
tup = ('foo', 'bar', 'baz')
```


```python
b_list = list(tup)
```


```python
b_list
```




    ['foo', 'bar', 'baz']




```python
b_list[1] = 'peekaboo'
```


```python
b_list
```




    ['foo', 'peekaboo', 'baz']




```python
gen = range(10)
```


```python
gen
```




    range(0, 10)




```python
list(gen)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
# 添加删除元素
```


```python
b_list.append('dwarf')  # 在列表末尾添加元素
```


```python
b_list
```




    ['foo', 'peekaboo', 'baz', 'dwarf']




```python
b_list.insert(1, 'red')  # 在特定的位置插入元素
```


```python
b_list
```




    ['foo', 'red', 'peekaboo', 'baz', 'dwarf']




```python
b_list.pop(2)  # insert的逆运算是pop，它移除并返回指定位置的元素
```




    'peekaboo'




```python
b_list.append('foo')
```


```python
b_list
```




    ['foo', 'red', 'baz', 'dwarf', 'foo']




```python
b_list.remove('foo')  # 去除某个值，remove会先寻找第一个值并除去
```


```python
b_list
```




    ['red', 'baz', 'dwarf', 'foo']




```python
# 串联和组合列表
```


```python
[4, None, 'foo'] + [7, 8, (2, 3)]
```




    [4, None, 'foo', 7, 8, (2, 3)]




```python
x = [4, None, 'foo']
```


```python
x.extend([7, 8, (2, 3)])  # 用extend方法可以追加多个元素
```


```python
x
```




    [4, None, 'foo', 7, 8, (2, 3)]




```python
everything = []  # 比串联方法快
for chunk in list_of_lists:
    everything = everything + chunk
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-62-4bbc37b20511> in <module>()
          1 everything = []  # 比串联方法快
    ----> 2 for chunk in list_of_lists:
          3     everything = everything + chunk
    

    NameError: name 'list_of_lists' is not defined



```python
# 排序
```


```python
a = [7, 2, 5, 1, 3]
```


```python
a.sort()  # 将一个列表原地排序
```


```python
a
```




    [1, 2, 3, 5, 7]




```python
b = ['saw', 'small', 'He', 'foxes', 'six']
```


```python
b.sort(key=len)  # 按长度对字符串进行排序
```


```python
b
```




    ['He', 'saw', 'six', 'small', 'foxes']




```python
import bisect
```


```python
c = [1, 2, 2, 2, 3, 4, 7]
```


```python
bisect.bisect(c, 2)  # 找到插入值后仍保证排序的位置
```




    4




```python
bisect.bisect(c, 5)
```




    6




```python
bisect.insort(c, 6)  # 向这个位置插入值
```


```python
c
```




    [1, 2, 2, 2, 3, 4, 6, 7]




```python
# 切片
```


```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
```


```python
seq[1:5]
```




    [2, 3, 7, 5]




```python
seq[3:4] = [6, 3]
```


```python
seq
```




    [7, 2, 3, 6, 3, 5, 6, 0, 1]




```python
seq[:5]
```




    [7, 2, 3, 6, 3]




```python
seq[3:]
```




    [6, 3, 5, 6, 0, 1]




```python
seq[-4:]  # 负数表明从后向前切片
```




    [5, 6, 0, 1]




```python
# enumerate函数  返回(i, value)元组序列
```


```python
some_list = ['foo', 'bar', 'baz']
```


```python
mapping = {}
```


```python
for i, v in enumerate(some_list):
    mapping[v] = i
```


```python
mapping
```




    {'foo': 0, 'bar': 1, 'baz': 2}




```python
# sorted函数(从任意序列的元素返回一个新的排好序的列表, 可以接受和sort相同的参数)
```


```python
sorted([7, 1, 2, 6, 0, 3, 2])
```




    [0, 1, 2, 2, 3, 6, 7]




```python
sorted('horse race')
```




    [' ', 'a', 'c', 'e', 'e', 'h', 'o', 'r', 'r', 's']




```python
# zip函数(将多个列表、元组或其它序列成对组合成一个元组列表)
```


```python
seq1 = ['foo', 'bar', 'baz']
```


```python
seq2 = ['one', 'two', 'three']
```


```python
zipped = zip(seq1, seq2)
```


```python
list(zipped)
```




    [('foo', 'one'), ('bar', 'two'), ('baz', 'three')]




```python
seq3 = [False, True]
```


```python
list(zip(seq1, seq2, seq3))  # 可以处理任意多的序列，元素的个数取决于最短的序列
```




    [('foo', 'one', False), ('bar', 'two', True)]




```python
# zip的常见用法之一是同时迭代多个序列，可能结合enumerate使用
```


```python
for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{}: {}, {}'.format(i, a, b))
```

    0: foo, one
    1: bar, two
    2: baz, three
    


```python
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
```


```python
first_names, last_names = zip(*pitchers)  # zip可以被用来解压序列
```


```python
first_names
```




    ('Nolan', 'Roger', 'Schilling')




```python
last_names
```




    ('Ryan', 'Clemens', 'Curt')




```python
# reversed可以从后向前迭代一个序列
```


```python
list(reversed(range(10)))
```




    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]




```python
# 字典
```


```python
empty_dict = {}
```


```python
d1 = {'a': 'some value', 'b' : [1, 2, 3, 4]}
```


```python
d1
```




    {'a': 'some value', 'b': [1, 2, 3, 4]}




```python
d1[7] = 'an integer'
```


```python
d1
```




    {'a': 'some value', 'b': [1, 2, 3, 4], 7: 'an integer'}




```python
d1['b']
```




    [1, 2, 3, 4]




```python
'b' in d1
```




    True




```python
d1[5] = 'some value'
```


```python
d1['dummy'] = 'another value'
```


```python
d1
```




    {'a': 'some value',
     'b': [1, 2, 3, 4],
     7: 'an integer',
     5: 'some value',
     'dummy': 'another value'}




```python
del d1[5]  # del关键字删除值
```


```python
ret = d1.pop('dummy')  # pop方法（返回值得同时删除键）删除值
```


```python
ret
```




    'another value'




```python
d1.update({'b' : 'foo', 'c' : 12})  # 用update方法将一个字典与另一个融合
```


```python
d1
```




    {'a': 'some value', 'b': 'foo', 7: 'an integer', 'c': 12}




```python
# 用序列创建字典
```


```python
mapping = dict(zip(range(5), reversed(range(5))))
```


```python
mapping
```




    {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}




```python
words = ['apple', 'bat', 'bar', 'atom', 'book']
```


```python
by_letter = {}
```


```python
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
```


```python
by_letter
```




    {'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']}




```python
# 有效的键类型
```


```python
hash('string')  # 检测对象是否是可哈希的(不可变的标量类型)
```




    827150610984278387




```python
hash((1, 2, (2, 3)))
```




    1097636502276347782




```python
hash((1, 2, [2, 3]))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-133-8ffc25aff872> in <module>()
    ----> 1 hash((1, 2, [2, 3]))
    

    TypeError: unhashable type: 'list'



```python
d = {}
```


```python
d[tuple([1, 2, 3])] = 5
```


```python
d
```




    {(1, 2, 3): 5}




```python
# 集合
```


```python
# 集合是无序的不可重复的元素的集合。可以把它当做字典，但是只有键没有值。
# 可以用两种方式创建集合：通过set函数或使用尖括号set语句
```


```python
set([2, 2, 2, 1, 3, 3])
```




    {1, 2, 3}




```python
{2, 2, 2, 1, 3, 3}
```




    {1, 2, 3}




```python
a = {1, 2, 3, 4, 5}
```


```python
b = {3, 4, 5, 6, 7, 8}
```


```python
a.union(b)  # 合并是取两个集合中不重复的元素
```




    {1, 2, 3, 4, 5, 6, 7, 8}




```python
a | b
```




    {1, 2, 3, 4, 5, 6, 7, 8}




```python
a.intersection(b)  # 交集的元素包含在两个集合中
```




    {3, 4, 5}




```python
a & b
```




    {3, 4, 5}




```python
c = a.copy()
```


```python
c |= b
```


```python
c
```




    {1, 2, 3, 4, 5, 6, 7, 8}




```python
d = a.copy()
```


```python
d &= b
```


```python
a_set = {1, 2, 3, 4, 5}
```


```python
{1, 2, 3}.issubset(a_set)  # 检测一个集合是否是另一个集合的子集
```




    True




```python
a_set.issuperset({1, 2, 3})  # 检测一个集合是否是另一个集合的父集
```




    True




```python
{1, 2, 3} == {3, 2, 1}  # 集合的内容相同时，集合才对等
```




    True




```python
# 列表、集合和字典推导式
```


```python
# 列表推导式
# [expr for val in collection if condition]
```


```python
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
```


```python
[x.upper() for x in strings if len(x) > 2]
```




    ['BAT', 'CAR', 'DOVE', 'PYTHON']




```python
# 字典的推导式
# dict_comp = {key-expr : value-expr for value in collection if condition}
```


```python
# 集合的推导式
# set_comp = {expr for value in collection if condition}
```


```python
unique_lengths = {len(x) for x in strings}
```


```python
unique_lengths
```




    {1, 2, 3, 4, 6}




```python
set(map(len, strings))
```




    {1, 2, 3, 4, 6}




```python
loc_mapping = {val : index for index, val in enumerate(strings)}
```


```python
loc_mapping
```




    {'a': 0, 'as': 1, 'bat': 2, 'car': 3, 'dove': 4, 'python': 5}




```python
# 嵌套列表推导式
```


```python
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
```


```python
result = [name for names in all_data for name in names if name.count('e') >= 2]
```


```python
result
```




    ['Steven']




```python
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
```


```python
flattened = [x for tup in some_tuples for x in tup]
```


```python
flattened
```




    [1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
[[x for x in tup] for tup in some_tuples]
```




    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]




```python
# 函数也是对象
```


```python
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'south   carolina##', 'West virginia?']
```


```python
import re

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result
```


```python
clean_strings(states)
```




    ['Alabama',
     'Georgia',
     'Georgia',
     'Georgia',
     'Florida',
     'South   Carolina',
     'West Virginia']




```python
def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result
```


```python
clean_strings(states, clean_ops)
```




    ['Alabama',
     'Georgia',
     'Georgia',
     'Georgia',
     'Florida',
     'South   Carolina',
     'West Virginia']




```python
for x in map(remove_punctuation, states):
    print(x)
```

       Alabama 
    Georgia
    Georgia
    georgia
    FlOrIda
    south   carolina
    West virginia
    


```python
# 匿名（lambda）函数
```


```python
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
```


```python
strings.sort(key=lambda x: len(set(list(x))))
```


```python
strings
```




    ['aaaa', 'foo', 'abab', 'bar', 'card']




```python
# 柯里化：部分参数应用
```


```python
# 柯里化（currying）是一个计算机科学术语，它指的是通过“部分参数应用”（partial argument application）从现有函数派生出新函数的技术
```


```python
def add_numbers(x, y):
    return x + y
```


```python
add_five = lambda y: add_numbers(5, y)  # 第二个参数称为“柯里化的”（curried）
```


```python
# 生成器
```


```python
# 能以一种一致的方式对序列进行迭代（比如列表中的对象或文件中的行）是Python的一个重要特点。这是通过一种叫做迭代器协议（iterator protocol，它是一种使对象可迭代的通用方式）的方式实现的，一个原生的使对象可迭代的方法。
```


```python
# 生成器（generator）是构造新的可迭代对象的一种简单方式。一般的函数执行之后只会返回单个值，而生成器则是以延迟的方式返回一个值序列，即每返回一个值之后暂停，直到下一个值被请求时再继续。要创建一个生成器，只需将函数中的return替换为yeild即
```


```python
def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))
    for i in range(1, n + 1):
        yield i ** 2
```


```python
gen = squares()
```


```python
gen
```




    <generator object squares at 0x0000012AA52943B8>




```python
for x in gen:
    print(x, end=' ')
```

    Generating squares from 1 to 100
    1 4 9 16 25 36 49 64 81 100 


```python
# 生成器表达式
```


```python
gen = (x ** 2 for x in range(100))
```


```python
gen
```




    <generator object <genexpr> at 0x0000012AA5294258>




```python
sum(x ** 2 for x in range(100))
```




    328350




```python
dict((i, i **2) for i in range(5))
```




    {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}




```python
# itertools模块
```


```python
import itertools
```


```python
first_letter = lambda x: x[0]
```


```python
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
```


```python
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator
```

    A ['Alan', 'Adam']
    W ['Wes', 'Will']
    A ['Albert']
    S ['Steven']
    


```python
# 错误和异常处理
```


```python
def attempt_float(x):
    try:
        return float(x)
    except:
        return x
```


```python
attempt_float('1.2345')
```




    1.2345




```python
attempt_float('something')
```




    'something'




```python
# 文件和操作系统
```


```python
path = './ipython_script_test.py'
```


```python
f = open(path)
```


```python
lines = [x.strip() for x in open(path)]
```


```python
lines
```




    ['',
     '# coding: utf-8',
     '',
     '',
     'def f(x, y, z):',
     'return (x + y) / z',
     '',
     'a = 5',
     'b = 6',
     'c = 7.5',
     '',
     'result = f(a, b, c)',
     '']




```python
f.close()
```


```python
with open(path) as f:
    lines = [x.strip() for x in f]
```


```python
f = open(path)
```


```python
f.read(10)
```




    '\n# coding:'




```python
f2 = open(path, 'rb')
```


```python
f2.read(10)
```




    b'\n# coding:'




```python
f.tell()
```




    10




```python
f2.tell()
```




    10




```python
import sys
```


```python
sys.getdefaultencoding()
```




    'utf-8'


