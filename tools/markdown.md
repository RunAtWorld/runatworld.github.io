# 一、简明篇
## 1. 斜体和粗体
使用 `*` 和 `**` 表示斜体和粗体。

示例：

这是 *斜体*，这是 **粗体**。

## 2. 分级标题
使用 `===` 表示一级标题，使用 `---` 表示二级标题。

示例：

这是一个一级标题
============================
这是一个二级标题
--------------------------------------------------
### 这是一个三级标题
你也可以选择在行首加井号表示不同级别的标题 (H1-H6)，例如：# H1, ## H2, ### H3，#### H4。

## 3. 外链接
使用 `[描述](链接地址)` 为文字增加外链接。

示例：

这是去往 [本人博客](https://runatworld.gitbook.io) 的链接。

## 4. 无序列表
使用 `*，+，-` 表示无序列表。

示例：

* 无序列表项 一
* 无序列表项 二
* 无序列表项 三

## 5. 有序列表
使用数字和点表示有序列表。

示例：

1. 有序列表项 一
1. 有序列表项 二
1. 有序列表项 三

## 6. 文字引用
使用 `>` 表示文字引用。

示例：

> 野火烧不尽，春风吹又生。

## 7. 行内代码块
使用 `代码` 表示行内代码块。

示例：

`让我们聊聊 html。`

## 8. 代码块
使用 四个缩进空格 表示代码块。

示例：

    这是一个代码块，此行左侧有四个不可见的空格。


## 9. 插入图像
使用 `![描述](图片链接地址) ` 插入图像。

示例：

![我的头像](https://www.zybuluo.com/static/img/my_head.jpg)

# 二、高阶篇
## 1. 内容目录
在段落中填写 `[TOC]` 以显示全文内容的目录结构。

[TOC]

## 2. 标签分类
在编辑区任意行的列首位置输入以下代码给文稿标签：

`标签： 数学 英语 Markdown`

或者

`Tags： 数学 英语 Markdown`

## 3. 删除线
使用 `~~` 表示删除线。

~~这是一段错误的文本~~。

## 4. 注脚
使用 `[^keyword]` 表示注脚。

这是一个注脚[^1]的样例。

这是第二个注脚[^2]的样例。

## 5. LaTeX 公式
`$` 表示行内公式：

质能守恒方程可以用一个很简洁的方程式 $E = mc^2$ 来表达。

`$$` 表示整行公式：

$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$


访问 [MathJax](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference) 参考更多使用方法。

## 6. 加强的代码块

```
```表示代码块
```

shell 示例
```shell
$ sudo apt-get install vim-gnome
```

Python 示例：

```python
@requires_authorization
def somefunc(param1='', param2=0):
    '''A docstring'''
    if param1 > param2: # interesting
        print 'Greater'
    return (param2 - param1 + 1) or None
class SomeClass:
    pass
>>> message = '''interpreter
... prompt'''
```

JavaScript 示例：
```javascript
/**
* nth element in the fibonacci series.
* @param n >= 0
* @return the nth element, >= 0.
*/
function fib(n) {
  var a = 1, b = 1;
  var tmp;
  while (--n >= 0) {
    tmp = a;
    a += b;
    b = tmp;
  }
  return a;
}
document.write(fib(10));
```

## 7. 流程图
示例

```
st=>start: Start:>http://www.google.com[blank]
e=>end:>http://www.google.com
op1=>operation: My Operation
sub1=>subroutine: My Subroutine
cond=>condition: Yes
or No?:>http://www.google.com
io=>inputoutput: catch something...
para=>parallel: parallel tasks
st->op1->cond
cond(yes)->io->e
cond(no)->para
para(path1, bottom)->sub1(right)->op1
para(path2, top)->op1
```

更多语法参考：[程图语法参考](http://adrai.github.io/flowchart.js/)

## 8. 序列图
示例

```
Andrew->China: Says Hello
Note right of China: China thinks\nabout it
China-->Andrew: How are you?
Andrew->>China: I am good thanks!
```
更多语法参考：[序列图语法参考](https://bramp.github.io/js-sequence-diagrams/)

## 9. 表格支持
示例：
```
项目 | 价格 | 数量
--- | --- |---
计算机 | $1600 | 5
手机 | $12	|  12
管线 | $1 | 234
```

项目 | 价格 | 数量
--- | --- |---
计算机 | $1600 | 5
手机 | $12	|  12
管线 | $1 | 234

## 10. Html 标签
本站支持在 Markdown 语法中嵌套 Html 标签，譬如，你可以用 Html 写一个纵跨两行的表格：
```
<table>
    <tr>
        <th rowspan="2">值班人员</th>
        <th>星期一</th>
        <th>星期二</th>
        <th>星期三</th>
    </tr>
    <tr>
        <td>李强</td>
        <td>张明</td>
        <td>王平</td>
    </tr>
</table>
```


<table>
    <tr>
        <th rowspan="2">值班人员</th>
        <th>星期一</th>
        <th>星期二</th>
        <th>星期三</th>
    </tr>
    <tr>
        <td>李强</td>
        <td>张明</td>
        <td>王平</td>
    </tr>
</table>

11. 内嵌图标
在文档中输入

<i class="icon-weibo"></i>
即显示微博的图标： 

替换 上述 i 标签 内的 icon-weibo 以显示不同的图标，例如：

<i class="icon-renren"></i>
即显示人人的图标： 

更多的图标和玩法可以参看 font-awesome 官方网站。

## 13. 待办事宜 Todo 列表
使用带有 `[ ]` 或 `[x]` （未完成或已完成）项的列表语法撰写一个待办事宜列表，并且支持子列表嵌套以及混用Markdown语法，例如：

```
- [ ] **Cmd Markdown 开发**
    - [ ] 改进 Cmd 渲染算法，使用局部渲染技术提高渲染效率
    - [ ] 支持以 PDF 格式导出文稿
    - [x] 新增Todo列表功能 [语法参考](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
    - [x] 改进 LaTex 功能
        - [x] 修复 LaTex 公式渲染问题
        - [x] 新增 LaTex 公式编号功能 [语法参考](http://docs.mathjax.org/en/latest/tex.html#tex-eq-numbers)
- [ ] **七月旅行准备**
    - [ ] 准备邮轮上需要携带的物品
    - [ ] 浏览日本免税店的物品
    - [x] 购买蓝宝石公主号七月一日的船票
```
对应显示如下待办事宜 Todo 列表：


- [ ] **Cmd Markdown 开发**
    - [ ] 改进 Cmd 渲染算法，使用局部渲染技术提高渲染效率
    - [ ] 支持以 PDF 格式导出文稿
    - [x] 新增Todo列表功能 [语法参考](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
    - [x] 改进 LaTex 功能
        - [x] 修复 LaTex 公式渲染问题
        - [x] 新增 LaTex 公式编号功能 [语法参考](http://docs.mathjax.org/en/latest/tex.html#tex-eq-numbers)
- [ ] **七月旅行准备**
    - [ ] 准备邮轮上需要携带的物品
    - [ ] 浏览日本免税店的物品
    - [x] 购买蓝宝石公主号七月一日的船票

---
---
[1] 这是一个 注脚 的 文本。 

[2] 这是另一个 注脚 的 文本。