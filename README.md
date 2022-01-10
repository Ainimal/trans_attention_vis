# Trans_attention_vis
This is a super simple visualization toolbox (script) for transformer attention visualization ✌

<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/all.jpg" height="300" width="170" alt="input images"></td>
</tr>

</table>

## 1. How to prepare your attention matrix?
Just convert it to numpy array like this 👇
```python
# build an attetion matrixs as torch-output like
token_num = 6
case_num = 3
layer_num = 2
head_num = 4
attention_map_mhml = [np.stack([make_attention_map_mh(head_num, token_num)]*case_num, 0) for _ in range(layer_num)] # 4cases' 3 layers attention, with 3 head per layer( 每个case相同）
_ = [print(i.shape) for i in attention_map_mhml]

"""
>>>(3, 4, 6, 6)
(3, 4, 6, 6)
"""

```

## 2. Just try the following lines of code 👇
```python

# import function
from transformer_attention_visualization import *

# build canvas
scale = 3
canvas = np.zeros([120*scale,60*scale]).astype(np.float)

# build an attetion matrixs as torch-output like
token_num = 6
case_num = 3
layer_num = 2
head_num = 4
attention_map_mhml = [np.stack([make_attention_map_mh(head_num, token_num)]*case_num, 0) for _ in range(layer_num)] # 4cases' 3 layers attention, with 3 head per layer( 每个case相同）

_ = [print(i.shape) for i in attention_map_mhml]


# run for getting visualization picture (on the canvas)
import datetime
tic = datetime.datetime.now()
attention_vis2 = draw_attention_map_multihead_multilayer(canvas, attention_map_mhml, line_width=0.007)
toc = datetime.datetime.now()
h, remainder = divmod((toc - tic).seconds, 3600)
m, s = divmod(remainder, 60)
time_str2 = "Cost Time %02d h:%02d m:%02d s" % (h, m, s)
print(time_str2)


# show the visualization result
import matplotlib.pyplot as plt
def show2D(img2D, title = ''):
    ax1 = plt.gca()
    ax1.imshow(img2D,cmap=plt.cm.gray)
    ax1.yaxis.tick_right()
    ss = (scale) * 120 / (token_num - 1)
    y_spots = [(i)*ss for i in range(token_num)]
    ax1.set_yticks(y_spots)
    ax1.set_yticklabels(['Token '+ str(i) for i in range(token_num)])
    ax1.set_xticklabels([''])

    plt.title(title)
    plt.show()

case_index = 1
layer_index = 1
head_index = 1
beta = 2  # much bigger, contrast gose much higher

# show all token attention
show2D(attention_vis2[layer_index][case_index][head_index][0]**beta, title='case%d layer%d head%d all token'%(case_index, layer_index,head_index))

# show per token attention
for i in range(token_num):
    show2D(attention_vis2[layer_index][case_index][head_index][1][i]**beta, title='case%d layer%d head%d token%d'%(case_index, layer_index,head_index, i))


```
<table>

<!-- Line 1: Original Input -->
<tr>
    <th>all</th>
    <th>token 1</th>
    <th>token 2</th>
    <th>token 3</th>
    <th>token 4</th>
    <th>···</th>
</tr>
    
<tr>
    <td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/all.jpg" height="300" width="170" alt="input images"></td>
    <td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/1.jpg" height="300" width="170" alt="input images"></td>
    <td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/2.jpg" height="300" width="170" alt="input images"></td>
    <td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/3.jpg" height="300" width="170" alt="input images"></td>
    <td><img src="https://github.com/WAMAWAMA/trans_attention_vis/blob/main/pic/4.jpg" height="300" width="170" alt="input images"></td>
        <th>···</th>
</tr>

</table>
