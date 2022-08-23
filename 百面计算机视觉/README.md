# 深度学习基础

## 1*1卷积

最早出现在Network in Network， 两点贡献：

- MLpconv，即引入1*1卷积

传统的CNN的一层卷积相当于一个线性操作，如下图 a，所以只提取了线性特征，隐含了假设特征是线性可分的，实际却并非如此，NIN 中引入 下图 b 中的 mlpconv layer，实质是像素级的全连接层，等价于 1x1 卷积，在其后跟 ReLU激活函数，引入更多的非线性元素。

- 将分类的全连接层用global average pooling代替。

全连接容易产生过拟合，减弱了网络的泛化能力。

![](README.assets/1conv1.png)

此后GoogLeNet的Inception结构中延用了1*1卷积，如下图

![](README.assets/1conv3.png)

### 1*1卷积的作用

1. 减低维度或上升维度

   在网络中增加1*1卷积，使得网络更深，**通过在3×3或5×5卷积前，用1×1卷积降低维度，没有增加权重参数的负担**。

2. 跨通道信息交互（cross-channel correlations and spatial correlations）

   1x1卷积核，从图像处理的角度，乍一看也没有意义，在网络中，**这样的降维和升维的操作其实是 channel 间信息的线性组合变化。**

   **补充：**cross-channel correlation 和 spatial correlation的学习可以进行解耦。1x1的卷积相当于学习了feature maps之间的cross-channel correlation。实验证明了这种解耦可以在不损害模型表达能力的情况下大大减少参数数量和计算量。**但是需要注意的是，1x1 的卷积层后面加上一个 normal 的卷积层，这种解耦合并不彻底，正常卷积层仍然存在对部分的 cross-channel correlation 的学习。**之后就有了 depth-wise seperable convolution(后面记录 MobileNet 后，在这添加链接)。在 depth-wise seperable convolution中，1x1 的卷积层是将每一个 channel 分为一组，那么就不存在对cross-channel correlation的学习了，就实现了对cross-channel correlation和spatial correlation的彻底解耦合。这种完全解耦的方式虽然可以大大降低参数数量和计算量，但是正如在 mobile net 中所看到的，性能会受到很大的损失。

3. 增加非线性特性

   1x1卷积核，可以在保持 feature maps size不变的（即不损失分辨率）的前提下大幅增加非线性特性（**利用后接的非线性激活函数**）。

**总结**
1x1 卷积在图像处理的角度，乍一看好像没什么意义，但在 CNN 网络中，能实现降维，减少 weights 参数数量，能够实现升维，来拓宽 feature maps，在不改变 feature maps 的 size 的前提下，实现各通道之间的线性组合，实际上是通道像素之间的线性组合，后接非线性的激活函数，增加更多样的非线性特征。这就是为什么 GoogLeNet 用 1x1 卷积来降维，减少了计算量，但模型效果却没有降低，此外网络深度更深。可以说 1x1 卷积很 nice.

## CV中的Attention

#### Non-local Attention

![](README.assets/640.png)

#### CBAM

CBAM由Channel Attention和Spatial Attention组合而成。

<img src="README.assets/640-16612683554207.png" style="zoom:80%;" />

其中**Channel Attention，主要是从$C×W×W$的维度，学习到一个$C×1×1$的权重矩阵。**

论文原图如下：

![](README.assets/90347033-1270-47ff-9d1e-b407665ca71e-166126864193014.png)

```python
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
		
        # 共享MLP权重
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
```

对于$Spatial$ $Attention$，如图所示：


![](README.assets/8551d439-9b14-4825-a9c4-103e76cfc33c.png)

参考代码如下

```python
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
```

#### cgnl

论文分析了下如$Spatial$ $Attention$与$Channel$ $Attention$均不能很好的描述特征之间的关系，这里比较极端得生成了N * 1 * 1 * 1的$MASK$.

![](README.assets/a44002c5-e1e0-4a09-82b2-6b94b403b822.png)

Attention 代码：

```python
def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x
```

### Cross-layer non-local

论文中分析了，同样的层之间进行 $Attention$计算，感受野重复，会造成冗余，如左边的部分图，而右边的图表示不同层间的感受野不同，计算全局 $Attention$也会关注到更多的区域。


![](README.assets/ccd91ecf-e976-4549-b668-7e47a8e92720.png)

这里采用跨层之间的 $Attention$生成。

![](README.assets/ddb58e75-5cef-4eb6-959b-e0b0a1e8c4bb.png)

代码

```python
# query : N, C1, H1, W1
# key: N, C2, H2, W2
# value: N, C2, H2, W2
# 首先，需要使用1 x 1 卷积，使得通道数相同
q = query_conv(query) # N, C, H1, W1
k = key_conv(key) # N, C, H2, W2
v = value_conv(value) # N, C, H2, W2
att = nn.softmax(torch.bmm(q.view(N, C, H1*W1).permute(0, 1, 2), k.view(N, C, H2 * W2))) # (N, H1*W1, H2*W2)
out = att * value.view(N, C2, H2*W2).permute(0, 1, 2) #(N, H1 * W1, C)
out = out.view(N, C1, H1, W1)
```

