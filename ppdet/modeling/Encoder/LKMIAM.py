import paddle
import paddle.nn as nn

class LKMIAM(nn.Layer):
    def __init__(self, in_channel, ratio=2):
        super(LKMIAM, self).__init__()

        self.ratio = ratio
        self.LKMIA = LKMIA(in_channel // self.ratio)

    @staticmethod
    def channel_shuffle(x, ratio):
        b, c, h, w = x.shape

        x = x.reshape((b, ratio, -1, h, w))
        x = x.transpose([0, 2, 1, 3, 4])

        # flatten
        x = x.reshape((b, -1, h, w))

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape((b * self.ratio, -1, h, w))
        out = self.LKMIA(x)

        out = out.reshape((b, -1, h, w))
        out = self.channel_shuffle(out, 2)
        return out

class LKMIA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim , 1)
        self.conv2 = nn.Conv2D(dim, dim , 1)
        self.conv = nn.Conv2D(dim , dim, 1)
        self.GCSE=GCSE(dim*2)

    def forward(self, x):
        # b, c, h, w = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = paddle.concat([attn1, attn2], axis=1)

        attn3=self.GCSE(attn)
        attn1=attn1*attn3*attn2
        attn = self.conv (attn1)*x

        return  attn


class GCSE(nn.Layer):
    def __init__(self, in_channels, scale=16,numbers=4):
        super(GCSE, self).__init__()
        self.in_channels = in_channels
        self.numbers = numbers
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2D(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(axis=1)

        self.Conv_value1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
        )

        self.Conv_value2 = nn.Sequential(
            nn.Conv2D(self.out_channels*self.numbers, self.in_channels//2, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
        query = x.reshape([b, c, h * w])
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = paddle.matmul(query, key)
        concate_QK = concate_QK.reshape([b, c, 1, 1])
        value1 = self.Conv_value1(concate_QK)
        for i in range(1, self.numbers):
            key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
            query = x.reshape([b, c, h * w])
            # [b, c, h*w] * [b, H*W, 1]
            concate_QK = paddle.matmul(query, key)
            concate_QK = concate_QK.reshape([b, c, 1, 1])
            value2 = self.Conv_value1(concate_QK)
            value1 = paddle.concat([value1, value2], axis=1)
        out=self.Conv_value2(value1)

        return out



