
import paddle
import paddle.nn as nn



class CAM(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim//4, 5, padding=2, groups=dim//4)
        self.conv0_1 = nn.Conv2D(dim, dim//4, (1, 7), padding=(0, 3), groups=dim//4)
        self.conv0_2 = nn.Conv2D(dim//4, dim//4, (7, 1), padding=(3, 0), groups=dim//4)

        self.conv1_1 = nn.Conv2D(dim, dim//4, (1, 11), padding=(0, 5), groups=dim//4)
        self.conv1_2 = nn.Conv2D(dim//4, dim//4, (11, 1), padding=(5, 0), groups=dim//4)

        self.conv2_1 = nn.Conv2D(dim, dim//4, (1, 17), padding=(0, 8), groups=dim//4)
        self.conv2_2 = nn.Conv2D(dim//4, dim//4, (17, 1), padding=(8, 0), groups=dim//4)
        self.conv3 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        attn_0 = self.conv0_1(x)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(x)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(x)
        attn_2 = self.conv2_2(attn_2)

        attn = paddle.concat([attn , attn_0 ,attn_1 , attn_2], axis=1)

        attn = self.conv3(attn)* u
        attn =attn + u
        return attn