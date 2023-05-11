from grconvnet_paddle import GenerativeResnet
class Model(nn.Layer):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single GenerativeResnet
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fundus_branch = GenerativeResnet(input_channels=3, channel_size=64, num_classes=0) # remove final fc
        self.oct_branch = GenerativeResnet(input_channels=256, channel_size=64, num_classes=0) # remove final fc
        self.decision_branch = nn.Linear(1024 * 2, 3) # 

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = paddle.flatten(b1, 1)
        b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(paddle.concat([b1, b2], 1))

        return logit