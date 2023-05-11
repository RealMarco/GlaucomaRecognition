import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import BasicBlock # ,BottleneckBlock

class GenerativeResnet(nn.Layer):

    def __init__(self, input_channels=3, output_channels=1, channel_size=64, 
				 dropout=False, prob=0.0, 
				 num_classes=1000):
        super(GenerativeResnet, self).__init__()
        
        self.num_classes = num_classes
		
        self.conv1 = nn.Conv2D(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2D(channel_size)

        self.conv2 = nn.Conv2D(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2D(channel_size * 2)

        self.conv3 = nn.Conv2D(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2D(channel_size * 4)

        self.res1 = BasicBlock(channel_size * 4, channel_size * 4)  # BottleneckBlock
        self.res2 = BasicBlock(channel_size * 4, channel_size * 4)
        self.res3 = BasicBlock(channel_size * 4, channel_size * 4)
        self.res4 = BasicBlock(channel_size * 4, channel_size * 4)
        self.res5 = BasicBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.Conv2DTranspose(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2D(channel_size * 2)

        self.conv5 = nn.Conv2DTranspose(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2D(channel_size)

        self.conv6 = nn.Conv2DTranspose(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
		
		# Classification
        self.avgpool1 = nn.AdaptiveAvgPool2D((128, 128))
        
        self.conv7 = nn.Conv2D(channel_size, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.avgpool2 = nn.AdaptiveAvgPool2D((32, 32))

        if num_classes > 0:
            self.fc = nn.Linear(1024, num_classes)
		
        """
		self.pos_output = nn.Conv2D(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2D(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2D(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2D(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)
		

        for m in self.modules():
            if isinstance(m, (nn.Conv2D, nn.Conv2DTranspose)):
                nn.init.xavier_uniform_(m.weight, gain=1)
		"""

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
		
        x = self.avgpool1(x)
        x = self.conv7(x)
        x = self.avgpool2(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x
		
        """
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
		"""
