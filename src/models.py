def get_LeNet(conv_type=nn.Conv2d, activation=F.selu):
    
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()        
            self.conv1 = conv_type(3, 40, 5, 1)
            self.conv2 = conv_type(40, 40, 5, 1)
            self.fc1 = nn.Linear(5 * 5 * 40, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
                
            x = activation(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = self.conv2(x)
            
            x = activation(x)
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 5 * 5 * 40)
            x = activation(self.fc1(x))
            x = self.fc2(x)
            return x

    def name(self):
        return "LeNet"

    return LeNet()
