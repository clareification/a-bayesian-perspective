import torch.nn.functional as F
import torch.nn as nn

from torchvision import models, datasets, transforms

class SmallNet(nn.Module):
    def __init__(self, class_size, n_channels, dropout_rate, num_filters=2):
        super(SmallNet, self).__init__()

        #parameters 
        self.class_size = class_size
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.n_channels, num_filters*4, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7*7*num_filters*8, self.class_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x



class TwoLayer(nn.Module):
  def __init__(self, D_in, H, D_out):
      """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
      """
      super(TwoLayer, self).__init__()
      self.linear1 = torch.nn.Linear(D_in, H)
      self.linear2 = torch.nn.Linear(H, D_out)

  def forward(self, x):
      """
      In the forward function we accept a Tensor of input data and we must return
      a Tensor of output data. We can use Modules defined in the constructor as
      well as arbitrary operators on Tensors.
      """
      h_relu = self.linear1(x).clamp(min=0)
      
      y_pred = self.linear2(h_relu)
      return y_pred

class Linear(nn.Module):
  def __init__(self, D_in, D_out):
    super(Linear, self).__init__()
    self.linear = torch.nn.Linear(D_in, D_out)
  
  def forward(self, x):
    o = self.linear(x)
    return o 
class ConvexCombo(nn.Module):
    def __init__(self, D_in, D_out):
      super(ConvexCombo, self).__init__()
      self.d_in = D_in
      self.linear = torch.nn.Linear(D_in, D_out)
  
    def forward(self, x):
      ps = self.linear(torch.eye(self.d_in))
      o = torch.nn.Softmax(0)(ps)
      o = x @ o
      return o