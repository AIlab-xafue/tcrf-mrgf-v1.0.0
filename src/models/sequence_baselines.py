from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
 def __init__(self):super().__init__();self.c=nn.Conv1d(25,32,2);self.d=nn.Dropout(.3);self.o=nn.Linear(32,1)
 def forward(self,x):return self.o(F.adaptive_avg_pool1d(self.d(F.relu(self.c(x.transpose(1,2)))),1).squeeze(-1)).squeeze(-1)

class RNN(nn.Module):
 def __init__(self,k):super().__init__();self.r=k(25,32,1,batch_first=True);self.d=nn.Dropout(.3);self.o=nn.Linear(32,1)
 def forward(self,x):return self.o(self.d(self.r(x)[0][:,-1])).squeeze(-1)
