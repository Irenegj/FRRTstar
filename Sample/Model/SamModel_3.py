import torch.nn as nn
import torch
import torch.nn.functional as F
import math
'''
功能：提供STARN模型的计算流程
'''
class EmbforPosition(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbforPosition, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
    def forward(self, position):
        _, embp = self.l(position)
        #print(embp.size())
        out = F.relu(embp)
        return out
class EmbforThreats(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbforThreats, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(self.input_size, hidden_size)
        self.l2 = torch.nn.Linear(self.hidden_size, hidden_size)
    def forward(self, threats):
        embt1 = self.l1(threats)
        embt2 = self.l2(embt1)
        out = F.relu(embt2)
        return out

class Sample(nn.Module):
    def __init__(self,hidSizeforSam, outputSize, hidSizeforPosition, hidSizeforThreat):
        super(Sample, self).__init__()
        self.hidSizeforPosition = hidSizeforPosition
        self.hidSizeforThreat = hidSizeforThreat
        self.hidSizeforSam = hidSizeforSam
        self.outputSize = outputSize
        self.l1 = torch.nn.Linear(self.hidSizeforPosition+self.hidSizeforThreat, self.hidSizeforSam)
        self.l2 = torch.nn.Linear(self.hidSizeforSam, outputSize)
    def forward(self, emb):
        h1 = self.l1(emb)
        h2 = F.relu(h1)
        out = self.l2(h2)

        return out

class Model_layer(nn.Module):
    def __init__(self,inpSizeforPosition,inpSizeforThreat,hidSizeforPosition,hidSizeforThreat,hidSizeforSam,outputSize):
        super(Model_layer, self).__init__()
        self.inpSizeforPosition = inpSizeforPosition
        self.inpSizeforThreat = inpSizeforThreat
        self.hidSizeforPosition = hidSizeforPosition
        self.hidSizeforThreat = hidSizeforThreat
        self.EmbforPosition = EmbforPosition(self.inpSizeforPosition,self.hidSizeforPosition)
        self.EmbforThreats = EmbforThreats(self.inpSizeforThreat,self.hidSizeforThreat)
        self.hidSizeforSam = hidSizeforSam
        self.outputSize = outputSize
        self.Sample = Sample(self.hidSizeforSam,self.outputSize,self.hidSizeforPosition,self.hidSizeforThreat)
    def forward(self, position,threats):
        #print(position.size())
        Ep = self.EmbforPosition(position)
        EPnew = Ep.squeeze()
        # print(Ep.size())
        Et = self.EmbforThreats(threats)
        # print(Et.size())
        # print(EPnew.size())
        if len(EPnew) == self.hidSizeforPosition:
            EPnew = EPnew.unsqueeze(0)
        #print(EPnew)
        e = torch.cat((EPnew, Et), 1)
        out1 = self.Sample(e)
        #out2 = out1 * 2 * math.pi
        return out1