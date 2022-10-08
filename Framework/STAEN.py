import torch.nn as nn
import torch
import torch.nn.functional as F
'''
功能：提供STARN模型的计算流程
'''
class Attention(nn.Module):
    def __init__(self, use_tanh=False, C=10, name='Dot'):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
    def forward(self, query, ref):
        if self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)
        else:
            raise NotImplementedError
        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return logits
class Relation (nn.Module):
    '''relation embedding'''
    def  __init__(self, input_size, effect_size,nlayers_1, Hidden_size):
        super(Relation, self).__init__()
        self.input_size = input_size
        self.effect_size = effect_size
        self.nlayers_1 = nlayers_1
        self.Hidden_size = Hidden_size
        self.GRU1_1 = nn.GRU(input_size=self.input_size, hidden_size=Hidden_size, num_layers=self.nlayers_1)
        self.out1_1 = torch.nn.Linear(Hidden_size, Hidden_size)
        self.out1_1_1 = torch.nn.Linear(Hidden_size, effect_size)
        self.GRU1_2 = nn.GRU(input_size=self.input_size, hidden_size=Hidden_size, num_layers=self.nlayers_1)
        self.out1_2 = torch.nn.Linear(Hidden_size, Hidden_size)
        self.out1_1_2 = torch.nn.Linear(Hidden_size, effect_size)
    def forward(self, x1, x2):
        '''x1,X2: [N_SAM/N_RADAR, seq_len, batch_size, input_size]'''
        tag = 0
        for i in x1:
            eff, hidden = self.GRU1_1(i)
            effect = self.out1_1(eff)
            effect = F.relu(effect)
            effect = self.out1_1_1(effect)
            if tag == 0:
                effect_n = effect.unsqueeze(0)
                tag += 1
                continue
            effect_n = torch.cat((effect_n, effect.unsqueeze(0)), 0)
        for j in x2:
            eff, hidden= self.GRU1_2(j)
            effect = self.out1_2(eff)
            effect = F.relu(effect)
            effect = self.out1_1_2(effect)
            effect_n = torch.cat((effect_n, effect.unsqueeze(0)), 0)
        return effect_n
class Evaluation(nn.Module):
    def __init__(self, effect_size, output_size, nlayers_2,  Hidden_size):
        super(Evaluation, self).__init__()
        self.hid = 20
        self.attention_1 = Attention(use_tanh=True, C=10, name='Dot')
        self.GRU2_1 = nn.GRU(input_size=effect_size, hidden_size=effect_size, num_layers=nlayers_2,batch_first= True)
        self.GRU2_2 = nn.GRU(input_size=effect_size, hidden_size= self.hid, num_layers=nlayers_2)
        self.GRU2_3 = nn.GRU(input_size=self.hid, hidden_size=effect_size, num_layers=nlayers_2)
        self.GRU2_4 = nn.GRU(input_size=self.hid, hidden_size=1, num_layers=nlayers_2)
        self.out2_1 = torch.nn.Linear(effect_size, Hidden_size)
        self.out2_2 = torch.nn.Linear(Hidden_size, output_size)
    def wei_sum(self, effect,probs):
        '''effect: [batch_size, relation_num, effect_size]'''
        batch_size = effect.size(0)
        relation_num = effect.size(1)
        for i in range(batch_size):
            eff = torch.zeros(effect.size(2))
            for j in range(relation_num):
                eff += effect[i][j] * probs[i][j]
            if i == 0:
                eff_one = eff.unsqueeze(0)
                continue
            eff_one = torch.cat((eff_one,eff.unsqueeze(0)),0)
        return eff_one
    def forward(self, effect_n,hidden = None):
        effect_n = effect_n.permute(1,2,0,3)
        tag_1 = 0
        for i in effect_n:
            _, hidden = self.GRU2_1(i,hidden)
            query_1 = hidden.squeeze(0)
            '''spatial attention'''
            logits_1_one = self.attention_1(query_1, i)
            probs_1 = F.softmax(logits_1_one,dim=1)
            seq_one_effect_1 = self.wei_sum(i,probs_1)
            if tag_1 == 0:
                seq_effect = seq_one_effect_1.unsqueeze(0)
                logits_1 = probs_1.unsqueeze(0)
                tag_1+=1
                continue
            seq_effect = torch.cat((seq_effect,seq_one_effect_1.unsqueeze(0)),0)
            logits_1 = torch.cat((logits_1,probs_1.unsqueeze(0)),0)
        out_1,hidden_2 = self.GRU2_2(seq_effect)
        '''temporal attention'''
        out_2_one, hidden_3 = self.GRU2_4(out_1)
        probs_2 = F.softmax(out_2_one.squeeze(2), dim=0)
        probs_3 = probs_2.unsqueeze(2).expand(probs_2.size(0), probs_2.size(1), self.hid)
        out_2, hidden_4 = self.GRU2_3(out_1 * probs_3)
        out_3 = self.out2_1(hidden_4)
        out_4 = F.dropout(out_3, p=0.2)
        out_5 = F.relu(out_4)
        outs = self.out2_2(out_5)
        return outs,logits_1,probs_2
class Model_layer(nn.Module):
    def __init__(self,input_size,effect_size,output_size,nlayers_1,nlayers_2,Hidden_size = 60):
        super(Model_layer, self).__init__()
        self.Relation = Relation(input_size, effect_size,nlayers_1,  Hidden_size)
        self.Evaluation = Evaluation(effect_size, output_size, nlayers_2, Hidden_size)
    def forward(self, x1,x2):
        effect_n = self.Relation(x1,x2)
        output,logits_1,logits_2 = self.Evaluation(effect_n)
        return output,logits_1,logits_2







