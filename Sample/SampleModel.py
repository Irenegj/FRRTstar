
import os
import torch
import math


#DNNSample 095332
class sample:
	def __init__(self, mapEva):
		self.modelName = "Sample 155256"
		self.mapEva = mapEva
		with open(os.path.join(os.getcwd(), "Sample", self.modelName, "model.pt"), 'rb') as f:
			self.model = torch.load(f)
		threats = str(self.mapEva.radarPosiList)+str(self.mapEva.weaponPosiList)+str(self.mapEva.attRadar)+str(self.mapEva.attWeapon)
		self.inputofThreats = torch.Tensor(self.getList(threats))
	def getList(self,l1):
		b = l1.replace("]][[", ",").replace("]", "").replace("[", "").split(",")
		l = []
		for i in b:
			l.append(float(i))
		return l
	def sample_2(self,Position,pathforSample, threatPositions, maxStep):
		self.model.eval()
		with torch.no_grad():
			output = self.model(torch.Tensor([pathforSample]), torch.Tensor([threatPositions]))
		angle = output.tolist()
		pnew = [Position[0] + maxStep * math.cos(angle[0][0]), Position[1] + maxStep * math.sin(angle[0][0])]
		return pnew

	def sample_3(self,Position,pathforSample, maxStep):
		self.model.eval()
		with torch.no_grad():
			output = self.model(torch.Tensor([pathforSample]), self.inputofThreats.view(1,-1))
		angle = output.tolist()
		pnew = [Position[0] + maxStep * math.cos(angle[0][0]), Position[1] + maxStep * math.sin(angle[0][0])]
		return pnew

	def sample(self,Position,maxStep):
		inputofPosition = torch.Tensor(Position).view(1,-1)
		self.model.eval()
		with torch.no_grad():
			output = self.model(inputofPosition, self.inputofThreats.view(1,-1))
		angle = output.tolist()
		pnew = [Position[0] + maxStep * math.cos(angle[0][0]), Position[1] + maxStep * math.sin(angle[0][0])]
		return pnew