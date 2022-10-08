import Config
import numpy as np
import numpy.linalg as lng
import os
import torch
'''
功能：将规划算法生成的轨迹转化为可以评估的轨迹，并调用评估模型完成轨迹评估
'''
class Model():
	def __init__(self,evaMap):
		self.con = Config.config()
		self.nameModel = self.con.exiModel
		self.disStep = self.con.disStep
		self.intervalPoint = self.con.intervalPoint
		self.length = self.con.routeLength
		self.numWeapon = self.con.numWeapon
		self.numRadar = self.con.numRadar
		self.attWeapon = self.con.attWeapon
		self.attRadar = self.con.attRadar
		self.inputSize = self.con.inputSize
		self.cutLabel = self.con.cutLabel
		self.radarPosiList = evaMap.radarPosiList
		self.weaponPosiList = evaMap.weaponPosiList

		self.tempPath = []
		self.tempRisk = []
		self.tempAttentionT = []
		self.tempAttentionS = []

		p = os.getcwd()
		modelPath = os.path.join(p, self.con.exiModel, 'model.pt')
		self.model = torch.load(modelPath)

	def evalution(self,path):
		finalPath = self.DiscretizePath(path)
		'''_______________________________为了观测__________________________________'''
		self.tempPath = finalPath
		'''_______________________________结束观测__________________________________'''
		x1,x2 = self.generateData(finalPath)
		enemyEntities = self.generateEnemy()
		output, logits_1, logits_2 = self.model(torch.Tensor(x1), torch.Tensor(x2))
		self.tempRisk = float(output)*0.1
		self.tempAttentionS = logits_1
		self.tempAttentionT = logits_2
		return float(output)*0.1,logits_1,logits_2,finalPath,enemyEntities

	def generateData(self,path):
		p = path
		tempx1 = []
		tempx2 = []
		for i in range(len(p)):
			tempWeapon = []
			tempRadar = []
			for w in range(self.numWeapon):
				temp = []
				temp.extend(p[i])
				temp.extend(self.weaponPosiList[w])
				temp.extend(self.attWeapon[w])
				tempWeapon.append(temp)
			tempx1.append(tempWeapon)
			for r in range(self.numRadar):
				temp = []
				temp.extend(p[i])
				temp.extend(self.radarPosiList[r])
				temp.extend(self.attRadar[r])
				tempRadar.append(temp)
			tempx2.append(tempRadar)
		x2 = np.array(tempx1).transpose(1, 0, 2).reshape(self.numWeapon, len(p), 1, self.inputSize)
		x1 = np.array(tempx2).transpose(1, 0, 2).reshape(self.numRadar, len(p), 1, self.inputSize)
		return x1, x2

	def generateEnemy(self):
		enemyEntities  = []
		for r in range(self.numRadar):
			temp = []
			temp.extend(self.radarPosiList[r])
			temp.extend(self.attRadar[r])
			enemyEntities.append(temp)
		for w in range(self.numWeapon):
			temp = []
			temp.extend(self.weaponPosiList[w])
			temp.extend(self.attWeapon[w])
			enemyEntities.append(temp)
		return enemyEntities

	def DiscretizePath(self,OriginalPath):
		#将轨迹颠倒
		OP = OriginalPath[::-1]
		#生成近似轨迹
		tempPath = self.ApproximatePath(OP)
		#长轨迹切割
		if self.cutLabel: newTempPath = self.ClipPath(tempPath)
		else: newTempPath = self.SupplementPath(tempPath)
		#生成最终的轨迹
		FinalPath = self.IntervalPath(newTempPath)
		return FinalPath

	def IntervalPath(self,OriginalPath):
		OP = OriginalPath
		path = []
		tempNum = 1
		for position in OP:
			if tempNum == self.intervalPoint:
				path.append(position)
				tempNum = 1
			else:tempNum += 1
		return path

	def ApproximatePath(self,OriginalPath):
		OP = OriginalPath
		temp = OriginalPath[0]
		path = [list(temp)]
		for position in OP[1:]:
			tempDis = self.dis(temp, position)
			while tempDis > self.disStep:
				direction = (np.array(position) - np.array(temp)) / self.dis(temp, position)
				x = np.add(temp, self.disStep * direction)
				path.append(list(x))
				tempDis -= self.dis(x, temp)
				temp = x
			if list(position) == list(OP[-1]):
				if tempDis > (self.disStep / 2):
					direction = (np.array(position) - np.array(temp)) / self.dis(temp, position)
					x = np.add(temp, self.disStep * direction)
					path.append(list(x))
		return path

	def ClipPath(self,OriginalPath):
		OP = OriginalPath
		path = [OP[0]] * self.length * self.intervalPoint
		if len(OP) >= self.length * self.intervalPoint:
			path[:self.length * self.intervalPoint] = OP[len(OP)-(self.length * self.intervalPoint):]
		else:
			path[len(path) - len(OP):] = OP
		return path

	def SupplementPath(self,OriginalPath):
		OP = OriginalPath
		path = [OP[0]] * self.length * self.intervalPoint
		if len(OP) < self.length * self.intervalPoint:
			path[len(path) - len(OP):] = OP
			return path
		else:
			return OP

	def dis(self,x1, x2):
		return lng.norm(np.array(x1) - np.array(x2))

	def transform(self,list):
		x = []
		y = []
		for i in range(len(list)):
			x.append(list[i][0])
			y.append(list[i][1])
		return x, y

