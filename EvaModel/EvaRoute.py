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
		# 是否将输入数据正则化，根据不同评估模型的要求来开启和关闭
		self.ifstandardized = False
		# 标准化参数
		self.position1_median_x1 = 0.0
		self.position1_max_x1 = 6.6333
		self.position1_min_x1 = 0.0
		self.position2_median_x1 = 6.0
		self.position2_max_x1 = 7.0
		self.position2_min_x1 = 5.0
		self.range_median_x1 = 5.0
		self.range_max_x1 = 5.0
		self.range_min_x1 = 2.0
		self.position1_median_x2 = 0.0
		self.position1_max_x2 = 6.6333
		self.position1_min_x2 = 0.0
		self.position2_median_x2 = 6.0
		self.position2_max_x2 = 8.0
		self.position2_min_x2 = 3.0
		self.range_median_x2 = 3.0
		self.range_max_x2 = 3.0
		self.range_min_x2 = 1.0

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
		# print(np.shape(x1))
		# print(np.shape(x2))
		if self.ifstandardized:
			new_x1 = self.standardized_x1(x1)
			new_x2 = self.standardized_x2(x2)
			output, logits_1, logits_2 = self.model(torch.Tensor(new_x1), torch.Tensor(new_x2))
		else:
			output, logits_1, logits_2 = self.model(torch.Tensor(x1), torch.Tensor(x2))
		self.tempRisk = float(output)*0.1
		self.tempAttentionS = logits_1
		self.tempAttentionT = logits_2
		return float(output)*0.1,logits_1,logits_2

	def standardized_x1(self,data):
		data[:,:,:, 0:2] = (data[:,:,:, 0:2] - self.position1_median_x1) * (2 / (self.position1_max_x1 - self.position1_min_x1))
		data[:, :,:,2:4] = (data[:,:,:, 2:4] - self.position2_median_x1) * (2 / (self.position2_max_x1 - self.position2_min_x1))
		data[:,:,:, 4:6] = (data[:, :,:,4:6] - self.range_median_x1) * (2 / (self.range_max_x1 - self.range_min_x1))
		# print(data)
		return data

	def standardized_x2(self,data):
		data[:,:,:, 0:2] = (data[:,:,:, 0:2] - self.position1_median_x2) * (2 / (self.position1_max_x2 - self.position1_min_x2))
		data[:,:,:, 2:4] = (data[:,:,:, 2:4] - self.position2_median_x2) * (2 / (self.position2_max_x2 - self.position2_min_x2))
		data[:,:,:, 4:6] = (data[:,:,:, 4:6] - self.range_median_x2) * (2 / (self.range_max_x2 - self.range_min_x2))
		# print(data)
		return data

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

	def DiscretizePath(self,OriginalPath):
		# 将原始轨迹颠倒
		OP = OriginalPath[::-1]
		# 生成近似轨迹
		tempPath = self.ApproximatePath(OP)
		# 将轨迹进行切割
		if self.cutLabel: newTempPath = self.ClipPath(tempPath)
		else: newTempPath = self.SupplementPath(tempPath)
		# 生成最中的评估轨迹
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

	# 将长轨迹切割
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

