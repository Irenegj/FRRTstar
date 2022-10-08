
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import numpy as np
import numpy.linalg as lng
import sys
from RRT import StoreFunc
import EvaModel.EvaRouteforFeed as EvaRoute
import EvaModel.EvaRoute as EvaRoute2
import Env.EnvRoute as EnvRoute
import copy
import math
from Sample import SampleModel
sys.setrecursionlimit(3000)

class RRT:
	def __init__(self, mapEva,method,maxIteration,minStep,maxStep,goalProb,nearDis,sceneType,methodofEvaluation,showAnimation,observation,storeData,dimension,stopIteration=None):
		self.cutLabel = True
		self.intervalPoint = 1
		self.disStep = 1
		# 路径长度
		self.length = 15
		# 是否进行展示
		self.showAnimation = showAnimation
		self.observation =  observation
		self.storeData = storeData
		self.sceneType = sceneType
		self.methodofEvaluation = methodofEvaluation
		self.mapEva = mapEva
		# 如果存储数据则生成数据存储对象
		self.store = StoreFunc.storeFunc(self.showAnimation)
		#风险网络的评估模型对象
		self.EvaModel = EvaRoute2.Model(self.mapEva)
		self.reEvaModel = EvaRoute.Model(self.mapEva)
		self.logger = []
		#仿真环境构建的真实路径评估模型对象
		self.Modelfortest = EnvRoute.Model(self.mapEva)
		self.sample = SampleModel.sample(mapEva)

		self.method = method
		self.maxIteration = maxIteration
		self.minStep = minStep
		self.maxStep = maxStep
		self.goalProb = goalProb
		self.stopIteration = stopIteration
		self.nearDis = nearDis
		self.dimension = dimension

		self.initNode = Node(x=self.mapEva.iniPosition,children=[],cost=0)
		self.goalNode = Node(x=self.mapEva.goalPostion,children=[])
		self.trees = []
		self.path = []
		self.pathCost = float('inf')
		self.pathRisk = None
		self.fundNumber = 0

		self.reDecisionNumber = 0

	def Search(self,pathofPicture = None):
		'''
		记录RRT*算法的运行时间，调用和运行RRT*搜索，判断搜索结果，调用画图函数，是RRT*类的上层调用主函数。
		'''
		txt = "method: "+self.method
		if self.storeData:
			self.SaveMap()
		self.logger.append(txt+'\n')
		# Search
		start_time = time.time()
		ret, pathfortest = self.RRTStarSearch()
		#运行RRT*算法

		end_time = time.time()
		testrisk = 100
		if ret:
			self.getPath()
			t, attentionS, sttentionT = self.Modelfortest.evalution(pathfortest)
			testrisk = t
			result = "path cost(distance): "+ str(self.pathCost)+"risk:"+str(self.pathRisk)+"steps: "+str(len(self.path))+" time_use: "+str(end_time - start_time)
		else:
			if self.stopIteration:
				result = "Search %i times, but still no feasible track is found"%(int(self.stopIteration))
			else:
				result = "Search %s times, but still no feasible track is found" % ("None")
		if self.storeData:
			self.SaveTrees()
		self.drawGraph()
		self.drawPath()
		self.SavePicture(pathofPicture)
		return result, self.fundNumber, self.reDecisionNumber, self.pathCost, self.pathRisk, testrisk, (end_time - start_time)

	def RRTStarSearch(self):
		# Based the initial node initialize the initial tree (new a tree)
		tree = Tree(self.initNode)
		# Add the initial tree to the trees
		self.trees.append(tree)
		#Start the iterations in the maximum number of iterations
		tag = 0 #记录生成的轨迹数
		i = 0
		while i < self.maxIteration:
			i+=1
			# Generating a random sampling point by function
			xrand = self.SampleRandom()
			nnearest, dis = tree.getNearest(xrand,self.sceneType)
			tagforJudgeSuccess, nnew = self.Extend(nnearest, xrand)
			# 规划标记的默认值为重规划标记的值，False表示规划失败需要重规划，True则表示规划成功
			if tagforJudgeSuccess == False:
				'''  ___________________________'''
				xrandnew, tree, selectedPosition = self.reDecision(nnew,tree)
				if xrandnew:
					tagforJudgeSuccess, nnew = self.Extend(selectedPosition, xrandnew)
					i += 1
					self.reDecisionNumber += 1
				'''  ___________________________'''
			if tagforJudgeSuccess == True and Node.distancenn(nnew, self.goalNode) != 0:
				tree.addChild(nnew)
				tree.addNode(nnew)
				self.reParent(nnew, tree)
				self.reWire(nnew, tree)
				if self.stopIteration == None:
					if (Node.distancenn(nnew, self.goalNode) < self.maxStep):
						if self.sceneType[0] == "Risk":
							CurrentPath = self.ObtainCurrentPath(nnew, self.goalNode.x)
							risk, attentionS, sttentionT = self.EvaModel.evalution(CurrentPath)
							if risk <= self.sceneType[1]:
								self.goalNode.risk, self.goalNode.attentionS, self.goalNode.sttentionT = risk, attentionS, sttentionT
								tree.addNode(self.goalNode, parent=nnew)
								print("self.goalNode.risk:", self.goalNode.risk)
								print("iter ", i, " find!")
								self.fundNumber = i
								return True, CurrentPath
				else:
					dis = Node.distancenn(nnew, self.goalNode)
					if dis < self.maxStep:
						if nnew.cost + dis < self.goalNode.cost:
							if self.sceneType[0] == "Risk":
								CurrentPath = self.ObtainCurrentPath(nnew, self.goalNode.x)
								risk, attentionS, sttentionT = self.EvaModel.evalution(CurrentPath)
								self.goalNode.risk, self.goalNode.attentionS, self.goalNode.sttentionT = risk, attentionS, sttentionT
							self.goalNode.parent = nnew
							self.goalNode.lcost = dis
							self.goalNode.cost = nnew.cost + dis
							tag += 1
							self.drawTempPath(CurrentPath)
							self.SaveTempPath(CurrentPath,tag)
				if self.showAnimation:
					self.drawGraph(rnd=xrand, new=nnew)
					plt.pause(0.0001)
			if i == self.stopIteration:
				if tag == 0: return False, 0
				else:
					self.goalNode.cost = self.goalNode.parent.cost + self.goalNode.lcost
					tree.addNode(self.goalNode)
					return True, 0
		return False, 0

	def SampleRandom(self):
		r = random.random()
		if r < self.goalProb:
			return self.mapEva.goalPostion
		else:
			ret = self.DirectionRandom()
			return ret

	def DirectionRandom(self):
		ret = []
		for i in range(self.dimension):
			ret.append(random.random() * self.mapEva.randLength[i] + self.mapEva.randBias[i])
		return ret

	def Extend(self, nnearest, xrand):
		step = self.maxStep
		dis = Node.distancenx(nnearest, xrand)
		if dis < step:
			xnew = xrand
		else:
			xnew = np.array(nnearest.x) + step * (np.array(xrand) - np.array(nnearest.x)) / Node.distancenx(nnearest,xrand)
			dis = Node.distancenx(nnearest, xnew)
		CurrentPath = self.ObtainCurrentPath(nnearest,xnew)
		self.drawTempPath(CurrentPath,"yellow")
		tag, risk, spacialAtt, temporalAtt = self.CollisionRisk(CurrentPath,self.sceneType[1])
		if tag == False:
			return False, CurrentPath
		else:
			nnew = Node(xnew, parent=nnearest, lcost=dis, children=[],risk = risk,attentionT=spacialAtt, attentionS=temporalAtt)
			return True, nnew


	def _CollisionPoint(self, x):
		weaponsPosition  = self.mapEva.weaponPosiList
		weaponsAtt = self.mapEva.attWeapon
		weaponsUnmber = self.mapEva.numWeapon
		for i in range(weaponsUnmber):
			if Node.distancexx(x, weaponsPosition[i]) <= weaponsAtt[i][0]:
				return True
		return False

	def _CollisionLine(self, x1, x2):
		dis = Node.distancexx(x1, x2)
		if dis < self.minStep:
			return False
		nums = int(dis / self.minStep)
		direction = (np.array(x2) - np.array(x1)) / Node.distancexx(x1, x2)
		for i in range(nums + 1):
			x = np.add(x1, i *self.minStep * direction)
			if self._CollisionPoint(x): return True
		if self._CollisionPoint(x2): return True
		return False

	def ObtainPath(self, node):
		path = []
		tempNode = node
		while tempNode.parent:
			path.append(tempNode.x)
			tempNode = tempNode.parent
		path.append(tempNode.x)
		return path

	def ObtainCurrentPath(self, node,x):
		path = [x]
		tempNode = node
		while tempNode.parent:
			path.append(tempNode.x)
			tempNode = tempNode.parent
		path.append(tempNode.x)
		return path

	def CollisionRisk(self, path,threshold):
		risk, spacialAtt, temporalAtt = self.EvaModel.evalution(path)
		if risk > threshold:
			return False, risk, spacialAtt, temporalAtt
		return True, risk, spacialAtt, temporalAtt

	'''----------------------重新决策-----------------------'''
	def reDecision(self,path,tree):
		risk, spacialAtt, temporalAtt, finalPath, enemyEntities = self.reEvaModel.evalution(path)
		oneEneAtt, position= self.proInformation(spacialAtt,temporalAtt,finalPath,enemyEntities)
		if self.showAnimation:
			plt.plot(oneEneAtt[0], oneEneAtt[1], "or",label = "oneThreat")
			plt.plot(position[0], position[1], "o", label = "position")
			plt.legend()
			self.SaveTempPicture(str(self.reDecisionNumber)+"front")
			plt.pause(0.001)
		# 依据辅助决策信息修剪决策树
		tree,childrenFinal,currentPath = self.trimforBranchChildren(tree,position,path)
		pathforSample = self.DiscretizePath(currentPath)
		if childrenFinal.switch:
			xnew = self.makeDecision(childrenFinal.x,pathforSample)
			childrenFinal.switch = False
			return xnew, tree, childrenFinal
		else:
			return None, tree, childrenFinal

#路径预处理
	def DiscretizePath(self,OriginalPath):
		OP = OriginalPath[::-1]
		tempPath = self.ApproximatePath(OP)
		if self.cutLabel: newTempPath = self.ClipPath(tempPath)
		else: newTempPath = self.SupplementPath(tempPath)
		FinalPath = self.IntervalPath(newTempPath)
		return FinalPath

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

	def dis(self,x1, x2):
		return lng.norm(np.array(x1) - np.array(x2))

	def SupplementPath(self,OriginalPath):
		OP = OriginalPath
		path = [OP[0]] * self.length * self.intervalPoint
		if len(OP) < self.length * self.intervalPoint:
			path[len(path) - len(OP):] = OP
			return path
		else:
			return OP

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

	def ClipPath(self,OriginalPath):
		OP = OriginalPath
		path = [OP[0]] * self.length * self.intervalPoint
		if len(OP) >= self.length * self.intervalPoint:
			path[:self.length * self.intervalPoint] = OP[len(OP)-(self.length * self.intervalPoint):]
		else:
			path[len(path) - len(OP):] = OP
		return path


	def proInformation(self,spacialAtt,temporalAtt,path,enemyEntities):
		spacialAttList = spacialAtt.detach().numpy().squeeze().tolist()
		temporalAttList = temporalAtt.detach().numpy().squeeze().tolist()
		# 获取注意力机制最大的时间点
		maxTemAtt = self.getMaxTemAtt(temporalAttList)
		tempApaAtt = self.getTempApaAtt(maxTemAtt, spacialAttList)
		oneEneAtt = self.getOneEneAtt(tempApaAtt, enemyEntities)
		oneApaAttStep = self.getApaAttStep_2(tempApaAtt, spacialAttList)
		position = self.getPosition(oneApaAttStep, path)
		"使用全部的敌方实体的空间注意力值，通过实体之间的空间注意力差值来计算点"
		'''________________________为了观测________________________'''
		if self.observation:
			self.saveSampleText(temporalAttList, spacialAttList, maxTemAtt, tempApaAtt, oneEneAtt, oneApaAttStep, position)
		'''________________________观测结束________________________'''
		return oneEneAtt, position

	def getThreatPositions(self,enemyEntities,oneEneAtt):
		threatPositions = enemyEntities[0][0:2]
		threatPositions.extend(oneEneAtt[0:2])
		threatPositions.extend(enemyEntities[0][2:4])
		threatPositions.extend(oneEneAtt[2:4])
		return threatPositions
	def trimforAllChildren(self,tree,position):
		nnearest, dis = tree.getNearest(position, self.sceneType)
		childrenListOne = nnearest.children
		if childrenListOne:
			childrenListBegin = copy.copy(childrenListOne)
			childrenList = self.fundChildren(childrenListOne,childrenListBegin)
		else:
			childrenList = None
		return childrenList,nnearest

	def trimforBranchChildren(self,tree,position,path):
		#如果选到的Nearest点是路径顶点的话，由于该点不在tree中，所以nnearest是None值
		nnearest, dis = tree.getPathNearest(position, path,self.sceneType)
		# 找到选定点的在路径末端的单一方向的孩子节点，并减去所有从该孩子节点延伸的所有节点
		if nnearest == None:
			nnearestNext,dis = tree.getNearest(path[1],self.sceneType)
			currentPath = self.ObtainPath(nnearestNext)
			return tree,nnearestNext, currentPath
		else:
			childrenList, childrenFinal = self.fundOneBranchChildren(tree, path, nnearest)
			if childrenList:
				for i in childrenList:
					tree.nodes.remove(i)
					tree.deleteChild(i)
			if self.showAnimation:
				plt.plot(nnearest.x[0], nnearest.x[1], "ob", label = "nnearest")
				plt.plot(childrenFinal.x[0], childrenFinal.x[1], "oy", label="childrenFinal")
				self.drawTree(tree,'r')
			#nnearest 为选点位置对应的最近选定点 childreFinal是剪裁位置
			currentPath = self.ObtainPath(nnearest)

			return tree, nnearest, currentPath

	def fundAllBranchChildren(self,tree,path,nnearest):
		evaNode = None
		pathList = np.array(path).tolist()
		for node in tree.nodes:
			if node in nnearest.children:
				if (list(node.x) in pathList):
					evaNode = node
					nnearest.children.remove(node)
		if evaNode:
			childrenListOne = [evaNode]
			childrenListBegin = copy.copy(childrenListOne)
			childrenList = self.fundChildren(childrenListOne, childrenListBegin)
		else: childrenList = None
		return childrenList

	def fundOneBranchChildren(self,tree,path,nnearest):
		pathList = np.array(path).tolist()
		childrenList = []
		for x in pathList[1:]:
			node, nn = self.judgeNode(x,tree,nnearest,pathList)
			if nn:
				childrenList.append(node)
			else:
				childrenFinal = node
				break
		return childrenList,childrenFinal

	def judgeNode(self,x,tree,nnearest, pathList):
		tt = 0
		l = []
		for node in tree.nodes:
			l.append(node.x)
			if list(node.x) == x:
				if len(node.children) == 1 and node!=nnearest:
					if list(node.children[0].x) in pathList:
						return node, True
					else:
						return node, False
				elif len(node.children) == 0 and node!=nnearest:
					return node, True
				else:
					return node, False
			else: tt+=1
		if tt == len(tree.nodes):
			print("不存在")


	def fundChildren(self,childrenListOne,childrenList):
		for i in childrenListOne:
			if i.children:
				childrenList.extend(i.children)
				self.fundChildren(i.children,childrenList)
			else:
				childrenList.extend(i.children)
				continue
		return childrenList


	def makeDecision(self,position, pathforSample):
		# 计算下一步采样的点
		xnew = self.sample.sample_3(position,pathforSample,self.maxStep)
		if self.showAnimation:
			plt.plot(xnew[0], xnew[1], "ok", label = "xnew")
			plt.legend()
			self.SaveTempPicture(str(self.reDecisionNumber) + "after")
			plt.pause(0.001)
		return xnew
	def getAngle(self,p1,p2):
		a =  math.atan2(p1[1] - p2[1], p1[0] - p2[0])
		if a < 0:
			b = 2 * math.pi + a
			return b
		else: return a
	def decforAngle(self,angle_b,angle_a): #默认设定为偏转60度
		if angle_a <= angle_b:
			return angle_a - math.radians(15)
		else:
			return angle_a + math.radians(15)

	def getMaxTemAtt(self,temporalAtt): #获取时间注意力最高的值以及其下标
		attentionList = temporalAtt
		maxTemAtt = [attentionList.index(max(attentionList)),max(attentionList)]
		return maxTemAtt

	def getTempApaAtt(self,maxTemAtt, spacialAtt): #获得最大时间注意力值的时间步所对应的空间注意力值
		return spacialAtt[maxTemAtt[0]]

	def getOneEneAtt(self,tempApaAtt, enemyEntities): #获得最大空间注意力值对应的实体的属性
		maxIndex = tempApaAtt.index(max(tempApaAtt))
		return enemyEntities[maxIndex]

	def getApaAttStep_1(self, tempApaAtt, spacialAtt): # 获得对应实体最大值的位置（需要设置阈值）
		# 该方法中需要和其他的进行对比
		maxIndex = tempApaAtt.index(max(tempApaAtt))
		maxAtt = (1/self.mapEva.numWeapon)+0.1 #设置的阈值
		length = len(spacialAtt)
		l = np.linspace(0,length-1,length)[::-1]
		maxStep = None
		for i in l:
			spaList = spacialAtt[int(i)]
			maxApa = max(spaList)
			if maxIndex == spaList.index(maxApa) and maxApa > maxAtt:
				maxStep = i
			else:
				break
		if maxStep == None:
			print("无法找到合适的位置")
		return maxStep

	def getApaAttStep_2(self, tempApaAtt, spacialAtt):
		maxIndex = tempApaAtt.index(max(tempApaAtt))
		attList = np.array(spacialAtt).transpose()
		oneApaAtt = attList[maxIndex]
		l = []
		for i in range(len(oneApaAtt)):
			if i != 0:
				l.append(oneApaAtt[i]-oneApaAtt[i-1])
		if maxIndex < self.mapEva.numRadar:
			index = l.index(min(l))
		else:
			index = l.index(max(l))
		return index

	def getPosition(self,oneApaAttStep,path): #获取被攻击的位置

		return path[int(oneApaAttStep)]

	'''----------------------重决策结束-----------------------'''

	def reParent(self, node, tree):
		# TODO: check node in tree
		if self.sceneType[0] == "Obstacle" :
			nears = tree.getNearby(node,self.nearDis)
			for n in nears:
				if self._CollisionLine(n.x, node.x):
					continue
				newl = Node.distancenn(n, node)
				if n.cost + newl < node.cost and node != n:
					tree.deleteChild(node)
					node.parent = n
					tree.addChild(node)
					node.lcost = newl
					node.cost = n.cost + newl
		elif self.sceneType[0] == "Risk":
			#获得所有比较近的点
			nears = tree.getNearby(node,self.nearDis)
			for n in nears:
				CurrentPath = self.ObtainCurrentPath(n, node.x)
				newl = Node.distancenn(n, node)
				if n.cost + newl < node.cost and node != n:
					tag, risk, spacialAtt, temporalAtt = self.CollisionRisk(CurrentPath, self.sceneType[1])
					if tag == True:
						tree.deleteChild(node)
						node.parent = n
						tree.addChild(node)
						node.lcost = newl
						node.cost = n.cost + newl

	# what if combine the both?
	def reWire(self, node, tree):
		nears = tree.getNearby(node,self.nearDis)
		for n in nears:
			if not all(n.x == np.array([0,0])):
				CurrentPath = self.ObtainCurrentPath(n, node.x)
				a = CurrentPath[1]
				b = CurrentPath[0]
				CurrentPath[0] = a
				CurrentPath[1] = b
				newl = Node.distancenn(n, node)
				if node.cost + newl < n.cost and node != n:
					tag, risk, spacialAtt, temporalAtt = self.CollisionRisk(CurrentPath, self.sceneType[1])
					if tag == True:
						tree.deleteChild(n)
						n.parent = node
						tree.addChild(n)
						n.lcost = newl
						n.cost = node.cost + newl
						tree.fundChildren(n)

	def getPath(self):
		# knowing that no more than 2 trees
		t = self.trees[0]
		n = t.nodes[-1]
		self.pathRisk = n.risk
		sum = 0
		while n.parent:
			self.path.append(n.x)
			sum += n.lcost
			n = n.parent
		self.pathCost = sum


	def SaveTrees(self):
		if self.sceneType[0] == "Obstacle":
			heads = ["Num","Postion","Parent","Lcost","Cost","Children"]
			for temp, tree in enumerate(self.trees):
				nodeList = []
				for index, note in enumerate(tree.nodes):
					childrenPos = []
					for child in note.children:
						childrenPos.append(child.x)
					if note.parent == None:
						nodeList.append([index + 1, note.x, "None", note.lcost, note.cost, childrenPos])
					else:
						nodeList.append([index + 1, note.x, note.parent.x, note.lcost, note.cost, childrenPos])
				self.store.saveTrees(heads, nodeList, temp + 1)
		elif self.sceneType[0] == "Risk":
			heads = ["Num", "Postion", "Parent", "Lcost", "Cost", "risk","Children","attentionS","attentionT"]
			for temp, tree in enumerate(self.trees):
				nodeList = []
				for index, note in enumerate(tree.nodes):
					childrenPos = []
					for child in note.children:
						childrenPos.append(child.x)
					if note.parent == None:
						nodeList.append([index + 1, note.x, "None", note.lcost, note.cost, note.risk, childrenPos, note.attentionS, note.attentionT])
					else:
						nodeList.append([index + 1, note.x, note.parent.x, note.lcost, note.cost, note.risk, childrenPos, note.attentionS, note.attentionT])
				self.store.saveTrees(heads, nodeList, temp + 1)
		else:
			print("Unsupported SceneType, please choose one of:")
			print("Obstacle, Risk, (Risk and the risk threshold)")

	def SaveMap(self):
		mapDict = {}
		heads = ["dimension","iniPosition","goalPostion","randLength","randBias","numRadar","numWeapon","attRadar","attWeapon",
				 "radarPosiList","weaponPosiList","weaponTag"]
		data = [self.mapEva.dimension,self.mapEva.iniPosition,self.mapEva.goalPostion,self.mapEva.randLength,self.mapEva.randBias,
				self.mapEva.numRadar,self.mapEva.numWeapon,self.mapEva.attRadar,self.mapEva.attWeapon,self.mapEva.radarPosiList,
				self.mapEva.weaponPosiList,self.mapEva.weaponTag]
		if len(heads) == len(data):
			for i in range(len(heads)):
				mapDict[heads[i]]=data[i]
			self.store.saveMap(mapDict)
		else:
			print("RRTstarfortwo.py: Please ensure that the number of keys and values is equal")
			sys.exit()

	def SavePicture(self,pathofPicture):
		self.store.savePicture(pathofPicture)

	def SaveTempPicture(self,name):
		self.store.saveTempPicture(name)

	def SaveTempPath(self,path,tag):
		self.store.saveTempPath(path,tag)


	def drawGraph(self, rnd=None, new=None, drawnodes=True):
		numWeapon = self.mapEva.numWeapon
		numRadar = self.mapEva.numRadar
		attWeapon = self.mapEva.attWeapon
		attRadar = self.mapEva.attRadar
		weaponPosiList = self.mapEva.weaponPosiList
		radarPosiList = self.mapEva.radarPosiList
		if self.dimension == 2:
			plt.clf()
			for j in range(numRadar):
				circle = mpathes.Circle(radarPosiList[j],attRadar[j][0])
				circle.set(ec = "black",fc = "b", alpha = 0.3)
				plt.gcf().gca().add_artist(circle)
			for i in range(numWeapon):
				circle = mpathes.Circle(weaponPosiList[i],attWeapon[i][0])
				circle.set(ec="black",fc = "r", alpha = 0.3)
				plt.gcf().gca().add_artist(circle)
			if rnd is not None:
				plt.plot(rnd[0], rnd[1], "^k")
			if new is not None:
				plt.plot(new.x[0], new.x[1], "og")

			if drawnodes:
				self.drawTree()

			plt.plot(self.mapEva.iniPosition[0], self.mapEva.iniPosition[1], "xr")
			plt.plot(self.mapEva.goalPostion[0], self.mapEva.goalPostion[1], "xr")
			plt.axis([self.mapEva.iniPosition[0] + self.mapEva.randBias[0],
					  self.mapEva.iniPosition[0] + self.mapEva.randBias[0] + self.mapEva.randLength[0],
					  self.mapEva.iniPosition[1] + self.mapEva.randBias[1],
					  self.mapEva.iniPosition[1] + self.mapEva.randBias[1] + self.mapEva.randLength[1]])
			plt.grid(True)
		elif self.dimension == 3:
			pass

	def drawTree(self, tree=None, color='g'):
		if tree == None:
			trees = self.trees
		else:
			trees = [tree]
		if self.dimension == 2:
			for t in trees:
				for node in t.nodes:
					if node.parent is not None:
						plt.plot([node.x[0], node.parent.x[0]],
								 [node.x[1], node.parent.x[1]], '-' + color)
		elif self.dimension == 3:
			pass

	def drawPath(self):
		if self.dimension == 2:
			plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], '-r', color ="r")
		elif self.dimension == 3:
			pass

	def drawTempPath(self,TempPath,color="blue"):
		path = np.array(TempPath)
		if self.dimension == 2:
			plt.plot([x for x in path[:,0]], [y for y in path[:,1]], '-', color =color)
		elif self.dimension == 3:
			pass

	def saveSampleText(self,temporalAttList, spacialAttList, maxTemAtt, tempApaAtt, oneEneAtt, oneApaAttStep, position):
		self.store.saveSampleText(temporalAttList, spacialAttList, maxTemAtt, tempApaAtt, oneEneAtt, oneApaAttStep, position)


class Node:
	def __init__(self, x, children, lcost=0.0, cost=float('inf'), parent=None, risk=float('inf'), attentionT=[],
				 attentionS=[], sumPara=[], switch = True):
		self.x = np.array(x)
		self.children = children
		self.lcost = lcost
		self.cost = cost
		self.parent = parent
		self.risk = risk
		self.attentionT = attentionT
		self.attentionS = attentionS
		self.sumPara = sumPara
		self.switch = switch
		if parent:
			self.cost = self.lcost + parent.cost

	@staticmethod
	def distancenn(n1, n2):
		return lng.norm(np.array(n1.x) - np.array(n2.x))

	@staticmethod
	def distancenx(n, x):
		return lng.norm(n.x - np.array(x))

	@staticmethod
	def distancexx(x1, x2):
		return lng.norm(np.array(x1) - np.array(x2))

# TODO
class Tree:
	def __init__(self, nroot):
		self.root = nroot
		self.nodes = [nroot]

	def getNearest(self, x,sceneType):
		dis = float('inf')
		nnearest = None
		if sceneType[0] == "Obstacle" or sceneType[0] == "Risk":
			for node in self.nodes:
				curDis = Node.distancenx(node, x)
				if curDis < dis:
					dis = curDis
					nnearest = node
		return nnearest, dis

	def getPathNearest(self,x,path,sceneType):
		dis = float('inf')
		xnearest = None
		nnearest = None
		if sceneType[0] == "Obstacle" or sceneType[0] == "Risk":
			for p in path:
				curDis = Node.distancexx(p,x)
				if curDis < dis:
					dis = curDis
					xnearest = p
		nxList = []
		for n in self.nodes:
			nxList.append(n.x)
			if (n.x == xnearest ).all():
				nnearest = n
		return nnearest, dis

	def getNearby(self, nto, dis=None):
		ret = []
		for n in self.nodes:
			if Node.distancenn(nto, n) < dis:
				if not all(n.x == nto.x):
					ret.append(n)
		return ret

	def addChild(self, node):
		node.parent.children.append(node)

	def addNode(self, n, parent=None):
		if parent:
			n.parent = parent
			n.cost = n.lcost + parent.cost
		self.nodes.append(n)

	def deleteChild(self, node):
		node.parent.children.remove(node)

	def fundChildren(self, node):
		if node.children != []:
			for child in node.children:
				child.cost = child.parent.cost + child.lcost

