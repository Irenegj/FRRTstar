import datetime
import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.setrecursionlimit(3000000)
'''
功能：完成数据存储为.csv文件、.txt文件，主要是数据记录以及实验日志记录。
'''
class storeFunc():
	def __init__(self,showAnimation):
		currentTime = str(datetime.datetime.now())[:-7].replace(':','').replace('-','')[2:]
		self.mapFile = None
		self.searchFile = None
		self.recordFile = None
		self.folder = "Record"
		self.folderChild = "RRT"
		self.mapFileName = "mapDict.json"
		self.treeFileName = "Tree.csv"
		self.searchFileName = "Search.txt"
		self.pictureFileName = "result.jpg"
		self.graphFileName = "graph.jpg"
		self.tempPathFileName = "tempPath.txt"

		self.observationFileName = "observation.txt"
		self.evaluationFileName = "evaluation.txt"

		self.path = os.path.join(os.getcwd(),self.folder,self.folderChild,currentTime)
		if showAnimation:
			os.mkdir(self.path)
	def recordTxt(self,text):
		file = open(os.path.join(self.path, self.searchFileName), 'w+')
		file.writelines(text)
		file.close()
	def saveTrees(self,heads,nodeList,temp):
		file = open(os.path.join(self.path, str(temp)+self.treeFileName), 'w+')
		writer = csv.writer(file)
		writer.writerow(heads)
		writer.writerows(nodeList)
		file.close
	def saveMap(self,mapDict):
		jsObj = json.dumps(mapDict, indent=4)  # indent参数是换行和缩进
		fileObject = open(os.path.join(self.path, self.mapFileName), 'w')
		fileObject.write(jsObj)
		fileObject.close()
	def savePicture(self,pathofPicture):
		if pathofPicture:
			plt.savefig(pathofPicture)
		else:
			plt.savefig(os.path.join(self.path, self.pictureFileName))
	def saveTempPath(self,path,tag):
		file = open(os.path.join(self.path, self.tempPathFileName),'a')
		file.write(str(tag)+"\n")
		file.write(str(path)+"\n")
		file.close
	def saveTempPicture(self,name):
		plt.savefig(os.path.join(self.path, name))

	'''_____________________________为了观测_____________________________'''
	def saveGraph(self,i):
		plt.savefig(os.path.join(self.path, str(i)+self.graphFileName))
		return False

	def saveOBText(self,i, nnew,OBPath,EvaPath, risk, attentionS, attentionT,iList = None,tempEnvRisk = None):
		file = open(os.path.join(self.path,self.observationFileName), 'a')
		ProOBPath = self.ProOBPath(OBPath)
		ProOBPath.reverse()
		file.write("-"*15+"step:"+str(i)+"-"*15+"\n")
		file.write("--nnew:"+str(np.round(nnew,4))+"\n")
		file.write("--OBPath:" +str(ProOBPath)+ "\n")
		file.write("--EvaPath:" + "\n")
		file.write(str(np.round(EvaPath,4)) + "\n")
		file.write("--risk:" +str(risk) + "\n")
		if tempEnvRisk != None:
			file.write("--Envrisk:" +str(tempEnvRisk) + "\n")
			file.write("--Index:"+str(iList)+"\n")
		ProAttentionS = self.ProAttentionS(attentionS)
		file.write("--attentionS:" + "\n")
		for i in ProAttentionS:
			file.write(str(i) + "\n")
		ProAttentionT = self.ProAttentionT(attentionT)
		file.write("--attentionT:" + "\n")
		for i in ProAttentionT:
			file.write(str(i) + "\n")
		file.close

	def saveSampleText(self,path,temporalAttList, spacialAttList, maxTemAtt, tempApaAtt, oneEneAtt, oneApaAttStep, position):
		file = open(os.path.join(self.path,self.observationFileName), 'a')
		file.write("-"*15+"开始处理注意力信息"+"-"*15+"\n")
		file.write("当前路径为：" + "\n")
		file.write("  " + str(np.round(path, 4)) + "\n")
		file.write("时间注意力信息为："+"\n")
		file.write("  "+str(np.round(temporalAttList,4))+"\n")
		file.write("空间注意力信息为：" + "\n")
		file.write("  "+str(np.round(spacialAttList,4))+"\n")
		file.write("时间注意力信息最大的时间点：" + "\n")
		file.write("  " + str(np.round(maxTemAtt, 4)) + "\n")
		file.write("时间注意力信息最大的时间点对应的空间注意力：" + "\n")
		file.write("  " + str(np.round(tempApaAtt, 4)) + "\n")
		file.write("时间实体信息：" + "\n")
		file.write("  " + str(np.round(oneEneAtt, 4)) + "\n")
		file.write("梯度最高位置（拐点）：" + "\n")
		file.write("  " + str(np.round(oneApaAttStep, 4)) + "\n")
		file.write("决定的位置：" + "\n")
		file.write("  " + str(np.round(position, 4)) + "\n")
		file.close

	def ProOBPath(self,OBPath):
		ProOBPath = []
		for i in OBPath:
			ProOBPath.append(list(np.round(i,4)))
		return ProOBPath

	def ProAttentionS(self,attention):
		ndarray = np.round(attention.detach().numpy(), 4)
		ProAttention = []
		for i in ndarray:
			ProAttentionOne = []
			for j in i :
				ProAttentionOne.append(list(j))
			ProAttention.append(ProAttentionOne)
		return ProAttention

	def ProAttentionT(self,attention):
		ProAttention = []
		for i in attention:
			ProAttention.append(round(float(i),4))
		return ProAttention

	'''_____________________________观测结束_____________________________'''

	'''_____________________________为了评估测试_____________________________'''

	def saveEvaTest(self,riskList1,riskList2,iList):
		print(len(riskList1))
		file = open(os.path.join(self.path, self.evaluationFileName), 'a')
		mseVal = self.mse(np.array(riskList1),np.array(riskList2))
		MaxDiff = abs(np.array(riskList1) - np.array(riskList2))
		MaxDiffSort = self.getSort(MaxDiff)
		MaxDiffSortIndex = self.getSortIndex(MaxDiff)
		iListSort = self.getISort(MaxDiffSortIndex,iList)
		file.write("mseVal:" + "\n")
		file.write(str(mseVal) + "\n")
		file.write("MaxDiffSort:" + "\n")
		file.write(str(np.round(MaxDiffSort,4)) + "\n")
		print(np.round(MaxDiffSort,4))
		file.write("iListSort:" + "\n")
		file.write(str(iListSort) + "\n")
		Evaluationsort1 = self.getISort(MaxDiffSortIndex, (riskList1))
		Evaluationsort2 = self.getISort(MaxDiffSortIndex, (riskList2))
		file.write("Evaluationsort:" + "\n")
		file.write(str(np.round(Evaluationsort1,4)) + "\n")
		file.write("Truesort:" + "\n")
		file.write(str(np.round(Evaluationsort2,4)) + "\n")
		file.write("Evaluation:" + "\n")
		file.write(str(np.round(riskList1,4)) + "\n")
		file.write("True:" + "\n")
		file.write(str(np.round(riskList2,4)) + "\n")
		file.close

	def mse(self,predictions,targets):
		return ((predictions-targets)**2).mean()

	def getSort(self,path):
		MaxDiff = list(path)
		MaxDiff.sort(reverse=True)
		return MaxDiff

	def getSortIndex(self,path):
		MaxDiffIndex = np.argsort(-path)
		return MaxDiffIndex

	def getISort(self,Index,path):
		iList = []
		for i in Index:
			iList.append(path[i])
		return iList
	'''_____________________________评估测试结束_____________________________'''
