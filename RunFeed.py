import Config
import EvaModel.ExperScenario as Exper
import matplotlib.pyplot as plt
from RRT import FeedRRTstar
import json
import datetime
import os
import csv
def main(map):
	print("Start rrt planning")

	path,pathofPicture = getPath()
	number = 1
	file1 = open(os.path.join(path, "guocheng.csv"), 'w+')
	file2 = open(os.path.join(path, "zongjie.txt"), 'a')
	writer1 = csv.writer(file1)
	heads = ["Num","fundNumber","pathCost","pathRisk"]
	writer1.writerow(heads)

	con = Config.config()
	if con.exiMap == None:
		# create map
		mapEva = Exper.creatMap(dimension=con.dimension,iniPosition=con.iniPosition,goalPostion=con.goalPostion,randLength=con.randLength,
								randBias=con.randBias,numRadar=con.numRadar,numWeapon=con.numWeapon,attRadar=con.attRadar,
								attWeapon=con.attWeapon,minPosRadar=con.minPosRadar,maxPosRadar=con.maxPosRadar)
	else: mapEva = loadMap(map)

	sumfundNumber = 0
	sumpathCost = 0
	sumpathRisk = 0
	sumtestRisk = 0
	numofsuccess = 0
	numoffail = 0
	sumTime = 0
	sumReDecisionNumber = 0

	for i in range(number):
		rrt = FeedRRTstar.RRT(mapEva=mapEva, method = con.algorithm, maxIteration=con.maxIteration, stopIteration=con.stopIteration, minStep=con.minStep,
						maxStep=con.maxStep, goalProb=con.goalProb, sceneType=con.sceneType, methodofEvaluation = con.methodofEvaluation,
						showAnimation=con.showAnimation, observation = con.observation, storeData=con.storeData, nearDis = con.nearDis, dimension = con.dimension)
		if con.showAnimation:
			rrt.drawGraph()
			plt.pause(0.01)
			input("Input any key to start")
		pictureFileName = str(i)+"result.jpg"
		pathPicture =  os.path.join(pathofPicture,pictureFileName)
		result, fundNumber, reDecisionNumber, pathCost, pathRisk, testrisk,  time = rrt.Search(pathPicture)
		file3 = open(os.path.join(path, "guocheng.txt"), 'a+')
		file3.write("****************************" + "\n")
		file3.write("InterNumber: " + str(i) + "\n")
		if pathRisk==None:
			numoffail += 1
			sumTime += time
			file3.write("result: " + "fail" + "\n")
			file3.write("time:"+str(time) + "\n")
			file3.write("sumtTime:" + str(sumTime) + "\n")
			file3.close()
		else:
			numofsuccess+=1
			sumfundNumber += fundNumber
			sumpathCost += pathCost
			sumpathRisk += pathRisk
			sumtestRisk += testrisk
			sumTime += time
			sumReDecisionNumber += reDecisionNumber
			file3.write("result: " + "success" + "\n")
			file3.write("fundNumber:" + str(fundNumber) + "\n")
			file3.write("sumfundNumber:" + str(sumfundNumber) + "\n")
			file3.write("pathCost:" + str(pathCost) + "\n")
			file3.write("sumpathCost:" + str(sumpathCost) + "\n")
			file3.write("pathRisk:" + str(pathRisk) + "\n")
			file3.write("sumpathRisk:" + str(sumpathRisk) + "\n")
			file3.write("testrisk:" + str(testrisk) + "\n")
			file3.write("sumtestRisk:" + str(sumtestRisk) + "\n")
			file3.write("reDecisionNumber:" + str(reDecisionNumber) + "\n")
			file3.write("sumReDecisionNumber:" + str(sumReDecisionNumber) + "\n")
			file3.write("time:" + str(time) + "\n")
			file3.write("sumtTime:" + str(sumTime) + "\n")
			file3.close()
		writer1.writerow([i,fundNumber, pathCost, pathRisk])
	file2.write("Number: "+str(number)+"\n")
	file2.write("numofsuccess: "+str(numofsuccess)+"\n")
	file2.write("numoffail: " + str(numoffail) + "\n")
	file2.write("sumtime: "+str(sumTime)+"\n")
	file2.write("avetime: " + str(round((sumTime / number),3)) + "\n")
	if numofsuccess != 0:
		file2.write("sumfundNumber: "+str(sumfundNumber)+"\n")
		file2.write("avefundNumber: "+str(round((sumfundNumber / numofsuccess),3))+"\n")
		file2.write("sumpathCost: "+str(sumpathCost)+"\n")
		file2.write("avepathCost: "+str(round((sumpathCost / numofsuccess),3))+ "\n")
		file2.write("sumpathRisk: "+str(sumpathRisk)+"\n")
		file2.write("avepathRisk: "+str(round((sumpathRisk / numofsuccess),3))+ "\n")
		file2.write("sumReDecisionNumber: " + str(sumReDecisionNumber) + "\n")
		file2.write("aveReDecisionNumber: " + str(round((sumReDecisionNumber / numofsuccess), 3)) + "\n")
		file2.write("sumtestRisk: " + str(sumtestRisk) + "\n")
		file2.write("avetestRisk: " + str(round((sumtestRisk / numofsuccess), 3)) + "\n")
	else:
		file2.write("all fail!!!!!")
	print("Finished")

def getPath():
	currentTime = str(datetime.datetime.now())[:-7].replace(':', '').replace('-', '')[2:]
	path = os.path.join(os.getcwd(), "Record", "test", currentTime)
	os.mkdir(path)
	pathofPicture = os.path.join(os.getcwd(), "Record", "test", currentTime,"pic")
	os.mkdir(pathofPicture)
	return path,pathofPicture

def loadMap(file):
	with open(file+'\mapDict.json') as f:
		dict = json.load(f)
	mapEva = Exper.loadMap(dimension=dict["dimension"],iniPosition=dict["iniPosition"],goalPostion=dict["goalPostion"],randLength=dict["randLength"],
								randBias=dict["randBias"],numRadar=dict["numRadar"],numWeapon=dict["numWeapon"],attRadar=dict["attRadar"],
								attWeapon=dict["attWeapon"],radarPosiList=dict["radarPosiList"],weaponPosiList=dict["weaponPosiList"],
						   weaponTag=dict["weaponTag"])
	return mapEva

if __name__ == '__main__':
	main('Env\Maps\\10-1')