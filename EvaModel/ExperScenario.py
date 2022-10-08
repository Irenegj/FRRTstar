import random
import sys
'''
功能：生成一个场景（生成输入输出需要与仿真环境中的Experiment_Scenario.py保持一致）
'''
class map2D():
    def __init__(self,dimension,iniPosition,goalPostion,randLength,randBias,numRadar,numWeapon,attRadar,attWeapon):
        self.dimension = dimension
        self.iniPosition = iniPosition
        self.goalPostion = goalPostion
        self.randLength = randLength
        self.randBias = randBias
        self.numRadar = numRadar
        self.numWeapon = numWeapon
        self.attRadar = attRadar
        self.attWeapon = attWeapon
        self.radarPosiList = []
        self.weaponPosiList = []
        self.weaponTag = []
    def tag_2_3(self):
        sum = 0
        weaponNum = []
        for j in range(self.numRadar):
            if j == (self.numRadar - 1):
                weaponNum.append(self.numWeapon - sum)
            else:
                rand = random.randint(1, min(3, self.numWeapon - sum - 1))
                weaponNum.append(rand)
                sum += rand
        return weaponNum

    def tag_1_2(self):
        weaponNum = [2]
        return weaponNum
    def tag_1_1(self):
        weaponNum = [1]
        return weaponNum

    def generate(self,radarPosi_X, radarPosi_y,radar,weapon):
        d = self.attRadar[radar][0] - self.attWeapon[weapon][0]
        x = random.randint(max(self.iniPosition[0], (radarPosi_X - d)), (min((radarPosi_X + d), self.goalPostion[0])))
        y = random.randint(max(self.iniPosition[1], (radarPosi_y - d)), (min((radarPosi_y + d), self.goalPostion[1])))
        return [x, y]

def creatMap(dimension,iniPosition,goalPostion,randLength,randBias,numRadar,numWeapon,attRadar,attWeapon,minPosRadar,maxPosRadar):
    if dimension == 2:
        mapEva = map2D(dimension=dimension,iniPosition=iniPosition,goalPostion=goalPostion,randLength=randLength,randBias=randBias,
                      numRadar=numRadar,numWeapon=numWeapon,attRadar=attRadar,attWeapon=attWeapon)
        if numRadar == 2 and numWeapon == 3:
            weaponNum = mapEva.tag_2_3()
        elif numRadar == 1 and numWeapon == 2:
            weaponNum = mapEva.tag_1_2()
        elif numRadar == 1 and numWeapon == 1:
            weaponNum = mapEva.tag_1_1()
        else:
            print("The number of radars and weapons does not meet the standard")
            sys.exit()
        for i in range(numRadar):
            radarPosi_X = random.randint(minPosRadar[i][0], maxPosRadar[i][0])
            radarPosi_y = random.randint(minPosRadar[i][1], maxPosRadar[i][1])
            mapEva.radarPosiList.append([radarPosi_X,radarPosi_y])
            for j in range(weaponNum[i]):
                mapEva.weaponTag.append(i)
                mapEva.weaponPosiList.append(mapEva.generate(radarPosi_X,radarPosi_y,radar=i,weapon=j))
        return mapEva
    else:
        print("The dimension does not meet the standard")
        sys.exit()
def loadMap(dimension,iniPosition,goalPostion,randLength,randBias,numRadar,numWeapon,attRadar,attWeapon,radarPosiList,weaponPosiList,weaponTag):
    if dimension == 2:
        mapEva = map2D(dimension=dimension, iniPosition=iniPosition, goalPostion=goalPostion, randLength=randLength,
                       randBias=randBias,numRadar=numRadar, numWeapon=numWeapon, attRadar=attRadar, attWeapon=attWeapon)
        mapEva.radarPosiList = radarPosiList
        mapEva.weaponPosiList = weaponPosiList
        mapEva.weaponTag = weaponTag
    return mapEva