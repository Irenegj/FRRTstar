class config():
	def __init__(self):

		'''parameters of scene'''

		# 是否展示算法中的特定内容
		self.showAnimation = False  #展示开启（开启规划过程的展示）
		# 是否存储算法中生成的相关数据，包括图像数据
		self.storeData = False   #存储开启(存储内容中包括图像数据，所以在开启存储的时候必须开启展示)
		# 是否对评估模型的内部流程进行观察
		self.observation = False
		# 场景尺寸
		self.size = 10

		'''parameters of search'''
		self.algorithm = "FRRT*"
		# the maximum number of iteration is max greater than the stopIteration
		self.maxIteration = 400
		# None or a number
		self.stopIteration = None
		# the minimum size of step and max size of step
		self.minStep = 0.1
		# the probability of selecting target point
		self.goalProb = 0.1
		self.sceneType = ["Risk", 0.4]
		self.methodofEvaluation = "EvaluationModel"
		# 用于size为10的场景
		self.maxStep = 1
		self.nearDis = 2
		#None或者已经存储的map
		self.exiMap = 'Env\Maps\\10forshow-1'
		# 运行时间基准上线
		self.uptime = 50
		# the dimension of map
		self.dimension = 2
		# the initial and goal postion
		self.iniPosition = [0, 0]
		self.goalPostion = [10, 10]
		# the length and bias of rand
		self.randLength = [11, 11]
		self.randBias = [-1, -1]
		# the numbers of radars and weapons
		self.numRadar = 1
		self.numWeapon = 2
		# the attributes of radar and weapon
		self.attWeapon = [[3, 1], [3, 1]]  # the max range and kill range
		self.attRadar = [[5, 2]]  # the max range and the detection time
		# the min and max position of radar
		self.minPosRadar = [[5, 5]]  # 为了保障出生点在雷达的外部，所以x,y的最小值均为4（也就是雷达半径的值）
		self.maxPosRadar = [[7, 7]]
		# 存储的风险网络模型
		self.exiModel = "Framework\STAEN 135233"
		# the size of step (离散RRT生成轨迹的步长)
		self.disStep = 1
		# 截取的轨迹点间隔
		self.intervalPoint = 1
		# the number of step
		self.routeLength = 20
		# 输入的size
		self.inputSize = 6
		# Whether to cut the track True or False
		self.cutLabel = True




