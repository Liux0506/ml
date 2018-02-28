#1、计算给定数据的香农熵
def calcShannonEnt(dataSet):
	#计算给定数据集的香农熵
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet：
		currentLable = featVec[-1]
		if currentLable not in labelCounts.keys():
			labelCounts[currentLable] = 0
		labelCounts[currentLable] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob*log(prob,2)  #计算log
	return shannonEnt

#2、创建数据的函数
def createDataSet():
	dataSet = [[1,1,'yes'],
			   [1,1,'yes'],
			   [1,0,'no'],
			   [0,1,'no'],
			   [0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels

#3、划分数据集，按照给定的特征划分数据集
#输入数据集dataSet里，筛选第axis个特征值且值为value的数据
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#4、选择最好的数据集划分方式，选出信息增益最大的特征
def chooseBestFeatureToSplit(dataSet)
	numFeatures = len(dataSet[0])-1	#len(dataSet[0])为特征值个数 2
	baseEntropy=calcShannonEnt(dataSet) 
	bestInfoGain=0.0; bestFeature = -1
	for i in range(numFeatures):	#i为特征
		featList = [example[i] for example in dataSet] #挨个取dataSet中特征
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:	#value为特征i的可能的取值
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob*calcShannonEnt(subDataSet)
		infoGain =bestInfoGain-newEntropy
		if(infoGain>bestInfoGain):
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature
		
