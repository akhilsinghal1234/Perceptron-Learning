import numpy,sys

def arg_max(perceptron,x):
	class_ = []
	for i in range(len(perceptron)):
		k = 0
		for j in range(len(perceptron[i].w)):
			k += x[j] * perceptron[i].w[j]
		if k < 0:
			class_.append(perceptron[i].pair[0])
		else:
			class_.append(perceptron[i].pair[1])
	# print(class_)
	hash_class = numpy.zeros(shape = (1,len(perceptron)))
	for index in class_:
		hash_class[0][index-1] += 1
	# print(hash_class)
	max_vote,index = 0,0
	for i in range(len(perceptron)):
		if hash_class[0][i] > max_vote:
			max_vote = hash_class[0][i]
			index = i
	# print(int(max_vote),index+1)
	return (index+1)

def parameters(testset,perceptron):
	r = len(perceptron)
	conf_mat = numpy.zeros(shape=(r,r))
	error_freq = 0
	for data in testset:
		k = arg_max(perceptron,[1,data[0],data[1]])
		conf_mat[data[2]-1][k-1] += 1
	print("Confusion Matrix: \n",conf_mat)
	accuracy = 0
	for i in range(len(conf_mat)):
		accuracy += conf_mat[i][i]
	accuracy = accuracy/len(testset)

	recall,precision,f_measure = [],[],[]
	r,p,f = 0,0,0
	for i in range(len(conf_mat)):
		s1,s2 = 0,0
		r = conf_mat[i][i]
		p = conf_mat[i][i]
		for j in range(len(conf_mat)):
			s1 += conf_mat[i][j]
			s2 += conf_mat[j][i]
		r = r/s1
		p = p/s2
		f = (2*p*r)/(p+r)
		recall.append(r)
		precision.append(p)
		f_measure.append(f)
	for i in range(len(conf_mat)):
		print("For file",i+1,":\n","Recall: ",round(recall[i],2),", Precision: ",round(precision[i],2),", F-score: ",round(f_measure[i],2),"\n")
	r,p,f = 0,0,0
	for i in range(len(conf_mat)):
		r += recall[i]
		p += precision[i]
		f += f_measure[i]
	r = r/len(recall)
	p = p/len(precision)
	f = f/len(f_measure)
	print("Mean Recall: ",round(r,2)," Precision: ",round(p,2)," F-score: ",round(f,2))
	print("Accuracy: ",accuracy)