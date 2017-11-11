import numpy,sys

def arg_max(perceptron,x):
	response,g,p,k = [],[],[],0
	response.append(g)	# g(x)
	response.append(p)	# class label

	for i in range(len(perceptron)):
		k = 0
		for j in range(len(perceptron[i].w)):
			k += x[j] * perceptron[i].w[j]
		response[0].append(k)
		response[1].append(perceptron[i].file_index)

	min_g = sys.maxsize
	for i in range(len(perceptron)):
		if response[0][i] < min_g:
			min_g = response[0][i]
			index = response[1][i]
	return index

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