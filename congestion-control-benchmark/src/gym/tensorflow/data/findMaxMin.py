file = open('./outputs.txt', 'r')
lines = file.readlines()

minVal = float('inf')
maxVal = -1 * float('inf')
for line in lines: 
	if float(line[1:-2]) > float(maxVal): 
		maxVal = line[1:-2]
	if float(line[1:-2]) < float(minVal): 
		minVal = line[1:-2]

print (maxVal) #== 3.6786385
print (minVal) #== -3.3128161


Have 
Input = 1 x 30 

W1 = 30 x 16 
Bias = 1 x 16 

W2 = 16 x 16
bias = 1 x 16

W3 = 16 x 1
bias = 1










# sent latency inflation, latency ratio, send ratio
lowerInput = [[-1.0], [1.0], [0.0], 
			  [-1.0], [1.0], [0.0],
			  [-1.0], [1.0], [0.0],
			  [-1.0], [1.0], [0.0],
 			  [-1.0], [1.0], [0.0], 
 			  [-1.0], [1.0], [0.0], 
 			  [-1.0], [1.0], [0.0],
 			  [-1.0], [1.0], [0.0],
  			  [-1.0], [1.0], [0.0],
  			  [-1.0], [1.0], [0.0]]

# smaller the range, the more accurate the verification
# reLu verifier - ReLuVal
# max_iter for reluVal set to 100 first, the larger the longer time to solve but more accurate result 
upperInput = [[10.0], [100.0], [10.0], 
			  [10.0], [100.0], [10.0],
			  [10.0], [100.0], [10.0],
			  [10.0], [100.0], [10.0],
 			  [10.0], [100.0], [10.0], 
 			  [10.0], [100.0], [10.0], 
 			  [10.0], [100.0], [10.0],
 			  [10.0], [100.0], [10.0],
  			  [10.0], [100.0], [10.0],
  			  [10.0], [100.0], [10.0]]




lowerInput = [-1.0, 1.0, 0.0, 
			  -1.0, 1.0, 0.0,
			  -1.0, 1.0, 0.0,
			  -1.0, 1.0, 0.0,
 			  -1.0, 1.0, 0.0, 
 			  -1.0, 1.0, 0.0, 
 			  -1.0, 1.0, 0.0,
 			  -1.0, 1.0, 0.0,
  			  -1.0, 1.0, 0.0,
  			  -1.0, 1.0, 0.0]

# smaller the range, the more accurate the verification
# reLu verifier - ReLuVal
# max_iter for reluVal set to 100 first, the larger the longer time to solve but more accurate result 
upperInput = [10.0, 100.0, 10.0, 
			  10.0, 100.0, 10.0,
			  10.0, 100.0, 10.0,
			  10.0, 100.0, 10.0,
 			  10.0, 100.0, 10.0, 
 			  10.0, 100.0, 10.0, 
 			  10.0, 100.0, 10.0,
 			  10.0, 100.0, 10.0,
  			  10.0, 100.0, 10.0,
  			  10.0, 100.0, 10.0]

upperInput = [10.0, 10000.0, 1000.0, 
			  10.0, 10000.0, 1000.0,
			  10.0, 10000.0, 1000.0,
			  10.0, 10000.0, 1000.0,
 			  10.0, 10000.0, 1000.0, 
 			  10.0, 10000.0, 1000.0, 
 			  10.0, 10000.0, 1000.0,
 			  10.0, 10000.0, 1000.0,
  			  10.0, 10000.0, 1000.0,
  			  10.0, 10000.0, 1000.0]