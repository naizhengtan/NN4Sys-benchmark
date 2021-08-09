using NeuralVerification, LazySets
import NeuralVerification: ReLU, Id
# Verifying increasing send ratio- delivered : sent packets ratio
# Constant sent latency and latency
# Should expect an increase in rate of sending. 
lowerInput = [10.0, 35.0, 0.0,
       10.0, 35.0, 6.0,
       10.0, 35.0, 16.0,
       10.0, 35.0, 22.0,
       10.0, 35.0, 24.0,
       10.0, 35.0, 30.0,
       10.0, 35.0, 36.0,
       10.0, 35.0, 42.0,
       10.0, 35.0, 52.0,
       10.0, 35.0, 58.0]

upperInput = [10.0, 35.0, 5.0,
       10.0, 35.0, 11.0,
       10.0, 35.0, 21.0,
       10.0, 35.0, 23.0,
       10.0, 35.0, 29.0,
       10.0, 35.0, 35.0,
       10.0, 35.0, 41.0,
       10.0, 35.0, 47.0,
       10.0, 35.0, 57.0,
       10.0, 35.0, 63.0]

# Run with different values of ranges.
# Check if NN satisfies human intuition of specifications. 
# Does it work when we experience congestion? 
# Find example of when NN does not follow our thoughts(corner cases). Find examples
# when they do...

x = Hyperrectangle(low = lowerInput, high = upperInput)
y = Hyperrectangle(low = [-1000], high = [-0.001])

model = "/Users/Chioma_N/Desktop/object/PCC-RL-master/src/gym/tensorflow/reLuNet/resNet"


net = read_nnet(model)
prob = Problem(net, x, y)
res = solve(ReluVal(max_iter=100), prob)
NeuralVerification.compute_output(net, res.counter_example)