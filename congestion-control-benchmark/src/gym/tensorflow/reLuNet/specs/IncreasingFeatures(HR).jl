using NeuralVerification, LazySets
import NeuralVerification: ReLU, Id
# Verifying increasing latency (delay) and increasing 
# sent latency. 
# Should expect a reduction in rate of sending. 
lowerInput = [1.0, 3.0, 0.0,
       5.0, 7.0, 6.0,
       9.0, 11.0, 16.0,
       13.0, 15.0, 22.0,
       17.0, 19.0, 24.0,
       21.0, 23.0, 30.0,
       25.0, 27.0, 36.0,
       29.0, 31.0, 42.0,
       34.0, 36.0, 52.0,
       38.0, 40.0, 58.0]

upperInput = [4.0, 6.0, 5.0,
       8.0, 10.0, 11.0,
       12.0, 14.0, 21.0,
       16.0, 18.0, 23.0,
       20.0, 22.0, 29.0,
       24.0, 26.0, 35.0,
       28.0, 30.0, 41.0,
       32.0, 34.0, 47.0,
       37.0, 39.0, 57.0,
       41.0, 43.0, 63.0]

# Run with different values of ranges.
# Check if NN satisfies human intuition of specifications. 
# Does it work when we experience congestion? 
# Find example of when NN does not follow our thoughts(corner cases). Find examples
# when they do...

x = Hyperrectangle(low = lowerInput, high = upperInput)
y = Hyperrectangle(low = [-10000], high = [-0.001])

model = "/Users/Chioma_N/Desktop/object/PCC-RL-master/src/gym/tensorflow/reLuNet/resNet"


net = read_nnet(model)
prob = Problem(net, x, y)
res = solve(ReluVal(max_iter=100), prob)
NeuralVerification.compute_output(net, res.counter_example)

# Output
show(counter_example)
counter_example = [2.5, 4.5, 2.5, 
                   6.5, 8.5, 8.5, 
                   10.5, 12.5, 18.5, 
                   14.5, 16.5, 22.5, 
                   18.5, 20.5, 26.5,
                   22.5, 24.5, 32.5, 
                   26.5, 28.5, 38.5, 
                   30.5, 32.5, 44.5, 
                   35.5, 37.5, 54.5, 
                   39.5, 41.5, 60.5]