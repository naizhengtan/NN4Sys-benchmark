using NeuralVerification, LazySets
import NeuralVerification: ReLU, Id
# Congestion case
# Constant high latency gradient, constant high latency ratio, constant high send ratio
# Ranges: Latency gradient (0 - 100.0), Latency ratio (1.0 - 10000.0), Send ratio (0.0 - 1000.0)
# Expect A reduction in send rate.
lowerInput = [95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0]

upperInput = [95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0,
       95.0, 7550.0, 800.0]

lowerInput = [95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0]

upperInput = [95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0,
       95.0, 50.0, 1000.0]

x = Hyperrectangle(low = lowerInput, high = upperInput)
y = Hyperrectangle(low = [-1000], high = [-0.001])


model = "/Users/Chioma_N/Desktop/object/PCC-RL-master/src/gym/tensorflow/reLuNet/resNet"


net = read_nnet(model)
prob = Problem(net, x, y)
res = solve(ReluVal(max_iter=100), prob)
NeuralVerification.compute_output(net, res.counter_example)