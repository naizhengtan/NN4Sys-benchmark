using NeuralVerification, LazySets
import NeuralVerification: ReLU, Id
# Zeroes case
# Zero sent latency, Zero high latency, Zero high send ratio
# A reduction in send rate.
lowerInput = [0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0]

upperInput = [0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0]

x = Hyperrectangle(low = lowerInput, high = upperInput)
y = PolytopeComplement(low = [-0.001], high = [0.001]) ??

model = "/Users/Chioma_N/Desktop/object/PCC-RL-master/src/gym/tensorflow/reLuNet/resNet"


net = read_nnet(model)
prob = Problem(net, x, y)
res = solve(NSVerify(max_iter=100), prob) ??
NeuralVerification.compute_output(net, res.counter_example)