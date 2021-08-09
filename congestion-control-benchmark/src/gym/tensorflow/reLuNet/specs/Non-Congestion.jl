using NeuralVerification, LazySets
import NeuralVerification: ReLU, Id
# Non-Congestion case
# Constant low latency gradient, constant low latency ratio, constant mid send rate
# Ranges: Latency gradient (0 - 100.0), Latency ratio (1.0 - 10000.0), Send ratio (0.0 - 1000.0)

# Non-congestion
lowerInput = [40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0]

upperInput = [40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0,
       40.0, 60.0, 6.0]

lowerInput = [5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0]

upperInput = [5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0,
       5.0, 350.0, 300.0]

x = Hyperrectangle(low = lowerInput, high = upperInput)
y = Hyperrectangle(low = [-1.5], high = [1000])

model = "/Users/Chioma_N/Desktop/object/PCC-RL-master/src/gym/tensorflow/reLuNet/resNet"


net = read_nnet(model)
prob = Problem(net, x, y)
res = solve(ReluVal(max_iter=100), prob)
