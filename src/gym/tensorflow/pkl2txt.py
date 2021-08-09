import pickle 
with open('./reLuNet/sup_weights.pkl', 'rb') as f:
    inputs = pickle.load(f)

with open('./show.txt', 'w') as f:
    for item in inputs:
        f.write("%s\n" % item)


# with open('sup_bias.pkl', 'rb') as f:
#     bias = pickle.load(f)


# with open('sup_bias_file.txt', 'w') as f:
#     for item in bias:
#         f.write("%s\n" % item)

