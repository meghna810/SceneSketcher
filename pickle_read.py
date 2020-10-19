import pickle
with open('result_final.pickle', 'rb') as handle:
    result = pickle.load(handle)
print(result)
