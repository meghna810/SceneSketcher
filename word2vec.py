from gensim.models import Word2Vec

labels = [['bicycle'], ['car'], ['motorbike'], ['aeroplane'], ['traffic light'], ['fire hydrant'], ['cat'], ['dog'], ['horse'], ['sheep'], ['cow'], ['elephant'], ['zebra'], ['giraffe']]

model = Word2Vec(labels, size=300,min_count=1)

words = model.wv.vocab

vector = model.wv['bicycle']
print(vector)
