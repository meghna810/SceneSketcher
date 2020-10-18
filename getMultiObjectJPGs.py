import ast

with open('pickle_read.csv') as f:
    data = f.read()

#print("Data type before reconstruction : ", type(data))

d = ast.literal_eval(data)

#print("Data type after reconstruction : ", type(d))

count = 0
multiObjImg = []
d_items = d.items()
for key, value in d.items():

    if len(value) > 1:
        count = count + 1
        # comment this out later
        print(key, '->', value)

        multiObjImg.append(key)


# comment this out later
print(count)


print(multiObjImg)