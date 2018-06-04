from sklearn import datasets

l = datasets.load_digits()
digits = l.data
targets = l.target
for digit, target in list(map(lambda x: tuple(x), list(zip(digits, targets)))):
    row = ','.join(list(map(str, digit)))
    row += ',' + str(target)
    print(row)
