from sklearn.naive_bayes import GaussianNB

x= [[170,65],[180,80], [160,50]] #altura y peso
y= ['hombre', 'hombre', 'mujer']

model = GaussianNB()
model.fit(x,y)
