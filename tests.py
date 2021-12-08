import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy as np

#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
ts = np.loadtxt("trainingSet.txt", usecols=[1,2,3])
X = ts[:,0:2]
Y = ts[:,2]

mf = [
    [
        ['gaussmf',{'mean':0.0,'sigma':1.0}],
        ['gaussmf',{'mean':-1.0,'sigma':2.0}],
        ['gaussmf',{'mean':-4.0,'sigma':10.0}],
        ['gaussmf',{'mean':-7.0,'sigma':7.0}]
    ],
    [
        ['gaussmf',{'mean':1.0,'sigma':2.0}],
        ['gaussmf',{'mean':2.0,'sigma':3.0}],
        ['gaussmf',{'mean':-2.0,'sigma':10.0}],
        ['gaussmf',{'mean':-10.05,'sigma':5.0}]
    ]
]


mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))

if (round(anf.consequents[-1][0],6) == -5.275538 and 
    round(anf.consequents[-2][0],6) == -1.990703 and 
    round(anf.fittedValues[9][0],6) == 0.0002249
    ):
	print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
