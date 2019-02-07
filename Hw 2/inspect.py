from __future__ import division
import sys
import math

inFile = sys.argv[1]
outFile = sys.argv[2]
i = open(inFile,"r")
o = open(outFile,"w")

data = i.read()
lines = data.splitlines()
totalObs = len(lines)
trueCount = 0
for j in range(totalObs) :
    #skip labels
    if (j==0): continue
    row = lines[j].split(",")
    if (j==1):
        indicator = row[-1]
    if (indicator == row[-1]):
        trueCount += 1

#calculate entropy
pTrue = trueCount / (totalObs-1)
pFalse = (totalObs-trueCount-1) / (totalObs-1)
entropy = -1 * ((pTrue * math.log(pTrue,2))+(pFalse * math.log(pFalse,2)))

#calculate error
if (trueCount > (totalObs/2)): error = pFalse
else: error = pTrue

o.write("entropy: %f" % entropy + "\n")
o.write("error: %f" % error)

i.close()
o.close()
