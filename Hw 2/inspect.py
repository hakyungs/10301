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
for (i in in range(totalObs)):
    if (i==0): continue
    row = lines.split(",")
    if (i==1):
        indicator = row[3]
    if (indicator == row[3]):
        trueCount += 1

#calculate entropy
pTrue = trueCount / totalObs
pFalse = (totalObs-trueCount) / totalObs
entropy = -1 * ((pTrue * math.log(pTrue,2))+(pFalse * math.log(pFalse,2)))

#calculate error


o.write(line + "\n")

i.close()
o.close()
