import sys

inFile = sys.argv[1]
outFile = sys.argv[2]

i = open(inFile,"r")
o = open(outFile,"w")

contents = i.read()
lines = contents.splitlines()
for line in lines[::-1]:
    o.write(line + "\n")

i.close()
o.close()
