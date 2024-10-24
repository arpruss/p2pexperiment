import sys
import math

chunks = {}

with open(sys.argv[1]) as f:
    for line in f:
        data = tuple(map(float, line.strip().split()))
        chunk = math.floor(0.5+data[2]/10)*10
        if chunk not in chunks:
            chunks[chunk] = []
        chunks[chunk].append( (data[2],data[3]) )
    
for i in sorted(tuple(chunks.keys())):
    z = 0
    rms = 0
    maxE = 0
    for pair in chunks[i]:
        z += pair[0]
        rms += pair[1]*pair[1]
        maxE = max(pair[1],maxE)
    z /= len(chunks[i])
    rms = math.sqrt(rms / len(chunks[i]))
    print(z,rms,maxE)