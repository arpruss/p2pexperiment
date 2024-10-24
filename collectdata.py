import sys
import math
import msvcrt

MARKER = "ARUCO_FULL_DATA: "
DATA_CHUNK = 50

count = 0
WAIT = 0
READ = 1
MESSAGE = 2

mode = MESSAGE
buffer = []

with open(sys.argv[1],"w") as out:
    while True:
        if mode == MESSAGE:
            print("SPACE = read, ESC = exit")
            mode = WAIT        
        if mode == WAIT:
            if (msvcrt.kbhit()):
                c = ord(msvcrt.getch())
                if c == 32:
                    print("Reading...")
                    mode = READ
                    count = 0
                elif c == 27:
                    print("Done")
                    sys.exit(0)
        line = sys.stdin.readline().strip()
        if mode == READ:
            try:
                idx = line.index(MARKER)
                data = tuple(map(float,line[idx+len(MARKER):].split()))
                out.write("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (data[0],data[1],data[2],
                    math.hypot(data[0]-data[3],data[1]-data[4],data[2]-data[5]),
                    abs(data[0]-data[3]),abs(data[1]-data[4]),abs(data[2]-data[5])))
                print('.',end='',flush=True)
                count += 1
                if count >= DATA_CHUNK:
                    print('')
                    mode = MESSAGE
            except ValueError:
                pass
                    
