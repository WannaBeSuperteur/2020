timelog = open('time_log.txt', 'r')
time_log = timelog.readlines()
timelog.close()

timeSum = {}
timeCnt = {}

lastCurrent = None
startAt = 2000 # except before deep learning

def x0(x): return x[0]

for i in range(startAt, len(time_log)):
    if i % 1000 == 0: print(i)
    
    time_log_items = time_log[i].split(' ')
    
    if len(time_log_items) == 3:
        func    = time_log_items[1].split('[')[1].split(']')[0]
        inout   = time_log_items[2].split('[')[1].split(']')[0]
        current = float(time_log_items[0])

    if lastCurrent != None:
        try:
            timeSum[func + ',' + inout] += current - lastCurrent
            timeCnt[func + ',' + inout] += 1
        except:
            timeSum[func + ',' + inout] = current - lastCurrent
            timeCnt[func + ',' + inout] = 1

    lastCurrent = current

print('avgTime is x10,000\n')
print('case\t\t\t time\t       count\t avgTime')
print('---------------------------------------------------------')
for case, time in sorted(timeSum.items(), key=x0):
    
    cnt     = timeCnt[case]
    avgTime = time / cnt

    print(case + ' '*(24-len(case)),
          round(time, 4), ' '*(12-len(str(round(time, 4)))),
          cnt, ' '*(8-len(str(cnt))),
          round(avgTime * 10000, 4))
