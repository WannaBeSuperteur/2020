timelog = open('time_log.txt', 'r')
time_log = timelog.readlines()
timelog.close()

timeSum = {}
lastCurrent = None

for i in range(len(time_log)):
    if i % 1000 == 0: print(i)
    
    time_log_items = time_log[i].split(' ')
    
    if len(time_log_items) == 3:
        func    = time_log_items[1].split('[')[1].split(']')[0]
        inout   = time_log_items[2].split('[')[1].split(']')[0]
        current = float(time_log_items[0])

    if lastCurrent != None:
        try:
            timeSum[func + ',' + inout] += current - lastCurrent
        except:
            timeSum[func + ',' + inout] = current - lastCurrent

    lastCurrent = current

for key, value in timeSum.items():
    print('before', key, '\t\t', round(value, 4))
