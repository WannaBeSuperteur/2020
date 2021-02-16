def recover_original_value(value, num):

    # training data 0: original
    # training data 1: difference
    if num == 0 or num == 1:
        return round(value)

    # training data 2: log(x+1) if x >= 0 and -log((-x)+1) if x < 0
    # training data 3: log(difference+1) if x >= 0 and -log((-difference)+1) if x < 0
    elif num == 2 or num == 3:

        # x        = log(y + 1)
        # 10^x     = y + 1
        # 10^x - 1 = y
        if value >= 0:
            return round(pow(10.0, value) - 1.0)

        # x           = -log((-y) + 1)
        # -x          = log((-y) + 1)
        # 10^(-x)     = -y + 1
        # 10^(-x) - 1 = -y
        # 1 - 10^(-x) = y
        else:
            return round(1.0 - pow(10.0, -value))

def read_valid_report(num):

    f = open('report_val_' + str(num) + '.txt')
    fl = f.readlines()
    f.close()

    count = len(fl) - 10
    correct = 0
    toWrite = ''

    # read each line
    for i in range(count):
        thisLine = fl[i]

        pred = float(thisLine.split('[')[2].split(']')[0])
        real = float(thisLine.split('[')[3].split(']')[0])

        pred_ = recover_original_value(pred, num)
        real_ = recover_original_value(real, num)

        if pred_ == real_: correct += 1
        toWrite += ('pred = ' + str(pred_) + ', real = ' + str(real_) + '\n')

    toWrite += ('count = ' + str(count) + ', correct = ' + str(correct) + '\n')
    toWrite += ('result = ' + str(correct / count))

    # write result
    fw = open('report_val_' + str(num) + '_result.txt', 'w')
    fw.write(toWrite)
    fw.close()

if __name__ == '__main__':
    read_valid_report(3)
