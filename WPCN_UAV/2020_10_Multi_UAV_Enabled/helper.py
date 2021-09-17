def loadSettings(args):

    argValues = {}

    # load settings and find learning rate
    settingsFile = open('settings.txt', 'r')
    settings = settingsFile.readlines()
    settingsFile.close()

    for i in range(len(settings)):
        argName = settings[i].split('=')[0]
        argValue = settings[i].split('\n')[0].split(' ')[0].split('=')[1]

        for args_name, args_type in args.items():
            if argName == args_name:

                if args_type == 'int':
                    argValues[argName] = int(argValue)
                    
                elif args_type == 'float':
                    argValues[argName] = float(argValue)
                    
                elif args_type == 'str':
                    argValues[argName] = str(argValue)
                    
                elif args_type == 'logical':
                    if argValue == 'False' or argValue == 'FALSE' or argValue == 'false' or argValue == '0':
                        argValues[argName] = False
                    else:
                        argValues[argName] = True
                    
                break

    return argValues

if __name__ == '__main__':
    argValues = loadSettings({'deviceName':'str',
                              'QTable_rate':'float',
                              'M':'int',
                              'printTimeDif':'logical'})

    print(argValues)
    print(argValues['deviceName'])
