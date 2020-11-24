def saveAsFile(array, dim, fn):
    result = ''

    # 파일이 이미 존재하면 쓰지 않기
    try:
        f = open(fn, 'r')
        print('the file "' + fn + '" already exists.')
        f.close()
        return
    except:
        doNothing = 0 # do nothing
    
    if dim == 1: # 1차원 배열의 집합
        for i in range(len(array)): # 각 1차원 배열
            for j in range(len(array[0])): # 각 1차원 배열 (의 원소)
                if j < len(array[0])-1: result += str(array[i][j]) + '\t'
                else: result += str(array[i][j]) + '\n'

    elif dim == 2: # 2차원 배열의 집합
        for i in range(len(array)): # 각 2차원 배열
            for j in range(len(array[0])): # 각 2차원 배열 (의 각 행)
                for k in range(len(array[0][0])): # 각 2차원 배열의 각 행 (의 각 원소)
                    if k < len(array[0])-1: result += str(array[i][j][k]) + '\t'
                    else: result += str(array[i][j][k]) + '\n'

    # 파일에 쓰기
    f = open(fn, 'w')
    f.write(result)
    f.close()
