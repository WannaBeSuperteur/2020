# 계획
# 1. 함수 -> y=x^2, y=x 등등
# 2. 분포 -> 정규분포의 표준편차 o를 더해서 분산되게 함
# 3. uniform distribution

import random
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import saveFiles as SF
import numpy as np

# 범위: x=0~N, y=0~N
N = 500

# 일차함수 생성
def linear(a, b, x):
    return a*x + b

# 이차함수 생성
def quad(a, b, c, x):
    return a*x*x + b*x + c

# 데이터 표시
def displayData(ys, title):
    xs = []
    for i in range(N): xs.append(i)

    plt.suptitle(title)
    plt.plot(xs, ys, '.')
    plt.axis([0, N, np.min(ys), np.max(ys)])
    plt.show()

# 값 생성
# gaussianDistDev : 정규분포의 표준편차
# uniformDistDev  : 균등분포의 최대편차 (양쪽이므로 값 분포범위의 크기는 그 2배)
# graphType       : 0이면 상수=0, 1이면 일차함수, 2이면 이차함수
# display         : 데이터 표시 여부
# imgHeight       : CNN의 입력으로 들어갈 이미지의 높이
# imgWidth        : CNN의 입력으로 들어갈 이미지의 너비
def createValues(gaussianDistDev, uniformDistDev, a, b, c, graphType, display, imgHeight, imgWidth):

    ys = [] # y 값을 저장한 배열

    # 함수 적용 (0은 상수함수, 1은 일차함수, 2는 이차함수)
    if graphType == 0: # 그대로
        for x in range(N): ys.append(0)
    elif graphType == 1: # 일차함수
        for x in range(N): ys.append(linear(a, b*N, x))
    elif graphType == 2: # 이차함수
        for x in range(N): ys.append(quad(a/N, b, c*N, x))

    # 편차 추가 전의 데이터 표시
    if display == True:
        if graphType != 0: displayData(ys, 'data before adding deviation')

    # 분포에 따른 편차 추가
    for i in range(N):
        ys[i] += uniformDistDev * (random.random() * 2 - 1)
        ys[i] += np.random.normal(0, gaussianDistDev, 1)[0]

    # 편차 추가 후의 데이터 표시
    if display == True: displayData(ys, 'data after adding dev, before fitting')

    # 데이터를 x = 0~N, y = 0~N 으로 fit
    yMax = np.max(ys)
    yMin = np.min(ys)
    for i in range(N): ys[i] = (ys[i] - yMin) / (yMax - yMin) * N

    # fitting 후의 데이터 표시
    if display == True: displayData(ys, 'data after fitting')

    # yRank를 이미지로 저장
    img = [[0 for j in range(imgWidth)] for i in range(imgHeight)]
    blockHeight = N / imgHeight
    blockWidth = N / imgWidth
    
    for i in range(N):
        img[imgHeight - int(ys[i] / blockHeight) - 1][int(i / blockWidth)] += 1

    # 이미지의 값을 0~1 사이로 표준화
    imgMax = np.max(img)
    for i in range(imgHeight):
        for j in range(imgWidth):
            img[i][j] /= imgMax

    # 이미지 출력
    if display == True:
        np.set_printoptions(precision=3, linewidth=150)
        img = np.array(img)
        for i in range(imgHeight): print(img[i])

        # 배열을 시각적으로 출력
        cmap = clr.LinearSegmentedColormap.from_list('dddd', ['#000', '#00F', '#0FF', '#0F0', '#FF0', '#F00'], N=256)
        plt.imshow(img, cmap=cmap)
        plt.show()

    # 이미지 반환
    return (ys, img)

# 메인 코드 실행
if __name__ == '__main__':
    imgs = []

    for i in range(100):
        gdd = random.random() * 150
        udd = random.random() * 150
        
        A = random.random()*2 - 1
        B = A * random.random() * (-2)
        C = random.random()
        
        (ys, img) = createValues(gaussianDistDev=gdd, uniformDistDev=udd, a=A, b=B, c=C,
                     graphType=1, display=False, imgHeight=10, imgWidth=10)
        imgs.append(img)

    SF.saveAsFile(imgs, 2, 'test_images.txt')
