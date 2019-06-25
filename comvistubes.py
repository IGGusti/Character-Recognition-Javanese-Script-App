import cv2
import numpy as np
MIN_MATCH_COUNT=8

detector=cv2.SIFT()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=3)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=[]
trainImg.append(cv2.imread("Karakter/ha.jpg",0))
trainImg.append(cv2.imread("Karakter/ra.jpg",0))
trainImg.append(cv2.imread("Karakter/na.jpg",0))
nama = ["Huruf HA","Huruf RA","Huruf NA"]
trainKP=[]
trainDesc=[]
for i in range(3):
    a,b=(detector.detectAndCompute(trainImg[i],None))
    trainKP.append(a)
    trainDesc.append(b)
hitung=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 1)
cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    for i in range(3):
        matches=flann.knnMatch(queryDesc,trainDesc[i],k=2)
        matchespembanding=flann.knnMatch(queryDesc,trainDesc[i-1],k=2)
        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.65*n.distance):
                goodMatch.append(m)
        
        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp=[]
            qp=[]
            for m in goodMatch:
                tp.append(trainKP[i][m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp=np.float32((tp,qp))
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            
            h,w=trainImg[i].shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
            hitung=hitung+1
            print "Ketemu- %d/%d, ke-%d"%(len(goodMatch),MIN_MATCH_COUNT,hitung)
        else:
            hitung=0
            print "Ga Ketemu- %d/%d, ke-%d"%(len(goodMatch),MIN_MATCH_COUNT,hitung)

        kellima=hitung%2
        if hitung>=2:

            if kellima == 0:
                if matches<matchespembanding:
                    cv2.cv.PutText(cv2.cv.fromarray(QueryImgBGR),nama[i],(w,h-1),font, (0,255,255))
                    print nama[i]
                else:
                    cv2.cv.PutText(cv2.cv.fromarray(QueryImgBGR),nama[i-1],(w,h-1),font, (0,255,255))
                    print nama[i-1]
        if i == 3:
            matchespembanding = 0
        cv2.imshow('Hasil Video Capture',QueryImgBGR)
        if cv2.waitKey(10)==ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
  
