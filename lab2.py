import numpy as np
import cv2

def main():

    # Question 1
    
    Rz = np.matrix([[np.cos(0.1),-np.sin(0.1),0], [np.sin(0.1), np.cos(0.1), 0], [0,0,1]])
    Ry = np.matrix([[np.cos(-0.5),0,np.sin(-0.5)], [0, 1, 0], [-np.sin(-0.5),0,np.cos(-0.5)]])
    Rx = np.matrix([[1,0,0], [0, np.cos(1.1), -np.sin(1.1)], [0,np.sin(1.1),np.cos(1.1)]])

    Rxyz = Rz@Ry@Rx

    print("Question 1a.")
    print(Rxyz)

    RxyzT = Rxyz.T
    RxyzInv = np.linalg.inv(Rxyz)

    print("\nQuestion 1b." )
    print("Rxyz Transpose:")
    print(RxyzT)
    print("Rxyz Inverse:")
    print(RxyzInv)

    Rzyx = Rx@Ry@Rz

    print("\nQuestion 1c.")
    print(Rzyx)
    print("\n")

    # Question 2

    addCol = np.array([[10,-25,40]])
    addRow = np.array([[0,0,0,1]])
    Hwc_n = np.concatenate((Rxyz,addCol.T), axis=1)
    Hwc = np.concatenate((Hwc_n,addRow), axis=0)
    print("\nQuestion 2a.")
    print(Hwc)

    Hcw = np.linalg.inv(Hwc)
    print("\nQuestion 2b.")
    print(Hcw)

    K = np.matrix([[400,0,256/2], [0,400,170/2], [0,0,1]])
    print("\nQuestion 2c.")
    print(K)

    p1 = np.array([[6.8158,-35.1954,43.0640,1]])
    p2 = np.array([[7.8493,-36.1723,43.7815,1]])
    p3 = np.array([[9.9579,-25.2799,40.1151,1]])
    p4 = np.array([[8.8219,-38.3767,46.6153,1]])
    p5 = np.array([[9.5890,-28.8402,42.2858,1]])
    p6 = np.array([[10.8082,-48.8146,56.1475,1]])
    p7 = np.array([[13.2690,-58.0988,59.1422,1]])

    blankIm = np.zeros((256,170,3))
    
    Mext = Hcw[0:3,:]
    # Mext = 
    print(Mext)

    p1c = K@Mext@p1.T
    p1cf = (p1c/p1c[2])[0:2].round().astype(int)
    
    print(p1cf)

    p2c = K@Mext@p2.T
    p2cf = (p2c/p2c[2])[0:2].round().astype(int)
    p3c = K@Mext@p3.T
    p3cf = (p3c/p3c[2])[0:2].round().astype(int)
    p4c = K@Mext@p4.T
    p4cf = (p4c/p4c[2])[0:2].round().astype(int)
    p5c = K@Mext@p5.T
    p5cf = (p5c/p5c[2])[0:2].round().astype(int)
    p6c = K@Mext@p6.T
    p6cf = (p6c/p6c[2])[0:2].round().astype(int)
    p7c = K@Mext@p7.T
    p7cf = (p7c/p7c[2])[0:2].round().astype(int)

    blankIm[p1cf[0],p1cf[1]] = (256,256,256)
    blankIm[p2cf[0],p2cf[1]] = (256,256,256)
    blankIm[p3cf[0],p3cf[1]] = (256,256,256)
    blankIm[p4cf[0],p4cf[1]] = (256,256,256)
    blankIm[p5cf[0],p5cf[1]] = (256,256,256)
    blankIm[p6cf[0],p6cf[1]] = (256,256,256)
    blankIm[p7cf[0],p7cf[1]] = (256,256,256)

    cv2.imshow("result", blankIm)
    cv2.waitKey(0)
    

    
    
        

if __name__ == "__main__":
    main()
