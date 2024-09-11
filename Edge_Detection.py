import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras import Model
from tensorflow.keras.activations import relu

THESH = 30

class EdgeDetection:

    def __init__(self, img_path):
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #Lọc ảnh bằng Gauss
        self.image = cv2.GaussianBlur(raw_img, ksize=(7, 7), sigmaX=0, sigmaY=0)
        self.Sobel_img = self.sobel_edge_detection()
        self.Prewit_img = self.prewit_edge_detection()
        self.Canny_img = self.canny_edge_detection()
        self.DNN_img = self.DNN_edge_detection()

    def edgeshow(self):
        titles = ["Origin", "Sobel", "Prewit", "Canny", "DNN"]
        n = len(titles)
        edges = [self.image, self.Sobel_img, self.Prewit_img, self.Canny_img, self.DNN_img]
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.imshow(edges[i], cmap='gray')
            plt.title(titles[i])
        plt.show()

    def sobel_edge_detection(self):
        x = np.uint8(cv2.Sobel(self.image, cv2.CV_64F, dx=1, dy=0, ksize=1))
        y = np.uint8(cv2.Sobel(self.image, cv2.CV_64F, dx=0, dy=1, ksize=1))
        sobel = (x+y)/2
        sobel[sobel<THESH]=0
        sobel[sobel>=THESH]=255
        return sobel
    
    def prewit_edge_detection(self):
        kernels, imgs = [0]*6, []
        kernels[0] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernels[1] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernels[2] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernels[3] = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernels[4] = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        kernels[5] = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

        for i in range(len(kernels)):
            imgs.append(cv2.filter2D(self.image, ddepth=-1, kernel=kernels[i]))
        img = np.mean(imgs, axis=0)
        img[img<THESH]=0
        img[img>=THESH]=255
        return img
    
    def canny_edge_detection(self):
        img = cv2.Canny(self.image, 50, 170)
        img[img<THESH]=0
        img[img>=THESH]=255
        return img   
    
    def DNN(self):

        w1 = np.array([
            [[[1, -1, 1, -1]],[[0, 0, 1, -1]],[[-1, 1, 1, -1]]],
            [[[1, -1, 0, 0]],[[0, 0, 0, 0]],[[-1, 1, 0, 0]]],
            [[[1, -1, -1, 1]],[[0, 0, -1, 1]],[[-1, 1, -1, 1]]]
        ], dtype=np.float32)

        w2 = np.array([[[[0.25], [0.25], [0.25], [0.25]]]], np.float32)

        #input = Input(shape= [*self.image.shape, 1], batch_size=1, name='input')
        #x = Conv2D(2, (3,3), (1,1), 'same', name='Gxy', activation = relu, use_bias=False, weights = [w1])(input)
        #x = Conv2D(1, (1,1), (1,1), 'same', name='avg', activation = relu, use_bias=False, weights = [w2])(x)
        #return Model(input, x, name='DNN_Edge_Detection')

        input_layer = Input(shape= [*self.image.shape, 1], batch_size=1, name='input')

        # Khởi tạo lớp Conv2D mà không dùng trọng số ngay lúc này
        x = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), padding='same', 
                name='Gxy', activation=relu, use_bias=False)(input_layer)

        # Tạo thêm lớp Conv2D khác
        x = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', 
                name='avg', activation=relu, use_bias=False)(x)

        # Khởi tạo mô hình
        model = Model(input_layer, x, name='DNN_Edge_Detection')

        # Gán trọng số sau khi khởi tạo model
        model.get_layer(name='Gxy').set_weights([w1])
        model.get_layer(name='avg').set_weights([w2])

        return model
    
    def DNN_edge_detection(self):
        img = np.array(self.image)
        img = img.reshape(1, *self.image.shape, 1)
        model = self.DNN()
        output = model.predict(img)
        rs = np.array(output[0,:,:,0])
        rs[rs<THESH]=0
        rs[rs>=THESH]=255
        return rs

    def hough_line_detection(self):
        dst = self.Canny_img
        dsts = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 15, None, minLineLength=25, maxLineGap=30)
        #cv2.HoughLinesP return các điểm (a_i, b_i) trong Hough Space, là đường thẳng trong Image Space y = a_i*x + b_i
        if linesP is not None:
            for i in range(len(linesP)):
                l = linesP[i][0]
                cv2.line(dsts, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
        return dsts
    
    def lineshow(self):
        rs = self.hough_line_detection()
        plt.imshow(rs)
        plt.show()

path_crt = os.getcwd()
data_path = os.path.join(path_crt, "CVA", "Unit4_Edge_Detection", "data")
paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]
EdgeDetection(paths[0]).edgeshow()

class Evaluation:
    def __init__(self, data_folder):
        self.list_paths = [os.path.join(data_folder, file_name) for file_name in os.listdir(data_folder)]

    def plot_histogram(self):
        histos = dict({"sobel":[], "prewit":[], "canny":[]})
        for img_path in self.list_paths:
            img_edge = EdgeDetection(img_path)
            edges = [img_edge.Sobel_img, img_edge.Prewit_img, img_edge.Canny_img]
            for i, k in enumerate(histos.keys()):
                histos[k].append(len(edges[i][edges[i]==255])) #thêm số pixel có giá trị 255 trong edges[i]
        for i, k in enumerate(histos.keys()):
            plt.subplot(1,3,i+1)
            plt.hist(histos[k], 30, [0, 2000])
            plt.title(str(k))
        plt.show()


#path_crt = os.getcwd()
#data_path = os.path.join(path_crt, "CVA", "Unit4_Edge_Detection", "data_lane")
#paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]
#print(paths)
#for i in paths:
    #plt.subplot(1,2,1)
    #plt.imshow(EdgeDetection(i).Canny_img)
    #plt.subplot(1,2,2)
    #EdgeDetection(i).lineshow()