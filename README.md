# -StereTracker
这是一个基于X角点检测的双目视觉测量系统
## 系统依赖
* OpenCV2.4.13
* Eigen3
* DirectShow

## 硬件平台
双目相机
![检测](https://github.com/yaoxinghua/-StereTracker/blob/master/docs/test_images/%E5%8F%8C%E7%9B%AE%E5%B9%B3%E5%8F%B0.png)  

## 原理介绍

### 角点检测
X角点检测采用了基于Fast特征点作为候选点，对候选点计算response响应值，并进行阈值选择的方法。
![检测](https://github.com/yaoxinghua/-StereTracker/blob/master/docs/test_images/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B.png)  

### 角点匹配
左右图像进行立体矫正之后，可实现图像行对其，对检测到的角点同一行进行匹配，对同一行存在多个的情况，排序后匹配。
![检测](https://github.com/yaoxinghua/-StereTracker/blob/master/docs/test_images/%E8%A7%92%E7%82%B9%E5%8C%B9%E9%85%8D.png) 

### 三角测量
获取同一角点在左右图像的坐标之后，通过三角测量，计算该点的空间坐标。
![检测](https://github.com/yaoxinghua/-StereTracker/blob/master/docs/test_images/%E4%B8%89%E8%A7%92%E6%B5%8B%E9%87%8F.png) 

### Marker检测
对所有点进行任意三个点组合，若三点可组成直角三角形，可认为近似Marker，与模板库中保存的Marker进行匹配，获取Marker的name。
![检测](https://github.com/yaoxinghua/-StereTracker/blob/master/docs/test_images/Marker%E8%AF%86%E5%88%AB.png) 



