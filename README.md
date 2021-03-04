# python enviroment
```
dlib==19.20.0
opencv_python==4.2.0.34
onnxruntime==1.2.0
fastapi==0.61.0
numpy==1.19.1
```

# How to use ?
##  1. build the target database
Put all target pictures in one directory (**data/mask_nomask** in this project), one person with two picture masked and none-masked is recommended, the file naming is **name.index.jpg**

![ target database](https://img-blog.csdnimg.cn/20200813183849696.png?#pic_center)
## 2. dowload the weight files
you can download the all weight files from 
1. [baiduyunpan](https://pan.baidu.com/s/1i9i7Y3eclsiz95BSMl0NUA) passwd：60ri  need a baidu account
2. [lanzhoucloud](https://wws.lanzous.com/igD2Ymgphfa)   no restriction to download

and put them in data directory

## 3. test a video demo

```bash
python demo_video.py  --face_db_root data/mask_nomask --input_video_path 0.mp4 --output_video_path output.mp4
```
![face_video_demo](https://img-blog.csdnimg.cn/20200813221028400.gif#pic_center)
## 4. test in fastapi server

```bash
uvicorn main:app --reload --host 0.0.0.0
```
you can upload a picture from http://127.0.0.1:8000/docs,and retrive the face recognition information [name,box,mask or not]
![fastapi](https://img-blog.csdnimg.cn/20200813221935981.jpg#pic_center)

# blogs
https://blog.csdn.net/jiangpeng59/article/details/107986046

# reference
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
https://github.com/AIZOOTech/FaceMaskDetection
https://github.com/davisking/dlib
