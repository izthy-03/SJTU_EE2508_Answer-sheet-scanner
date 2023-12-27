## SJTU-EE2508_Answer-sheet-scanner

<hr/>

克隆仓库

```bash
git clone https://github.com/izthy-03/SJTU_EE2508_Answer-sheet-scanner.git
```

安装运行依赖

```bash
pip install -r requirements.txt
```



#### 运行方法

- 视频流

更改main.py中的`video_src`自为己的摄像机id或RTSP URL，然后运行main.py

q键退出

- 单张图片

将图片移动至`./assets/img/` 下，更改test_once.py中的`imgpath`，然后运行test_once.py

#### 标准答案

预设答案保存在`./assets/standard.csv`
