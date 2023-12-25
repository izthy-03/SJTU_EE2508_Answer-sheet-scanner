import cv2
# 创建一个窗口 名字叫做cap
cv2.namedWindow('cap', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
'''
#打开USB摄像头
cap = cv2.VideoCapture(0)
'''
# 摄像头的IP地址,http://用户名：密码@IP地址：端口/
ip_camera_url = 'rtsp://admin:123456@10.180.249.36:8554/live'
# 创建一个VideoCapture
cap = cv2.VideoCapture(ip_camera_url)
print('IP摄像头是否开启： {}'.format(cap.isOpened()))
# 显示缓存数
print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
# 设置缓存区的大小
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# 调节摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 设置FPS
print('setfps', cap.set(cv2.CAP_PROP_FPS, 25))
print(cap.get(cv2.CAP_PROP_FPS))
while True:
    # 逐帧捕获
    ret, frame = cap.read()  
    cv2.imshow('cap', frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break
# 退出后，释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()