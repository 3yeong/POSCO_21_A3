import cv2
# 연결된 카메라의 개수를 확인하기 위한 인덱스 변수
index = 0
while True:
    # 카메라를 연결
    cap = cv2.VideoCapture(index)
    # 카메라 연결이 되지 않은 경우, 다음 인덱스로 이동
    if not cap.isOpened():
        index += 1
    # 카메라 연결이 된 경우, 카메라 번호를 출력하고 다음 인덱스로 이동
    else:
        print("Camera Number: ", index)
        cap.release()
        index += 1
    # 인덱스가 10이상이 되면 종료
    if index > 10:
        break