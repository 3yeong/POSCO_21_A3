import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import requests
import hmac
import hashlib
import json


cudnn.benchmark = True
plt.ion()   # interactive mode


@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):
    global model_pred


    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))
        
        violence_count = 0 # violece_count 0
        while(cap.isOpened): #loop until cap opened or video not complete
        
            # print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            # print("No of Objects in Current Frame : {}".format(n))

                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                        orig_shape=im0.shape[:2])
                        x1, y1, x2, y2 = map(int, xyxy)
                        cropped_img = im0[y1:y2, x1:x2]
                        # print(type(cropped_img))
                        if cropped_img is not None and cropped_img.size != 0:
                            pass
                        pred = detect_violence(cropped_img)
                        if pred[0]: # tensor(0)
                            # print('-'*30)
                            print('violence!!!!')
                            print(violence_count)
                            cv2.imwrite('./frames/saved_image{}_{}.jpg'.format(frame_count, det_index), cropped_img)
                            violence_count += 1 #violence count

                            if (violence_count == 20.0) or (violence_count == 20.5) : # tensor(0)
                                violence_count += 1 #violence count
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                print("점주에게 메시지가 전송되었습니다.")
                                # cv2.imwrite('./frames/saved_image{}_{}.jpg'.format(frame_count, det_index), cropped_img)
                                # # send SMS to specified number
                                # api_key = "-----------"
                                # secret_key = "------------------------"
                                # service_id = "-----------------------"
                                # sender_num = "----------------"
                                # receiver_num = "--------------"
                                # message = "Violence Detected!"
                                # send_sms(api_key, secret_key, service_id, sender_num, receiver_num, message)
                                # violence_count  = 0

                        else: # tensor(1)
                            print('normal!!!!')
                            if violence_count > 0:
                                violence_count -= 0.5 #violence count
                            
              
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  #writine('./frames

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def send_sms(api_key, secret_key, service_id, sender_num, receiver_num, message):
    method = "POST"
    url = f"https://sens.apigw.ntruss.com/sms/v2/services/{service_id}/messages"

    timestamp = str(int(time.time() * 1000))
    access_key = api_key
    secret_key = secret_key

    message_bytes = bytes(json.dumps({
        "type": "SMS",
        "from": sender_num,
        "content": message,
        "messages": [{"to": receiver_num}]
    }), 'utf-8')

    signature = make_signature(method, url, timestamp, access_key, secret_key, message_bytes)

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "x-ncp-apigw-timestamp": timestamp,
        "x-ncp-iam-access-key": access_key,
        "x-ncp-apigw-signature-v2": signature
    }

    body = {
        "type": "SMS",
        "from": sender_num,
        "content": message,
        "messages": [{"to": receiver_num}]
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 202: # 성공여부 확인 HTTP상태코드
        print("SMS Sent Successfully!")
    else:
        print("Failed to Send SMS: {}".format(response.text))

def make_signature(method, url, timestamp, access_key, secret_key, message_bytes):
    secret_key_bytes = bytes(secret_key, 'utf-8')
    message = f"{method} {url}\n{timestamp}\n{access_key}\n"
    message_bytes = bytes(message, 'utf-8') + message_bytes

    signing_key = hmac.new(secret_key_bytes, message_bytes, hashlib.sha256).digest()

    signature = hmac.new(signing_key, message_bytes, hashlib.sha256).hexdigest()

    return signature

    url = "https://api-sens.ncloud.com/v1/sms/services/{}/messages".format(service_id)

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 202: #성공여부 확인 HTTP상태코드
        print("SMS Sent Successfully!")
    else:
        print("Failed to Send SMS: {}".format(response.text))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    
    
def detect_violence(frame):
    predictions = []
    image_transforms = transforms.Compose([transforms.Resize(size=(244,244)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    f = Image.fromarray(frame)
    f = image_transforms(f)
    f = f.unsqueeze(0)
    prediction = model_pred(f)
    prediction = prediction.argmax()
    # print(prediction)
    predictions.append(prediction.data)
    return predictions   
    
def model_load():
    global model_pred
    #Model loading
    PATH = './CNN/cnnModels/weight/04.pth'
    model_pred = models.mobilenet_v2(pretrained=True)
    model_pred.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    model_pred.load_state_dict(torch.load(PATH))
    model_pred.eval()

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    model_load()
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
