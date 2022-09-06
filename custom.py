"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsColorizeEffect, QDesktopWidget
from PyQt5.QtGui import QPixmap, QColor, QPalette, QMovie



from os import startfile
from random import randint


from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


import sys
import time
import datetime
from pathlib import Path

import pandas as pd

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized







@torch.no_grad()
def run(weights='runs/weights/best.pt',  # model.pt path(s)
        source='1',  # file/dir/URL/glob, 0 for webcam
        imgsz=416,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Load Prices
    prices = pd.read_csv("Invetory_pricing.csv")
    price_list = pd.Series.tolist(prices["Price"])
    
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    net_det_count = [0]*len(names)
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
                        
            curr_det_count = [0]*len(names)
            
            #reset net det counter
            if ui.reset_flag > 0:
                ui.reset_flag = 0
                net_det_count = [0]*len(names)
                
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            listed = ""
            price_listed = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                total_price = 0
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    ls_item = f"{n} x {names[int(c)]}"
                    ls_price = f"Tk.{price_list[int(c)]}"
                   
                    tot_len = len(ls_item) + len(ls_price)
                    
                    listed += ls_item + "\n"
                    
                    curr_det_count[int(c)] += int(n)
                    item_price = int(n) * int(price_list[int(c)])
                    price_listed.append(item_price)
                    
                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                     

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        
                            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            
            if ui.button_flag: 
                if ui.lock_flag > 0:
                                     
                    ui.lock_flag = 0
                    ui.listPrice.clear()
                    ui.listWidget.clear()
                    
                    for i in range(len(names)):
                        net_det_count[i] += curr_det_count[i]
                        if net_det_count[i] > 0:
                            ui.update_item(f"{net_det_count[i]} x {names[i]}")   
                            ui.update_price("\u09F3" + str(net_det_count[i]*int(price_list[i])))
                    print(f"net_count = {net_det_count}")
                    ui.color_locked()
                    ui.update_total()             
                ui.clear_list()
                
            else:
                ui.lock_flag = 0 #reset lock flag if not billing
            
            if len(listed):
                if ui.button_flag:
                    test_list = []
                    ui.clear_list()
                    for i in range(len(names)):
                        if curr_det_count[i] > 0:
                            ui.update_item(f"{curr_det_count[i]} x {names[i]}")  
                            test_list.append(names[i])
                            ui.update_price("\u09F3" + str(curr_det_count[i]*int(price_list[i])))

                    #ui.update_total()
                    print(test_list)


            # Stream results
            if view_img:
                #cv2.imshow("Display", im0)
                print(im0.shape)
                ui.update_image(im0)
                cv2.waitKey(1)  # 1 millisecond
                
              

         


    print(f'Done. ({time.time() - t0:.3f}s)')



class helpPopup(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setGeometry((1920-658)//2, (1080-439)//2, 658, 439)
        Dialog.setStyleSheet("color: #ffffff; background-color: rgb(59, 62, 63);")
        Dialog.setWindowIcon(QtGui.QIcon('ui_elements/appicon.svg'))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, -1, 10, -1)
        self.verticalLayout.setSpacing(22)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Title = QtWidgets.QLabel(Dialog)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.verticalLayout.addWidget(self.Title)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setContentsMargins(250, 0, 250, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OkButton = QtWidgets.QPushButton(Dialog)
        self.OkButton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        self.OkButton.clicked.connect(Dialog.close)        
        
        self.OkButton.setObjectName("OkButton")
        self.horizontalLayout.addWidget(self.OkButton)
        self.horizontalLayout.setStretch(0, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Title.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:16pt; font-weight:600; color:#ffffff;\">How to Use</span></p></body></html>"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:12pt;\">To start billing, press the</span><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600;\"> &quot;START BILLING&quot;</span><span style=\" font-family:\'poppins\'; font-size:12pt;\"> Button or the</span><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600;\"> (S) Key</span><span style=\" font-family:\'poppins\'; font-size:12pt;\"> on your keyboard. While in billing mode, place items infront of the camera to add them to the list. </span></p><p><span style=\" font-family:\'poppins\'; font-size:12pt;\">To lock the current batch of items in the list press </span><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600;\">&quot;LOCK ITEMS&quot;</span><span style=\" font-family:\'poppins\'; font-size:12pt;\"> Button or the </span><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600;\">(Space) Key</span><span style=\" font-size:12pt;\"> on your keyboard to lock them in place. After that you can remove the current batch of products and bring in a new batch. </span></p><p><span style=\" font-size:12pt;\"><br/>Use the</span><span style=\" font-size:12pt; font-weight:600;\"> &quot;CLEAR LIST&quot;</span><span style=\" font-family:\'poppins\'; font-size:12pt;\"> Button or</span><span style=\" font-size:12pt; font-weight:600;\"> (X) Key</span><span style=\" font-size:12pt;\"> on your keyboard to clear the items on the list.</span></p><p><span style=\" font-size:12pt;\"><br/>After you are done billing your products, press the </span><span style=\" font-size:12pt; font-weight:600;\">&quot;STOP BILLING&quot;</span><span style=\" font-size:12pt;\"> Button or the</span><span style=\" font-size:12pt; font-weight:600;\"> (S) Key </span><span style=\" font-size:12pt;\">on your keyboard. You will be prompted if you want to generate a memo. Press yes to generate the memo.</span></p><p><span style=\" font-family:\'poppins\'; font-size:12pt;\"><br/></span></p></body></html>"))
        self.OkButton.setText(_translate("Dialog", "OK"))
        
class receipt_prompt(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Prompt")
        Dialog.setGeometry((1920-325)//2, (1080-138)//2, 325, 138)
        Dialog.setStyleSheet("color: #ffffff; background-color: rgb(59, 62, 63);")
        Dialog.setWindowIcon(QtGui.QIcon('ui_elements/appicon.svg'))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.yesButton = QtWidgets.QPushButton(Dialog)
        self.yesButton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
       
        self.yesButton.setObjectName("yesButton")
        self.yesButton.clicked.connect(lambda: self.prompt_reply(Dialog))
                
        self.gridLayout.addWidget(self.yesButton, 1, 0, 1, 1)
        self.noButton = QtWidgets.QPushButton(Dialog)
        self.noButton.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        self.noButton.setObjectName("noButton")
        self.noButton.clicked.connect(Dialog.close)
        self.gridLayout.addWidget(self.noButton, 1, 1, 1, 1)
        
        self.reply = 0
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Prompt"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:12pt;\">Do you want to generate a receipt ?</span></p></body></html>"))
        self.yesButton.setText(_translate("Dialog", "Yes"))
        self.noButton.setText(_translate("Dialog", "No"))

    def prompt_reply(self, Dialog):
        self.reply = 1
        Dialog.close()



class about_popup(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setGeometry((1920-650)//2, (1080-450)//2, 650, 450)
        Dialog.setStyleSheet("color: #ffffff; background-color: rgb(59, 62, 63);")
        Dialog.setWindowIcon(QtGui.QIcon('ui_elements/appicon.svg'))
        self.Title = QtWidgets.QLabel(Dialog)
        self.Title.setGeometry(QtCore.QRect(0, 20, 631, 25))
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(200, 80, 420, 500))
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 135, 151, 181))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("ui_elements/dulogo.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 400, 631, 29))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setContentsMargins(250, 0, 250, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OkButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.OkButton_2.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")

        self.OkButton_2.setObjectName("OkButton_2")
        self.horizontalLayout.addWidget(self.OkButton_2)
        self.horizontalLayout.setStretch(0, 1)
        self.OkButton_2.clicked.connect(Dialog.close)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Title.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:16pt; font-weight:600; color:#ffffff;\">About</span></p></body></html>"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"justify\"><span style=\" font-family:\'poppins\'; font-size:12pt;\">This is an Image recognition based billing system for supershops based on a state of the art object detection algorithm Yolov5. The application is made by -<br/><br/></span><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600; color:#5aff6d;\">Rusab Sarmun (SH-073-045)</span></p><p align=\"justify\"><span style=\" font-family:\'poppins\'; font-size:12pt; font-weight:600; color:#5aff6d;\">Fabiha Bushra (SK-073-070)</span><span style=\" font-family:\'poppins\'; font-size:12pt;\"><br/><br/>This project has been made as the final year project of BSc in DU EEE. </span></p></body></html>"))
        self.OkButton_2.setText(_translate("Dialog", "OK"))
        
        
        

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry((1920-1144)//2, (1080-851)//2, 1144, 851)
        MainWindow.setWindowIcon(QtGui.QIcon('ui_elements/appicon.svg'))
        
        palette = QPalette()
        palette.setColor(QPalette.Text, QtCore.Qt.white)     
        
        myFont=QtGui.QFont('Poppins')
        myFont.setBold(True)
        myFont.setPointSize(16)
        
        listfont=QtGui.QFont('Poppins')
        listfont.setBold(True)
        listfont.setPointSize(10)
        
        titleFont=QtGui.QFont('Poppins')
        titleFont.setBold(True)
        titleFont.setPointSize(30)
        
        self.button_flag = 0
        self.lock_flag = 0
        self.lock_pointer = 1
        self.reset_flag = 0
        
        MainWindow.setStyleSheet("""
        QMainWindow {
            background-color: rgb(37, 39, 40);
        }
        
        QMenuBar{
            background-color: rgb(30, 32, 33);
        }
        
        QMenuBar::item::selected {
            background-color: #000000;
        }
        
        QMenu {
            background-color: #3b3e3f;
        }
        
        QMenu::item::selected {
            background-color: #000000;
        }
        
        QMenu::item::disabled {
            color: #888888;
        }
                                 
                                 """)
                                 
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.camFeed = QtWidgets.QLabel(self.centralwidget)
        self.camFeed.setGeometry(QtCore.QRect(50, 170, 640, 480))
        self.camFeed.setText("")
        self.camFeed.setPixmap(QtGui.QPixmap("ui_elements/connectingcam.jpg"))  
        self.camFeed.setObjectName("camFeed")
        
        self.labelCam = QtWidgets.QLabel(self.centralwidget)
        self.labelCam.setGeometry(QtCore.QRect(50, 125, 151, 41))
        self.labelCam.setObjectName("labelCam")
        self.labelCam.setFont(myFont)
        self.labelCam.setPalette(palette)
        self.labelCam.setText("Camera Feed")
        self.labelCam.setStyleSheet("color:#ffffff")
        
        
        
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(720, 170, 241, 481))
        self.listWidget.setStyleSheet("color: #ffffff; background-color: rgb(59, 62, 63);")
        self.listWidget.setObjectName("listWidget")
        self.listWidget.setFont(listfont)
        
        self.listPrice = QtWidgets.QListWidget(self.centralwidget)
        self.listPrice.setGeometry(QtCore.QRect(970, 170, 121, 481))
        self.listPrice.setStyleSheet("color: #ffffff; background-color: rgb(59, 62, 63);")
        self.listPrice.setObjectName("listPrice")
        self.listPrice.setFont(listfont)
        
        self.listLabel = QtWidgets.QLabel(self.centralwidget)
        self.listLabel.setGeometry(QtCore.QRect(720, 125, 151, 41))
        self.listLabel.setObjectName("listLabel")
        self.listLabel.setText("Product List")
        self.listLabel.setFont(myFont)
        self.listLabel.setStyleSheet("color:#ffffff")
        
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(275, 40, 651, 71))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.Titlelayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.Titlelayout.setContentsMargins(0, 0, 0, 0)
        self.Titlelayout.setObjectName("Titlelayout")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 20, 631, 111))
        self.label_4.setText("CV Based Billing System")
        self.label_4.setObjectName("label_4")
        self.label_4.setFont(titleFont)
        self.label_4.setStyleSheet("color:#5aff6d")
        self.Titlelayout.addWidget(self.label_4)

        self.versionLabel = QtWidgets.QLabel(self.centralwidget)
        self.versionLabel.setGeometry(QtCore.QRect(760, 60, 151, 41))
        self.versionLabel.setObjectName("versionLabel")
        self.versionLabel.setText("v.1.1")
        self.versionLabel.setFont(myFont)
        self.versionLabel.setStyleSheet("color:#ffffff")
        self.Titlelayout.addWidget(self.versionLabel)
        
        self.billingButton = QtWidgets.QPushButton(self.centralwidget)
        self.billingButton.setGeometry(QtCore.QRect(80, 670, 181, 61))
        self.billingButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        
        self.icon = QtGui.QIcon()
        self.icon.addPixmap(QtGui.QPixmap("ui_elements/startbilling.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon4 =  QtGui.QIcon()
        self.icon4.addPixmap(QtGui.QPixmap("ui_elements/stopbilling.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        
        self.billingButton.setIcon(self.icon)
        self.billingButton.setIconSize(QtCore.QSize(27, 27))
        self.billingButton.setObjectName("billingButton")
        self.lockButton = QtWidgets.QPushButton(self.centralwidget)
        self.lockButton.setGeometry(QtCore.QRect(280, 670, 181, 61))
        self.lockButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("ui_elements/lockitems.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lockButton.setIcon(icon1)
        self.lockButton.setIconSize(QtCore.QSize(27, 27))
        self.lockButton.setObjectName("lockButton")
        self.resetButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(480, 670, 181, 61))
        self.resetButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("ui_elements/clearbill.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.resetButton.setIcon(icon2)
        self.resetButton.setIconSize(QtCore.QSize(27, 27))
        self.resetButton.setObjectName("resetButton")
        
        
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(720, 670, 371, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.totalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.totalLayout.setContentsMargins(0, 0, 0, 0)
        self.totalLayout.setObjectName("totalLayout")
        
        
        
        self.labelTotal = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.labelTotal = QtWidgets.QLabel(self.centralwidget)
        self.labelTotal.setAlignment(QtCore.Qt.AlignCenter)
        self.labelTotal.setGeometry(QtCore.QRect(760, 670, 151, 61))
        self.labelTotal.setText("Total: " +"\u09F3" +"0")
        self.labelTotal.setFont(myFont)
        self.labelTotal.setStyleSheet("color:#ffffff; border: 2px solid white;")
        self.labelTotal.setObjectName("labelTotal")
        self.totalLayout.addWidget(self.labelTotal)
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1144, 21))
        
       
       
        self.menubar.setPalette(palette)
        self.menubar.setStyleSheet("color:#ffffff;")
        self.menubar.setNativeMenuBar(True)
        self.menubar.setObjectName("menubar")
        self.menuFIle = QtWidgets.QMenu(self.menubar)
        self.menuFIle.setObjectName("menuFIle")
        self.menuFIle.setStyleSheet("background-color:#3b3e3f")
        
        self.menuAction = QtWidgets.QMenu(self.menubar)
        self.menuAction.setObjectName("menuAction")
        self.menuAction.setStyleSheet("background-color:#3b3e3f")
        
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuHelp.setStyleSheet("background-color:#3b3e3f")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionOpen_Price_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_Price_File.setObjectName("actionOpen_Price_File")
        
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(MainWindow.close)
        
        self.actionStart_Billing = QtWidgets.QAction(MainWindow)
        self.actionStart_Billing.setObjectName("actionStart_Billing")
        
        
        self.actionStop_Billing = QtWidgets.QAction(MainWindow)
        self.actionStop_Billing.setObjectName("actionStop_Billing")
        
        
        self.actionLock_Items = QtWidgets.QAction(MainWindow)
        self.actionLock_Items.setObjectName("actionLock_Items")
        
        self.actionClear_List = QtWidgets.QAction(MainWindow)
        self.actionClear_List.setObjectName("actionClear_List")
        
        self.actionHow_to_use = QtWidgets.QAction(MainWindow)
        self.actionHow_to_use.setObjectName("actionHow_to_use")
        self.actionHow_to_use.triggered.connect(self.open_help)
        
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.triggered.connect(self.open_about)
        
        self.menuFIle.addAction(self.actionOpen_Price_File)
        self.menuFIle.addAction(self.actionExit)
        
        self.actionStart_Billing.triggered.connect(self.button_clicked)
        self.actionStop_Billing.triggered.connect(self.button_clicked)
        self.actionLock_Items.triggered.connect(self.lock_clicked)
        self.actionClear_List.triggered.connect(self.reset_list)
        self.actionStop_Billing.setDisabled(True)
        
        self.menuAction.addAction(self.actionStart_Billing)
        self.menuAction.addAction(self.actionStop_Billing)
        self.menuAction.addAction(self.actionLock_Items)
        self.menuAction.addAction(self.actionClear_List)
        
        self.menuHelp.addAction(self.actionHow_to_use)
        self.menuHelp.addAction(self.actionAbout)
        
        self.menubar.addAction(self.menuFIle.menuAction())
        self.menubar.addAction(self.menuAction.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        
        self.lockButton.clicked.connect(self.lock_clicked)
        self.billingButton.clicked.connect(self.button_clicked)
        self.resetButton.clicked.connect(self.reset_list)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CV Based Billing System v.1.1"))
        self.billingButton.setText(_translate("MainWindow", " START BILLING"))
        self.lockButton.setText(_translate("MainWindow", "  LOCK ITEMS"))
        self.resetButton.setText(_translate("MainWindow", "  CLEAR LIST"))
        #self.labelTotal.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:16pt; font-weight:600; color:#ffffff;\">Total: </span></p></body></html>"))
        self.menuFIle.setTitle(_translate("MainWindow", "FIle"))
        self.menuAction.setTitle(_translate("MainWindow", "Action"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_Price_File.setText(_translate("MainWindow", "Open Price File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionStart_Billing.setText(_translate("MainWindow", "Start Billing"))
        self.actionStart_Billing.setShortcut(_translate("MainWindow", "S"))
        self.actionStop_Billing.setText(_translate("MainWindow", "Stop Billing"))
        self.actionStop_Billing.setShortcut(_translate("MainWindow", "S"))
        self.actionLock_Items.setText(_translate("MainWindow", "Lock Items"))
        self.actionLock_Items.setShortcut(_translate("MainWindow", "Space"))
        self.actionClear_List.setText(_translate("MainWindow", "Clear List"))
        self.actionClear_List.setShortcut(_translate("MainWindow", "X"))
        self.actionHow_to_use.setText(_translate("MainWindow", "How to use"))
        self.actionAbout.setText(_translate("MainWindow", "About"))

    def close_window(self):
        self.exit_app()
    
        
    def update_image(self, cv_img):
        qtimg = self.convert_cv_qt(cv_img)
        self.camFeed.setPixmap(qtimg)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(720, 1280, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
            
    def update_item(self, item):
        self.listWidget.addItem(item)
    
    def update_price(self, price):
        item = QtWidgets.QListWidgetItem(price)
        item.setTextAlignment(Qt.AlignHCenter)
        self.listPrice.addItem(item)
        
    def clear_list(self):
        current = self.listWidget.count()
        print(f"current:{current}")
        print(f"pointer:{self.lock_pointer}")

        start_row = int(self.lock_pointer) - 1
        for row in range(start_row, current):    
            self.listWidget.takeItem(start_row)
            self.listPrice.takeItem(start_row)
            print(row)

        print(f"new current:{self.listWidget.count()}")
        
    #clear list button actually resets list       
    def reset_list(self):
        self.listPrice.clear()
        self.listWidget.clear()
        self.reset_flag = 1
        self.lock_pointer = 1
        self.lock_flag = 0
        self.labelTotal.setText("Total: \u09F3" + "0")
        self.labelTotal.adjustSize()
        self.colorize(Qt.black)
    
    def update_total(self):
        total = 0
        for i in range(self.listPrice.count()):
           x = self.listPrice.item(i).text()
           x = x[1:]
           total += int(x)
        
        self.labelTotal.setText("Total: \u09F3" + str(total))
        #self.labelTotal.adjustSize()
        self.colorize(Qt.black)
       
        
        
    def button_clicked(self):
        self.button_flag = not self.button_flag
        _translate = QtCore.QCoreApplication.translate
        if self.button_flag:
            self.billingButton.setIcon(self.icon4)
            self.billingButton.setText(_translate("MainWindow", " STOP BILLING"))
            self.colorize(Qt.black)
            self.billingButton.setStyleSheet("background-color: rgb(255, 92, 92);\n"
"font: 500 12pt \"Poppins\";")
            self.actionStart_Billing.setDisabled(True)
            self.actionStop_Billing.setDisabled(False)
           
            
        else:
            self.billingButton.setIcon(self.icon)
            self.billingButton.setText(_translate("MainWindow", " START BILLING"))
            self.clear_list()
            self.update_total()
            self.colorize(Qt.darkGreen)
            self.billingButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
            self.actionStop_Billing.setDisabled(True)
            self.actionStart_Billing.setDisabled(False)
            self.open_receipt_prompt()
            self.reset_list()
           
       
            
            
    def lock_clicked(self):
        self.lock_flag = 1
           
    def color_locked(self):
        
        self.lock_pointer = self.listWidget.count() + 1
        print(self.lock_pointer)
        for i in range(self.lock_pointer - 1):
           x = self.listWidget.item(i)
           y = self.listPrice.item(i)
           x.setBackground(QColor("#7fc97f"))
           y.setBackground(QColor("#7fc97f"))
           x.setForeground(QColor("#252728"))
           y.setForeground(QColor("#252728"))
           
    def colorize(self, color):
        color_effect = QGraphicsColorizeEffect()
        color_effect.setColor(color)
        self.labelTotal.setGraphicsEffect(color_effect)
        
    def open_help(self):
        helpWindow = QtWidgets.QDialog()
        helpui = helpPopup()
        helpui.setupUi(helpWindow)
        rw = helpWindow.exec()
        
    def open_about(self):
        aboutWindow = QtWidgets.QDialog()
        helpui = about_popup()
        helpui.setupUi(aboutWindow)
        rw = aboutWindow.exec()
        
    def open_receipt_prompt(self):
        promptWindow = QtWidgets.QDialog()
        helpui = receipt_prompt()
        helpui.setupUi(promptWindow)
        rw = promptWindow.exec()
        if helpui.reply > 0:
            self.generate_reciept()
    
    def generate_reciept(self):
        receipt_no = str(randint(0000000, 9999999))
        file1 = open("receipts/" + receipt_no  + ".txt","w+")
        file1.write("Order\n")
        file1.write("Receipt #" + receipt_no  + "\n\n")
        file1.write("Shop Name: Sigma Supershop\n")
        date = str(datetime.datetime.now())
        file1.write("Date Time: "+ date[:-7] + "\n")
        file1.write("--------------------------------------------------\n")
        qtDes = "QTY   Description"
        spaces = " " * (40-len(qtDes))
        file1.write(qtDes + spaces + "Price\n")
        file1.write("--------------------------------------------------\n")
        price = []
        for i in range(self.lock_pointer - 1):
           x = self.listWidget.item(i).text()
           y = self.listPrice.item(i).text()
           qty = x[0]
           desc = x[4:]
           price.append(int(y[1:]))
           qtDes = qty + "     " + desc
           spaces = " " * (40-len(qtDes))
           file1.write(qtDes + spaces + "Tk." + y[1:] + "\n")
        file1.write("--------------------------------------------------\n")     
        spaces = " " * 33
        file1.write("Total: " + spaces + "Tk." + str(sum(price)) + "\n")
        file1.write("--------------------------------------------------\n")
        file1.write("Thank You For Shopping\n")
        file1.close()
        startfile(r"receipts\\" +receipt_no + ".txt",'open')
 

        
         
          

if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)    
    
    MainWindow.show()
         
    run()

    sys.exit(app.exec_())