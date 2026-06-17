import os
import pathlib
import sys

import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from natsort import natsorted
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
import pyqtgraph as pg
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow


mouse_list = ['RD121']  # RD109, RD116, RD119, RD121
mouse_session_list = []
rt_p = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Robin_Dard\\data'
for mouse in mouse_list:
    for sess in os.listdir(os.path.join(rt_p, mouse)):
        if not os.path.exists(os.path.join(rt_p, mouse, sess, 'suite2p')):
            continue
        if not os.path.exists(os.path.join(rt_p, mouse, sess, 'projection_neurons')):
            mouse_session_list.append(sess)
mouse_session_list = natsorted(mouse_session_list)

class ProjectionsGUI(QMainWindow):  

    def __init__(self):

        super(ProjectionsGUI, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setWindowTitle("Classify projection neurons depending on dual CTB-injections")
        self.setGeometry(50, 50, 1600, 500)
        self.home()


    def home(self):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.server_dir = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Robin_Dard\\data'
        mouse_list_unique = list(set(mouse_list))
        mouse_session_list_unique = natsorted(list(set(mouse_session_list)))

        cwidget = QtWidgets.QWidget()
        self.l0 =  QtWidgets.QHBoxLayout()
        self.buttons =  QtWidgets.QVBoxLayout()

        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        self.functional_win = pg.GraphicsLayoutWidget()
        self.redWin = pg.GraphicsLayoutWidget()
        self.farRedWin =  pg.GraphicsLayoutWidget()

        self.functional = pg.ImageItem(name='ROIs')
        self.red = pg.ImageItem(name='Red')
        self.far_red = pg.ImageItem(name='FarRed')

        self.l0.addLayout(self.buttons)
        self.l0.addWidget(self.functional_win)
        self.l0.addWidget(self.redWin)
        self.l0.addWidget(self.farRedWin)

        self.label_mouse_name = QLabel(self)
        self.label_mouse_name.setText('Choose mouse name:')
        self.label_mouse_name.adjustSize()

        self.comboBoxMouseName = QComboBox(self)
        self.comboBoxMouseName.addItem('')

        self.comboBoxMouseName.addItems(mouse_list_unique)
        self.comboBoxMouseName.activated.connect(self.load_data)

        self.label_mouse_name = QLabel(self)
        self.label_mouse_name.setText('Choose mouse session name:')
        self.label_mouse_name.adjustSize()
        self.comboBoxMouseSessionName = QComboBox(self)
        self.comboBoxMouseSessionName.addItem('')

        self.comboBoxMouseSessionName.addItems(mouse_session_list_unique)
        self.comboBoxMouseSessionName.activated.connect(self.load_data)

        self.RButton = QtWidgets.QPushButton("Red")
        self.RButton.clicked.connect(self.clickedRedButton)

        self.FRButton = QtWidgets.QPushButton("FarRed")
        self.FRButton.clicked.connect(self.clickedFarRedButton)

        self.UNButton = QtWidgets.QPushButton("UN")
        self.UNButton.clicked.connect(self.clickedUNButton)

        self.CloseButton = QtWidgets.QPushButton("Close")
        self.CloseButton.clicked.connect(self.close_application_immediately)

        self.labelRatio = QLabel(self)
        self.labelRatio.setText('Ratio FR/R = ')

        self.labelSsiGR = QLabel(self)
        self.labelSsiGR.setText('SSI G vs R = ')

        self.labelSsiRFR = QLabel(self)
        self.labelSsiRFR.setText('SSI R vs FR = ')

        self.ROI_button = QtWidgets.QPushButton("ROI")
        self.ROI_button.clicked.connect(self.clickedROIButton)

        self.buttons.addWidget(self.label_mouse_name)
        self.buttons.addWidget(self.comboBoxMouseName)
        self.buttons.addWidget(self.comboBoxMouseSessionName)
        self.buttons.addWidget(self.RButton)
        self.buttons.addWidget(self.FRButton)
        self.buttons.addWidget(self.UNButton)
        self.buttons.addWidget(self.labelRatio)
        self.buttons.addWidget(self.labelSsiGR)
        self.buttons.addWidget(self.labelSsiRFR)
        self.buttons.addWidget(self.ROI_button)

        self.buttons.addStretch(1)

        self.buttons.addWidget(self.CloseButton)

        self.show()


    def close_application_immediately(self):

        #exit application immediately
        QApplication.quit()


    def load_data(self):

        choiceMouse = (str(self.comboBoxMouseName.currentText()))
        choiceSession = (str(self.comboBoxMouseSessionName.currentText()))

        if choiceMouse == '' or choiceSession == '':
            self.sessionkey = {}
            self.sessionnamekey = {}

        else:
            self.sessionkey = {}
            self.sessionnamekey = {}
            self.sessionkey['mouse_name'] = choiceMouse
            self.sessionnamekey['session_name'] = choiceSession
            print(self.sessionkey)
            print(self.sessionnamekey)

            self.server_dir = pathlib.Path(os.path.join(self.server_dir, choiceMouse, choiceSession))

            if not(self.server_dir.is_dir()):
                print('Folder created!')
                os.makedirs(self.server_dir)

            suite2p_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Robin_Dard\\data\\{choiceMouse}\\{choiceSession}\\suite2p\\plane0'
            self.stat = np.load(os.path.join(suite2p_folder, 'stat.npy'), allow_pickle=True)
            self.ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).item()
            self.iscell = np.load(os.path.join(suite2p_folder, 'iscell.npy'), allow_pickle=True)
            self.iscell = self.iscell[:,0]

            for i, s in enumerate(self.stat):
                if 'inmerge' in s:
                    if not((s['inmerge'] == 0.0) or (s['inmerge'] == -1.0)):
                        self.iscell[i] = 0.0
            self.stat = self.stat[self.iscell==1]
            self.rois = range(self.stat.shape[0])

            self.load_images()


    def load_images(self):

        (self.ImgRed, self.ImgFarRed, self.ImgGreen,
         self.ImgRedFull,self.ImgFarRedFull, self.ImgGreenFull) = self.registerImages()

        self.calculateRatiosROIs()

        #check if files already exist

        self.CheckRedFile =  pathlib.Path(os.path.join(self.server_dir, 'projection_neurons', "RedRois.npy"))
        self.CheckFarRedFile =  pathlib.Path(os.path.join(self.server_dir, 'projection_neurons', "FarRedRois.npy"))
        self.CheckUNFile =  pathlib.Path(os.path.join(self.server_dir, 'projection_neurons', "UNRois.npy"))

        if not(self.CheckRedFile.is_file()) or not(self.CheckFarRedFile.is_file()) or not(self.CheckUNFile.is_file()):
            (self.Far_red, self.Red, self.UN) =  self.predefineRois()

        if self.CheckRedFile.is_file():
            self.Red = np.load(self.CheckRedFile, allow_pickle=True)

        if self.CheckFarRedFile.is_file():
            self.Far_red = np.load(self.CheckFarRedFile, allow_pickle=True)

        if self.CheckUNFile.is_file():
            self.UN = np.load(self.CheckUNFile, allow_pickle=True)

        (self.ROIsImage, self.iROIs, self.maximg,
         self.H, self.S, self.xrange, self.yrange, self.Lx, self.Ly) = self.makeRoisImage()

        self.plot0 = pg.PlotItem(name='Cells',title = 'ROIs Image')
        self.plot1 = pg.PlotItem(name='Red', title = 'Red Image')
        self.plot2 = pg.PlotItem(name='FarRedImage', title = 'FarRed Image')

        self.plot0.setAspectLocked()
        self.plot0.hideAxis('bottom')
        self.plot0.hideAxis('left')

        self.plot1.setAspectLocked()
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('left')

        self.plot2.setAspectLocked()
        self.plot2.hideAxis('bottom')
        self.plot2.hideAxis('left')

        self.functional_win.addItem(self.plot0, 0,0)
        self.plot0.addItem(self.functional)
        self.functional.setImage(self.ROIsImage)
        self.functional.setLevels([0, 0.5])

        self.redWin.addItem(self.plot1, 0,0)
        self.plot1.addItem(self.red)
        self.red.setImage(self.ImgRed)

        self.farRedWin.addItem(self.plot2, 0,0)
        self.plot2.addItem(self.far_red)
        self.far_red.setImage(self.ImgFarRed)

        #if the mouse is clicked proceed with roi selection
        self.plot0.scene().sigMouseClicked.connect(self.mouseClickedGreen)

        #if the mouse is clicked proceed with roi selection
        self.plot1.scene().sigMouseClicked.connect(self.mouseClickedRed)

        #if the mouse is clicked proceed with roi selection
        self.plot2.scene().sigMouseClicked.connect(self.mouseClickedFarRed)

        self.show()


    def ROIindex(self):

        #find the index of the clicked ROI

        #check if the clicked point is out of borders and if not select it

        posNeg = (self.x < 0) or (self.y < 0)
        posOut = (self.x >= self.ROIsImage.shape[0]) or (self.y >= self.ROIsImage.shape[1])
        if posNeg or posOut:
            pass
            roiSelected = -1
        else:
            print('x: ' + str(self.x))
            print('y: ' + str(self.y))
            roiSelected = int(self.iROIs[self.x,self.y])
            print('ROI selected: ' + str(roiSelected))

        return roiSelected


    def circle(self,med,r):

        """ returns pixels of circle with radius 2x radius of cell (r) """
        theta = np.linspace(0.0,2*np.pi,100)
        x = r*2 * np.cos(theta) + med[0]
        y = r*2 * np.sin(theta) + med[1]
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        return x,y


    def newROIimage(self):

        titleNew =  'ROIs Image, ' + 'ROI selected: ' + str(int(self.roiSelected))
        self.plot0.setTitle(titleNew)
        self.sessionkey.update({'roi_number': self.roiSelected})
        
        stats = self.stat
        stats = stats[self.sessionkey['roi_number']]

        #add a white color to the selected ROI
        roix = stats['xpix'].flatten()
        roiy = stats['ypix'].flatten()

        #make new green ROIs image
        HN = np.copy(self.H)
        SN = np.copy(self.S)
        VN = np.copy(self.maximg)

        HN[roix, roiy] = 0
        SN[roix, roiy] = 0
        VN[roix, roiy] = 1

        center = [int(np.round(np.abs(np.max(roix)-np.min(roix))/2 + np.min(roix))),
                  int(np.round(np.abs(np.max(roiy)-np.min(roiy))/2 + np.min(roiy)))]

        r = int(stats['radius'])

        (self.xc,self.yc) = self.circle(center,r)

        HN[self.xc, self.yc] = 0
        SN[self.xc, self.yc] = 1
        VN[self.xc, self.yc] = 1

        pixN = np.concatenate(((HN[:, :, np.newaxis]),
                                  SN[:, :, np.newaxis],
                                  VN[:, :, np.newaxis]), axis=-1)
        pixN = hsv_to_rgb(pixN)

        ROIsImageN = pixN[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]

        #make red new image with circle

        RedImage = np.copy(self.ImgRedFull)

        HRN = np.zeros([self.Lx,self.Ly])
        SRN = np.zeros([self.Lx,self.Ly])
        VRN = self.imageNorm(RedImage)

        HRN[self.xc, self.yc] = 0
        SRN[self.xc, self.yc] = 1
        VRN[self.xc, self.yc] = 1

        pixRN = np.concatenate(((HRN[:, :, np.newaxis]),
                                 SRN[:, :, np.newaxis],
                                 VRN[:, :, np.newaxis]), axis=-1)
        pixRN = hsv_to_rgb(pixRN)

        RedN = pixRN[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]

        #make far red new image with circle

        FarRedImage = np.copy(self.ImgFarRedFull)

        HFRN = np.zeros([self.Lx,self.Ly])
        SFRN = np.zeros([self.Lx,self.Ly])
        VFRN = self.imageNorm(FarRedImage)


        HFRN[self.xc, self.yc] = 0
        SFRN[self.xc, self.yc] = 1
        VFRN[self.xc, self.yc] = 1

        pixFRN = np.concatenate(((HFRN[:, :, np.newaxis]),
                                  SFRN[:, :, np.newaxis],
                                  VFRN[:, :, np.newaxis]), axis=-1)
        pixFRN = hsv_to_rgb(pixFRN)

        FarRedN = pixFRN[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]


        return(ROIsImageN,RedN,FarRedN)



    def mouseClickedGreen(self,evt):

        #get mouse coordinates in relation to image space
        pos = self.functional.mapFromScene(evt.scenePos())

        #round to get pixel size
        self.x = int(pos.y())
        self.y = int(pos.x())

        #get the index of the selected ROI after clicking
        self.roiSelected = self.ROIindex()

        if self.roiSelected != -1:

            (self.ROIsImageN, self.RedN, self.FarRedN) = self.newROIimage()

            self.functional.setImage(self.ROIsImageN)
            self.red.setImage(self.RedN)
            self.far_red.setImage(self.FarRedN)
            self.labelRatio.setText('Ratio FR/R = ' + str(np.round(self.Ratio[self.roiSelected],3)))
            self.labelSsiGR.setText('SSI G vs R = '+ str(np.round(self.ssiGR[self.roiSelected],3)))
            self.labelSsiRFR.setText('SSI R vs FR = '+ str(np.round(self.ssiRFR[self.roiSelected])))


    def clickedRedButton(self):

        if "roi_number" in self.sessionkey:

            print("ROI is selected and changed to red")

            stats = self.stat
            stats = stats[self.sessionkey['roi_number']]

            #add a white color to the selected ROI
            roix = stats['xpix'].flatten()
            roiy = stats['ypix'].flatten()

            #make new  ROIs image with changed color

            self.H[roix, roiy] = 0
            self.S[roix, roiy] = 1
            self.maximg[roix, roiy] = 1

            # make the circle around the new ROI

            center = [int(np.round(np.abs(np.max(roix)-np.min(roix))/2 + np.min(roix))),
                  int(np.round(np.abs(np.max(roiy)-np.min(roiy))/2 + np.min(roiy)))]

            r = int(stats['radius'])

            (xc,yc) = self.circle(center,r)

            HTemp = np.copy(self.H)
            STemp= np.copy(self.S)
            VTemp = np.copy(self.maximg)

            HTemp[xc, yc] = 0
            STemp[xc, yc] = 1
            VTemp[xc, yc] = 1

            pixAfterRedClick = np.concatenate(((HTemp[:, :, np.newaxis]),
                                  STemp[:, :, np.newaxis],
                                  VTemp[:, :, np.newaxis]), axis=-1)


            pixAfterRedClick  = hsv_to_rgb(pixAfterRedClick)

            ROIsImageAfterRedClick = pixAfterRedClick[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]

            self.functional.setImage(ROIsImageAfterRedClick)

            if self.sessionkey["roi_number"] in self.Far_red:

                IndexInFarRed = [i for (i, fr) in enumerate(self.Far_red) if (fr == self.sessionkey["roi_number"]) ]
                self.Far_red = np.delete(self.Far_red, IndexInFarRed[0])
                print('Deleted In FarRed at ' + str(IndexInFarRed[0]))

            elif self.sessionkey["roi_number"] in self.UN:

                IndexInUN = [i for (i, un) in enumerate(self.UN) if (un == self.sessionkey["roi_number"])]
                self.UN = np.delete(self.UN, IndexInUN[0])
                print('Deleted in UN at ' + str(IndexInUN[0]))

            # and file to the read and save the file

            if not(self.sessionkey["roi_number"] in self.Red) :

                self.Red = np.append(self.Red,int(self.sessionkey["roi_number"]))

            np.save(self.CheckFarRedFile,self.Far_red)
            np.save(self.CheckRedFile,self.Red)
            np.save(self.CheckUNFile,self.UN)

            print('File Saved!')


    def clickedFarRedButton(self):

        if "roi_number" in self.sessionkey:

            print("ROI is selected and changed to far red")

            stats = self.stat

            stats = stats[self.sessionkey['roi_number']]

            #add a white color to the selected ROI
            roix = stats['xpix'].flatten()
            roiy = stats['ypix'].flatten()

            #make new  ROIs image with changed color

            self.H[roix, roiy] = 0.67
            self.S[roix, roiy] = 1
            self.maximg[roix, roiy] = 1

            # make the circle around the new ROI

            center = [int(np.round(np.abs(np.max(roix)-np.min(roix))/2 + np.min(roix))),
                  int(np.round(np.abs(np.max(roiy)-np.min(roiy))/2 + np.min(roiy)))]

            r = int(stats['radius'])

            (xc,yc) = self.circle(center,r)

            HTemp = np.copy(self.H)
            STemp= np.copy(self.S)
            VTemp = np.copy(self.maximg)

            HTemp[xc, yc] = 0
            STemp[xc, yc] = 1
            VTemp[xc, yc] = 1

            pixAfterFarRedClick = np.concatenate(((HTemp[:, :, np.newaxis]),
                                  STemp[:, :, np.newaxis],
                                  VTemp[:, :, np.newaxis]), axis=-1)


            pixAfterFarRedClick  = hsv_to_rgb(pixAfterFarRedClick)

            ROIsImageAfterFarRedClick = pixAfterFarRedClick[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]

            self.functional.setImage(ROIsImageAfterFarRedClick)


            if self.sessionkey["roi_number"] in self.Red:

                IndexInRed = [i for (i, r) in enumerate(self.Red) if (r == self.sessionkey["roi_number"]) ]
                self.Red = np.delete(self.Red, IndexInRed[0])
                print('Deleted In Red at ' + str(IndexInRed[0]))

            elif self.sessionkey["roi_number"] in self.UN:

                IndexInUN = [i for (i, un) in enumerate(self.UN) if (un == self.sessionkey["roi_number"])]
                self.UN = np.delete(self.UN, IndexInUN[0])
                print('Deleted in UN at ' + str(IndexInUN[0]))


            # and file to the read and save the file

            if not(self.sessionkey["roi_number"] in self.Far_red) :

                self.Far_red = np.append(self.Far_red,int(self.sessionkey["roi_number"]))

            np.save(self.CheckFarRedFile,self.Far_red)
            np.save(self.CheckRedFile,self.Red)
            np.save(self.CheckUNFile,self.UN)

            print('File Saved!')


    def clickedUNButton(self):

        if "roi_number" in self.sessionkey:

            print("ROI is selected and changed to unidentified")
            stats = self.stat
            stats = stats[self.sessionkey['roi_number']]

            #add a white color to the selected ROI
            roix = stats['xpix'].flatten()
            roiy = stats['ypix'].flatten()

            #make new  ROIs image with changed color

            self.H[roix, roiy] = 0.33
            self.S[roix, roiy] = 1
            self.maximg[roix, roiy] = 1

            # make the circle around the new ROI

            center = [int(np.round(np.abs(np.max(roix)-np.min(roix))/2 + np.min(roix))),
                  int(np.round(np.abs(np.max(roiy)-np.min(roiy))/2 + np.min(roiy)))]

            r = int(stats['radius'])

            (xc,yc) = self.circle(center,r)

            HTemp = np.copy(self.H)
            STemp= np.copy(self.S)
            VTemp = np.copy(self.maximg)

            HTemp[xc, yc] = 0
            STemp[xc, yc] = 1
            VTemp[xc, yc] = 1

            pixAfterUNClick = np.concatenate(((HTemp[:, :, np.newaxis]),
                                  STemp[:, :, np.newaxis],
                                  VTemp[:, :, np.newaxis]), axis=-1)


            pixAfterUNClick  = hsv_to_rgb(pixAfterUNClick)

            ROIsImageAfterUNClick = pixAfterUNClick[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1],:]

            self.functional.setImage(ROIsImageAfterUNClick)


            if self.sessionkey["roi_number"] in self.Red:

                IndexInRed = [i for (i, r) in enumerate(self.Red) if (r == self.sessionkey["roi_number"]) ]
                self.Red = np.delete(self.Red, IndexInRed[0])
                print('Deleted In Red at ' + str(IndexInRed[0]))

            elif self.sessionkey["roi_number"] in self.Far_red:

                IndexInFarRed = [i for (i, fr) in enumerate(self.Far_red) if (fr == self.sessionkey["roi_number"])]
                self.Far_red = np.delete(self.Far_red, IndexInFarRed[0])
                print('Deleted in Far Red at ' + str(IndexInFarRed[0]))



            # and file to the read and save the file

            if not(self.sessionkey["roi_number"] in self.UN) :

                self.UN = np.append(self.UN,int(self.sessionkey["roi_number"]))

            np.save(self.CheckFarRedFile,self.Far_red)
            np.save(self.CheckRedFile,self.Red)
            np.save(self.CheckUNFile,self.UN)

            print('File Saved!')


    def clickedROIButton(self):

        if 'hide_ROIS' not in self.sessionkey.keys():
           self.sessionkey['hide_ROIs'] = False

        if self.sessionkey['hide_ROIs'] == False:
            self.functional.setImage(self.ImgGreen)
            self.sessionkey['hide_ROIs'] = True
            return


        


        

    def computeGreenImage(self,x,y):

        center = [x,y]

        r = 7

        (xc,yc) = self.circle(center,r)

        HNG = np.copy(self.H[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]])
        SNG = np.copy(self.S[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]])
        VNG = np.copy(self.maximg[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]])


        CoordinL = [(xL,yL) for (xL,yL) in zip(xc,yc) if (xL < HNG.shape[0] and yL < HNG.shape[1])]

        xf = []
        yf = []

        for p in CoordinL:

            xf.append(p[0])
            yf.append(p[1])

        HNG[xf,yf] = 0
        SNG[xf,yf] = 1
        VNG[xf,yf] = 1

        pixNG = np.concatenate(((HNG[:, :, np.newaxis]),
                                  SNG[:, :, np.newaxis],
                                  VNG[:, :, np.newaxis]), axis=-1)
        pixNG= hsv_to_rgb(pixNG)

        return pixNG



    def computeRedImage(self,x,y):

        center = [x,y]

        r = 7

        (xc,yc) = self.circle(center,r)

        HNR = np.zeros([self.Lx,self.Ly])
        HNR = HNR[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]]
        SNR = np.zeros([self.Lx,self.Ly])
        SNR = SNR[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]]
        VNR = np.copy(self.ImgRed)

        CoordinL = [(xL,yL) for (xL,yL) in zip(xc,yc) if (xL < HNR.shape[0] and yL < HNR.shape[1])]

        xf = []
        yf = []

        for p in CoordinL:

            xf.append(p[0])
            yf.append(p[1])

        HNR[xf,yf] = 0
        SNR[xf,yf] = 1
        VNR[xf,yf] = 1

        pixNR = np.concatenate(((HNR[:, :, np.newaxis]),
                                  SNR[:, :, np.newaxis],
                                  VNR[:, :, np.newaxis]), axis=-1)
        pixNR = hsv_to_rgb(pixNR)

        return pixNR


    def computeFarRedImage(self,x,y):

        center = [x,y]

        r = 7

        (xc,yc) = self.circle(center,r)

        HNFR = np.zeros([self.Lx,self.Ly])
        HNFR =  HNFR[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]]
        SNFR = np.zeros([self.Lx,self.Ly])
        SNFR =  SNFR[self.xrange[0]:self.xrange[1],self.yrange[0]:self.yrange[1]]
        VNFR = np.copy(self.ImgFarRed)

        CoordinL = [(xL,yL) for (xL,yL) in zip(xc,yc) if (xL < HNFR.shape[0] and yL < HNFR.shape[1])]

        xf = []
        yf = []

        for p in CoordinL:

            xf.append(p[0])
            yf.append(p[1])


        HNFR[xf,yf] = 0
        SNFR[xf,yf] = 1
        VNFR[xf,yf] = 1

        pixNFR = np.concatenate(((HNFR[:, :, np.newaxis]),
                                  SNFR[:, :, np.newaxis],
                                  VNFR[:, :, np.newaxis]), axis=-1)
        pixNFR= hsv_to_rgb(pixNFR)

        return pixNFR



    def mouseClickedRed(self,evt):

        #get mouse coordinates in relation to image space
        posR = self.red.mapFromScene(evt.scenePos())

        #delete roi selection and re-initialise labels

        self.sessionkey.pop('roi_number', None)

        self.labelRatio.setText('Ratio FR/R = ')
        self.labelSsiGR.setText('SSI G vs R = ')
        self.labelSsiRFR.setText('SSI R vs FR = ')

        #round to get pixel size
        self.xR = int(posR.y())
        self.yR = int(posR.x())

        print(self.xR,self.yR)

        #get the index of the selected ROI after clicking
        self.GreenNew = self.computeGreenImage(self.xR,self.yR)
        self.RedNew = self.computeRedImage(self.xR,self.yR)
        self.FarRedNew = self.computeFarRedImage(self.xR,self.yR)

        self.functional.setImage(self.GreenNew)
        self.red.setImage(self.RedNew)
        self.far_red.setImage(self.FarRedNew)


    def mouseClickedFarRed(self,evt):

        #get mouse coordinates in relation to image space
        posFR = self.far_red.mapFromScene(evt.scenePos())

        #delete roi selection and re-initialise labels

        self.sessionkey.pop('roi_number', None)


        self.labelRatio.setText('Ratio FR/R = ')
        self.labelSsiGR.setText('SSI G vs R = ')
        self.labelSsiRFR.setText('SSI R vs FR = ')


        #round to get pixel size
        self.xFR = int(posFR.y())
        self.yFR = int(posFR.x())

        #get the index of the selected ROI after clicking
        self.GreenNew = self.computeGreenImage(self.xFR,self.yFR)
        self.RedNew = self.computeRedImage(self.xFR,self.yFR)
        self.FarRedNew = self.computeFarRedImage(self.xFR,self.yFR)

        self.functional.setImage(self.GreenNew)
        self.red.setImage(self.RedNew)
        self.far_red.setImage(self.FarRedNew)


    def imageNorm(self,mimg):
        #normalise the image using percentiles (idea of this part came from suite2p code)

        mimg1 = np.percentile(mimg, 1)
        mimg99 = np.percentile(mimg, 99)
        mimg = (mimg - mimg1) / (mimg99 - mimg1)
        mimg = np.maximum(0, np.minimum(1, mimg))

        return mimg


    def registerImages(self):

        mouse_name = self.sessionkey['mouse_name']
        session_name = self.sessionnamekey['session_name']
        projection_path = (f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\data\\{mouse_name}\\Recording\\'
                           f'projection_neurons\\{session_name}_projections')
        images = os.listdir(projection_path)

        def _find_img(tag):
            match = next((im for im in images if tag in im), None)
            if match is None:
                return None
            path = os.path.join(projection_path, match)
            with ScanImageTiffReader(path) as reader:
                stack = reader.data()
            return stack.mean(axis=0).astype(stack.dtype) if stack.ndim == 3 else stack

        baseline_img = _find_img('gcamp')
        ctb_img_farred = _find_img('_FR')
        ctb_img_red = _find_img('ctb')
        red_reg_img = _find_img('_R2')

        _ref = next((im for im in [baseline_img, ctb_img_farred, ctb_img_red, red_reg_img] if im is not None), None)
        _black = np.zeros(_ref.shape if _ref is not None else (512, 512), dtype=np.uint16)
        baseline_img = baseline_img if baseline_img is not None else _black
        ctb_img_farred = ctb_img_farred if ctb_img_farred is not None else _black
        ctb_img_red = ctb_img_red if ctb_img_red is not None else _black
        red_reg_img = red_reg_img if red_reg_img is not None else _black

        suite2pImg = self.ops['meanImg']
        xrange = self.ops['xrange']
        yrange = self.ops['yrange']

        # normalise the images using cv2
        baselineImgNorm = cv2.normalize(src=baseline_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        redRegImgNorm = cv2.normalize(src=red_reg_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        suite2pImgNorm = cv2.normalize(src=suite2pImg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # suite2pImgNorm = cv2.normalize(src=suite2pImgMax, dst=None, alpha=0, beta=60, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ctbImgRedNorm = cv2.normalize(src=ctb_img_red, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ctbImgFarRedNorm = cv2.normalize(src=ctb_img_farred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # transform images to SITK formats before registration
        colorImgGreenImgFixed = sitk.GetImageFromArray(suite2pImgNorm)
        colorImgGreenImgMoving  = sitk.GetImageFromArray(baselineImgNorm)
        colorImgRed2Fixed = sitk.GetImageFromArray(redRegImgNorm)
        colorImgRed1Moving= sitk.GetImageFromArray(ctbImgRedNorm)
        colorImgFarRedMoving = sitk.GetImageFromArray(ctbImgFarRedNorm)

        #register the two red images (R1 and R2)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(colorImgRed2Fixed)
        elastixImageFilter.SetMovingImage(colorImgRed1Moving)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.Execute()

        # get transformed red image and transformation (registering the R1 to R2)
        transformedOnceImgRed1 = elastixImageFilter.GetResultImage()
        transformedOnceImgRed1Array  = sitk.GetArrayFromImage(transformedOnceImgRed1)
        transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()

        #apply this tranformation to the FR image (using the R1 to R2 transformation)
        transformedOnceImgFarRed = sitk.Transformix(colorImgFarRedMoving, transformParameterMapVector)
        transformedOnceImgFarRedArray = sitk.GetArrayFromImage(transformedOnceImgFarRed)

        # register the two green images (suite2p and baseline green image)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(colorImgGreenImgFixed)
        elastixImageFilter.SetMovingImage(colorImgGreenImgMoving)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.Execute()

        # get transformed green image and transformation matrix
        transformedbaselineImg = elastixImageFilter.GetResultImage()
        transformedbaselineImgArray = sitk.GetArrayFromImage(transformedbaselineImg)
        transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()

        #apply the same transformation to the other colors
        transformedctbImgRed = sitk.Transformix(transformedOnceImgRed1, transformParameterMapVector)
        transformedctbImgRedArray = sitk.GetArrayFromImage(transformedctbImgRed)
        transformedctbImgFarRed = sitk.Transformix(transformedOnceImgFarRed, transformParameterMapVector)
        transformedctbImgFarRedArray = sitk.GetArrayFromImage(transformedctbImgFarRed)

        #sitk.WriteImage(transformedctbImgRed, "C:\\Users\\foustouk\\Desktop\\test\\R.tif")
        #sitk.WriteImage(transformedctbImgFarRed, "C:\\Users\\foustouk\\Desktop\\test\\FR.tif")

        transformedctbImgRedArrayNorm =  cv2.normalize(src=transformedctbImgRedArray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        transformedctbImgFarRedArrayNorm =  cv2.normalize(src=transformedctbImgFarRedArray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        ImgRedFull = self.imageNorm(np.transpose(transformedctbImgRedArrayNorm))
        ImgFarRedFull = self.imageNorm(np.transpose(transformedctbImgFarRedArrayNorm))
        ImgGreenFull = self.imageNorm(np.transpose(transformedbaselineImgArray))

        ImgRed = ImgRedFull[xrange[0]:xrange[1],yrange[0]:yrange[1]]
        ImgFarRed = ImgFarRedFull[xrange[0]:xrange[1],yrange[0]:yrange[1]]
        ImgGreen = ImgGreenFull[xrange[0]:xrange[1],yrange[0]:yrange[1]]

        print('Registration done!')

        # return (ImgRed, ImgFarRed, ImgGreen, ImgRed, ImgFarRed, ImgGreen)
        return (ImgRed, ImgFarRed, ImgGreen, ImgRedFull, ImgFarRedFull, ImgGreenFull)


    def calculateRatiosROIs(self):

        # get the unique ROI numbers for this key
        # rois =  set((lsensGF.CalciumRois() & self.sessionkey).fetch('roi_number'))
        rois =  self.rois

        self.GreenRoi = []
        self.RedRoi = []
        self.FarRedRoi = []
        self.Ratio = []
        self.RoiImageGreen = []
        self.RoiImageGreenMax = []
        self.RoiImageRed = []
        self.RoiImageFarRed = []
        self.ssiGR = []
        self.ssiRFR = []

        for i in rois:

            print('Doing ROI : ' + str(i))
            # read ROI stats ti find indexes in the images
            roiStats = self.stat
            # roiStats = (lsensGF.CalciumRois() & self.sessionkey & ('roi_number = ' + str(r))).fetch('roi_stats')

            # the ROIs are extracted once for every mouse so they are same for every session
            roiStatsSingle = roiStats[i]
            xRoi = roiStatsSingle['xpix'].flatten()
            yRoi = roiStatsSingle['ypix'].flatten()

            # read the intesities for every ROI from the corresponding images (careful x and y are inversed)
            roiGreen = self.ImgGreenFull[xRoi,yRoi]
            roiRed =  self.ImgRedFull[xRoi,yRoi]
            roiFarRed = self.ImgFarRedFull[xRoi,yRoi]

            roiGreenBB = self.ImgGreenFull[np.min(xRoi):np.max(xRoi), np.min(yRoi):np.max(yRoi)]
            roiRedBB =  self.ImgRedFull[np.min(xRoi):np.max(xRoi), np.min(yRoi):np.max(yRoi)]
            roiFarRedBB = self.ImgFarRedFull[np.min(xRoi):np.max(xRoi), np.min(yRoi):np.max(yRoi)]

            roiGreenNorm =  cv2.normalize(src=roiGreenBB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            roiRedNorm =  cv2.normalize(src=roiRedBB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            roiFarRedNorm =  cv2.normalize(src=roiFarRedBB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            try:
                ssmiGR  = ssim(roiGreenNorm,roiRedNorm)
                ssmiRFR = ssim(roiRedNorm,roiFarRedNorm)
            except:
                ssmiGR = np.nan
                ssmiRFR = np.nan

            self.ssiGR.append(ssmiGR)
            self.ssiRFR.append(ssmiRFR)

            # add the data to lists
            self.RoiImageGreen.append(roiGreenBB)
            self.RoiImageRed.append(roiRedBB)
            self.RoiImageFarRed.append(roiFarRedBB)

            #plt.imshow(roiGreenBB, cmap = 'gray')

            self.GreenRoi.append(np.mean(roiGreen))
            self.RedRoi.append(np.mean(roiRed))
            self.FarRedRoi.append(np.mean(roiFarRed))
            self.Ratio.append(np.mean(roiFarRed)/np.mean(roiRed))

        print('done')


    def predefineRois(self):

        Red = []
        FarRed  = []
        UN = []

        ratioMean = np.mean(self.Ratio)
        ratioSigma = np.std(self.Ratio)

        for i, r in enumerate(self.Ratio):
            if (r > (ratioMean + ratioSigma)) and (self.ssiRFR[i] < 0.8):
                FarRed.append(i)
            elif (r < (ratioMean - ratioSigma)) and (self.ssiRFR[i] < 0.8) and (self.ssiGR[i] < 0.8):
                Red.append(i)
            else:
                UN.append(i)
        os.makedirs(os.path.dirname(self.CheckRedFile), exist_ok=True)
        os.makedirs(os.path.dirname(self.CheckFarRedFile), exist_ok=True)
        os.makedirs(os.path.dirname(self.CheckUNFile), exist_ok=True)
        np.save(self.CheckRedFile,Red)
        np.save(self.CheckFarRedFile, FarRed)
        np.save(self.CheckUNFile, UN)

        return (FarRed, Red, UN)


    def makeRoisImage(self):

        rois = self.rois
        imageMax = self.ops['max_proj']
        imageMaxNorm = self.imageNorm(imageMax)

        Lx = self.ops['Lx']
        Ly = self.ops['Ly']
        xrange = self.ops['xrange']
        yrange = self.ops['yrange']

        # initalise images and index image
        H = np.zeros([Lx,Ly])
        S = np.zeros([Lx,Ly])
        img = np.zeros([Lx,Ly])

        iROIs =  np.ones([Lx,Ly])*(-1)

        img[xrange[0]:xrange[1],yrange[0]:yrange[1]] = np.transpose(imageMaxNorm)

        for i in rois:

            roiStats = self.stat

            roiStatsSingle = roiStats[i]
            xRoi = roiStatsSingle['xpix'].flatten()
            yRoi = roiStatsSingle['ypix'].flatten()

            if i in self.Red:
                H[xRoi, yRoi] = 0
                S[xRoi ,yRoi] = 1
                img[xRoi ,yRoi] = 1

            if i in self.Far_red:
                H[xRoi, yRoi] = 0.67
                S[xRoi ,yRoi] = 1
                img[xRoi ,yRoi] = 1
            if i in self.UN:
                H[xRoi, yRoi] = 0.33
                S[xRoi ,yRoi] = 1
                img[xRoi ,yRoi] = 1

            iROIs[xRoi ,yRoi] = int(i)

        pix = np.concatenate(((H[:, :, np.newaxis]),
                               S[:, :, np.newaxis],
                               img[:, :, np.newaxis]), axis=-1)

        pix = hsv_to_rgb(pix)

        ROIsImage = pix[xrange[0]:xrange[1],yrange[0]:yrange[1],:]
        iROIsC = iROIs[xrange[0]:xrange[1],yrange[0]:yrange[1]]

        plt.imshow(iROIsC)

        return (ROIsImage, iROIsC, img, H, S, xrange, yrange, Lx, Ly)


def run():
    #run the application
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        projectionsGUI = ProjectionsGUI()
        app.exec_()

run()


