import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import math
import copy
import json

import PIL.Image

import torchvision
from torchvision import transforms
from torch.autograd import Variable

import torch2trt
from torch2trt import TRTModule

import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects

from scipy.spatial.distance import euclidean


############### Social Recognition ########################
# open-pose ver : index 0~8 
# trt-pose ver : 0, 17, 6, 8, 10, 5, 7, 9, (11+12)/2

vAllX = []
vAllY = []
nViewFrame = 5
nCheckFrame = nViewFrame * 3
fActionArr = [0 for _ in range(nCheckFrame)]
fActionProb = []
nNumJoint = 9
nNumAction = 15

class ETRIFace:
    def __init__(self):
        # class init
        self.rt = [-1, -1, -1, -1]
        self.fAge = -1.
        self.fGender = -1.
        self.fGlasses = -1.
        self.sID = ""
        self.fvScore = -1.
        self.ptLE = [-1, -1]
        self.ptRE = [-1, -1]
        self.ptLM = [-1, -1]
        self.ptRM = [-1, -1]
        self.ptN = [-1, -1]

        self.fYaw = -1.
        self.fPitch = -1.
        self.fRoll = -1.


def ETRI_Initialization(path):
    # Load & Init for Skeletons
    with open('./utils/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)
    
    print("trtPose start")
    model_skeleton = TRTModule()
    model_path = os.path.join(path, 'resnet18_baseline_att_224x224_A_epoch_249_trt_2.pth')
    model_skeleton.load_state_dict(torch.load(model_path))
    
    print("body action start")
    model_trt_ba = TRTModule()
    model_path = os.path.join(path, 'bodyaction_TRT.pth')
    model_trt_ba.load_state_dict(torch.load(model_path))

    print("hand action start")
    model_trt_ha = TRTModule()
    model_path = os.path.join(path, 'handaction_jc_TRT.pth')
    model_trt_ha.load_state_dict(torch.load(model_path))
   
    return topology, parse_objects, model_skeleton, model_trt_ba, model_trt_ha


def trtPose_preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def ETRI_Pose_Execute(model_skeleton, frame, parse_objects):
    frame_224 = copy.deepcopy(frame)
    frame_224 = cv2.resize(frame_224,(224,224),interpolation=cv2.INTER_LINEAR)
    data = trtPose_preprocess(frame_224)
    data = data.contiguous()
    cmap, paf = model_skeleton(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)

    return counts, objects, peaks


def ETRI_estimate_face_from_joint(topology,image, object_counts, objects, normalized_peaks):
    height = image.shape[0]
    width = image.shape[1]

    x_start =0
    x_end =0
    y_start =0
    y_end =0

    count = int(object_counts[0])
    list_ETRIFace = []

    nMaxIndex = -1
    nMaxWidth = -1
    for ii in range(count):
        ef = ETRIFace()
        bFindFace = False

        obj = objects[0][ii]

        if (obj[0]>=0 and obj[1]>=0 and obj[2]>=0 and obj[3]>=0 and obj[4]>=0):
            peak_3 = normalized_peaks[0][3][int(obj[3])]
            peak_4 = normalized_peaks[0][4][int(obj[4])]
            peak_c = normalized_peaks[0][0][int(obj[0])]
            peak_cx = peak_c[1]*width
            peak_cy = peak_c[0]*height
            length = (peak_3[1] - peak_4[1]) * width
            margin = length*0.2
            x_start = peak_cx - length/2 - margin
            y_start = peak_cy - length/2 - margin
            x_end = peak_cx + length/2 + margin
            y_end = peak_cy + length/2 + margin
            bFindFace = True
        elif (obj[0]>=0 and obj[1]>=0 and obj[2]>=0 and obj[3]<0 and obj[4]>=0):
            #print(1)
            peak_1 = normalized_peaks[0][1][int(obj[1])]
            peak_4 = normalized_peaks[0][4][int(obj[4])]
            peak_c = (peak_1 + peak_4)/2
            peak_cx = peak_c[1]*width
            peak_cy = peak_c[0]*height    
                        
            length = (peak_1[1] - peak_4[1]) * width * 1.2
            margin = length*0.2
            x_start = peak_cx - length/2 - margin
            y_start = peak_cy - length/2 - margin
            x_end = peak_cx + length/2 + margin
            y_end = peak_cy + length/2 + margin
            bFindFace = True
        elif (obj[0]>=0 and obj[1]>=0 and obj[2]>=0 and obj[3]>=0 and obj[4]<0):
            peak_2 = normalized_peaks[0][2][int(obj[2])]
            peak_3 = normalized_peaks[0][3][int(obj[3])]
            peak_c = (peak_2 + peak_3)/2
            peak_cx = peak_c[1]*width
            peak_cy = peak_c[0]*height 
            length = (peak_3[1] - peak_2[1]) * width * 1.2
            margin = length*0.2
            x_start = peak_cx - length/2 - margin
            y_start = peak_cy - length/2 - margin
            x_end = peak_cx + length/2 + margin
            y_end = peak_cy + length/2 + margin
            bFindFace = True
        
        if bFindFace:
            ef.rt = [x_start, y_start, x_end, y_end]

            ef.skeletons = []
            obj = objects[0][ii]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if (j >= 0):
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    ef.skeletons.append((x, y))
            list_ETRIFace.append(ef)

            tWidth = x_end - x_start
            if tWidth > nMaxWidth:
                nMaxWidth = tWidth
                nMaxIndex = len(list_ETRIFace) - 1
    
    return list_ETRIFace, nMaxIndex


def convertInputJointFormat(list_ETRIFace, nIndex):
    vInputJointX = []
    vInputJointY = []

    list_JointIndex = [0, 17, 6, 8, 10, 5, 7, 9]
    for jointIndex in list_JointIndex:
        nJointX, nJointY = list_ETRIFace[nIndex].skeletons[jointIndex]
        vInputJointX.append(nJointX)
        vInputJointY.append(nJointY)

    x = (list_ETRIFace[nIndex].skeletons[11][0] + list_ETRIFace[nIndex].skeletons[12][0])/2
    y = (list_ETRIFace[nIndex].skeletons[11][1] + list_ETRIFace[nIndex].skeletons[12][1])/2
    vInputJointX.append(x)
    vInputJointY.append(y)

    list_JointIndex = [2, 1]
    for jointIndex in list_JointIndex:
        nJointX, nJointY = list_ETRIFace[nIndex].skeletons[jointIndex]
        vInputJointX.append(nJointX)
        vInputJointY.append(nJointY)

    return vInputJointX, vInputJointY


# updateVector
def updateJoint(vInputJointX, vInputJointY):
    global vAllX, vAllY

    vAllX = vAllX + vInputJointX[0:9]
    vAllY = vAllY + vInputJointY[0:9]

    if len(vAllX) > nViewFrame * nNumJoint:
        del vAllX[0:nNumJoint]
        del vAllY[0:nNumJoint]


def convertToActionArr():
    global nNumJoint, nViewFrame, vAllX, vAllY
    mTransformed = np.zeros((nNumJoint, nViewFrame, 3), dtype="uint8")

    if len(vAllX) < nViewFrame*nNumJoint:
        mTransformed = cv2.resize(mTransformed, (64, 64))
        return mTransformed

    tX = copy.copy(vAllX)
    tY = copy.copy(vAllY)

    tX.sort()
    tY.sort()

    while 0 in tX:
        tX.remove(0)
    while 0 in tY:
        tY.remove(0)

    for ff in range(nViewFrame):
        for jj in range(nNumJoint):
            dX = (tX[len(tX) - 1] - tX[0])
            dY = (tY[len(tY) - 1] - tY[0])
            if dX == 0:
                dX = 0.0000000000001
            if dY == 0:
                dY = 0.0000000000001

            mTransformed[jj][ff][0] = min(255 * max(vAllX[ff * nNumJoint + jj] - tX[0], 0) / dX, 255)
            mTransformed[jj][ff][1] = min(255 * max(vAllY[ff * nNumJoint + jj] - tY[0], 0) / dY, 255)


    mTransformed = cv2.resize(mTransformed, (64, 64))
    return mTransformed


def EAR_BodyAction_Estimation(BA_Net, convertedImg):
    global fVelocityX, fVelocityY, vAllX, vAllY

    # cv2.imshow("ZZZZZZ", convertedImg)
    # if fVelocityY != 0:
    #     print("%f / %f" % (fVelocityX, fVelocityY))

    # bowing
    fShoulder = math.fabs(vAllX[nNumJoint * (nViewFrame - 1) + 5] - vAllX[nNumJoint * (nViewFrame - 1) + 2])
    fNeck = math.fabs(vAllY[nNumJoint*(nViewFrame-1) + 0] - vAllY[nNumJoint*(nViewFrame-1) + 1])
    if fNeck < fShoulder/6 or vAllY[nNumJoint*(nViewFrame-1) + 0] > vAllY[nNumJoint*(nViewFrame-1) + 1]:
        return 15

    convertedImg = cv2.cvtColor(convertedImg, cv2.COLOR_BGR2RGB)
    img_trim_in = convertedImg / 255
    img_trim_in = img_trim_in[np.newaxis, :, :, :]
    img_trim_in = np.transpose(img_trim_in,(0,3,1,2)).astype(np.float32)
    torch_input = torch.from_numpy(img_trim_in)
    torch_input = torch_input.cuda()
    torch_input = torch_input.contiguous()
    _, output = BA_Net(torch_input)
    output_np = output.cpu().detach().numpy().squeeze().tolist()

    return output_np.index(max(output_np))


def updateAction(nAction):
    global fActionArr, nCheckFrame
    for ii in range(nCheckFrame-1, 0, -1):
        fActionArr[ii] = fActionArr[ii-1]
    fActionArr[0] = nAction

sAction = ["foldarms", "handaction","neutral", "pickear", "restchin", "scratch", "waving", "fighting", "thumbchuck", "bitenail", "shakehand", "fingerok", "fingerheart", "covermouth", "touchnose", "bowing"]
def getTopNAction(nTopN):
    global fActionProb, fActionArr, nCheckFrame, nNumAction, sAction

    if nTopN > nNumAction:
        return "nTopN is out of scope."

    fActionProb = [0 for _ in range(nNumAction+1)]
    fTemp = [0 for _ in range(nNumAction+1)]

    for ii in range(nCheckFrame):
        fActionProb[fActionArr[ii]] = fActionProb[fActionArr[ii]] + 1
    fSum = 0.

    for ii in range(nNumAction+1):
        fExp = math.exp(fActionProb[ii])
        fSum = fSum + fExp
        fTemp[ii] = fExp

    for ii in range(nNumAction+1):
        fActionProb[ii] = (fTemp[ii] / fSum, ii)

    fActionProb.sort(reverse=True)


    sTopN = ""
    for ii in range(nTopN):
        sActionNProb = "{sAction} : {fProb:0.1f} / ".format(sAction=sAction[fActionProb[ii][1]]
                                                         , fProb=fActionProb[ii][0]*100)
        sTopN  = sTopN + sActionNProb

    return sTopN


def getKeypointDistance(kp1, kp2):
    return math.sqrt(pow(kp1[0] - kp2[0], 2) + pow(kp1[1]-kp2[1], 2))
    

def getHandPatch(img, vInputJointX, vInputJointY):
    try:
        if len(vInputJointX) != 0 and len(vInputJointY) != 0:
            nBorderMargin = int(img.shape[1] / 2)
            handPatchImg = np.zeros((128, 128, 3), np.uint8)
            borderImg = cv2.copyMakeBorder(img, nBorderMargin, nBorderMargin, nBorderMargin, nBorderMargin,
                                           cv2.BORDER_CONSTANT, 0)
            # borderImg = cv2.cvtColor(borderImg, cv2.COLOR_RGB2GRAY)

            # print("lenX : %d / lenY : %d" % (len(vInputJointX), len(vInputJointY)))
            # kp_distance = abs(vInputJointX[9] - vInputJointX[10])
            kp_distance = getKeypointDistance((vInputJointX[9], vInputJointY[9]), (vInputJointX[10], vInputJointY[10]))
            if kp_distance < 20:
                return 0, 0, 1
            
            width_half = int((kp_distance) * 2.0)

            # print("cp1")

            L_Wrist = (vInputJointX[4], vInputJointY[4])
            L_Elbow = (vInputJointX[3], vInputJointY[3])
            R_Wrist = (vInputJointX[7], vInputJointY[7])
            R_Elbow = (vInputJointX[6], vInputJointY[6])

            LhandPatchImg = RhandPatchImg = np.zeros((224, 224, 3), np.uint8)

            # print("cp2")

            nImageCenter = nBorderMargin
            # LH
            if L_Wrist[0] and L_Elbow[0]:
                LH_Center_X = L_Wrist[0] + (-0.5 * (L_Elbow[0] - L_Wrist[0])) + nImageCenter
                LH_Center_Y = L_Wrist[1] + (-0.5 * (L_Elbow[1] - L_Wrist[1])) + nImageCenter

                xmin = LH_Center_X - width_half
                xmax = LH_Center_X + width_half
                ymin = LH_Center_Y - width_half
                ymax = LH_Center_Y + width_half

                LH_Patch = cv2.resize(borderImg[int(ymin):int(ymax), int(xmin):int(xmax)], (224, 224))

                LhandPatchImg = copy.deepcopy(LH_Patch)
                LhandPatchImg = cv2.flip(LhandPatchImg, 1)

                # print("cp3")

            if R_Wrist[0] and R_Elbow[0]:
                RH_Center_X = R_Wrist[0] + (-0.5 * (R_Elbow[0] - R_Wrist[0])) + nImageCenter
                RH_Center_Y = R_Wrist[1] + (-0.5 * (R_Elbow[1] - R_Wrist[1])) + nImageCenter

                xmin = RH_Center_X - width_half
                xmax = RH_Center_X + width_half
                ymin = RH_Center_Y - width_half
                ymax = RH_Center_Y + width_half

                RH_Patch = cv2.resize(borderImg[int(ymin):int(ymax), int(xmin):int(xmax)], (224, 224))

                RhandPatchImg = copy.deepcopy(RH_Patch)

                # print("cp4")



            # print("cp5")

            return LhandPatchImg, RhandPatchImg, 0
        else:
            return 0, 0, 1

    except Exception as e:
        print(e)
        return 0, 0, 1

def list2SoftList(srcList):
    tmpList = srcList.copy()

    fSum = 0.

    for ii in range(len(srcList)):
        fExp = np.exp(srcList[ii])
        fSum = fSum + fExp
        tmpList[ii] = fExp
    for ii in range(len(srcList)):
        srcList[ii] = tmpList[ii] / fSum

    return srcList

transformations_HandAction = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def getHandActionIdx(HA_Net, handPatchImg):
    global fActionArr

    handPatchImg = cv2.cvtColor(handPatchImg, cv2.COLOR_BGR2RGB)
    PILImg = Image.fromarray(handPatchImg)
    img = transformations_HandAction(PILImg)
    img = img.contiguous()
    img = img.cuda()
    img = img.unsqueeze(0)

    output = HA_Net(img)

    output_cpu = output.cpu().detach().numpy().squeeze()
    max_idx = output_cpu.argmax()

    return max_idx + 7


def getSocialActionIndex(sActionResult):
    sSocialActionList = ["bitenail", "covermouth", "fighting", "fingerheart", "fingerok",
    "foldarms", "neutral", "pickear", "restchin", "scratch",
    "shakehand", "thumbchuck", "touchnose", "waving", "bowing"]

    sActionSplit = sActionResult.split(" ")[0]

    if sActionSplit =="handaction":
        return -1
    else:
        return sSocialActionList.index(sActionSplit)


################################################################3
################################################################3
################################################################3
################################################################3
################################################################3


avgNeutralJointX = [322.14026178524216, 323.3281960896681
    , 277.6280570269355, 265.22208367646823, 261.6198277323165
    , 369.14238240576293, 381.8753715723246, 384.12814252086275
    , 383.8150782772658, 377.16965215767823, 372.35474501571855, 371.4983903382251, 371.42920561757785
    , 382.2229180099679, 378.82329372475, 374.7004049838305, 372.06152992247354
    , 386.23766577882594, 382.318045646779, 377.9146361243536, 374.59103746959545
    , 388.5917371759107, 385.05130484878816, 380.5643692650183, 377.4295040421212
    , 389.13456373253734, 386.456162121497, 383.45871614932923, 381.25423902040205
    , 261.06534784463366, 266.6972531766876, 270.331130736252, 270.8570231111664, 271.0723326401262
    , 260.0929566000788, 264.0953283027428, 268.5716093939732, 271.32327342868064
    , 257.1165693398627, 261.6365024253106, 266.3998209939861, 269.5787362852653
    , 255.62472866798808, 259.84006482310855, 264.52176403682466, 267.37121442783734
    , 256.07010035496364, 259.17439073593306, 262.3886682265787, 264.53931257396704]
avgNeutralJointY = [187.05623015997642, 250.92758932685655
    , 251.01828426255068, 326.396158656766, 398.44751644240466
    , 250.45998458556156, 325.98409536873675, 396.96939004778125
    , 399.4718000294363, 405.37457221026983, 415.9773905655077, 426.3001862842175, 433.4880929239434
    , 424.5302855937776, 435.2456918539938, 438.14686639701415, 439.4700757764309
    , 425.23371987919376, 435.28300817741234, 437.4852645445141, 437.91386246124137
    , 424.20721414015026, 432.6158489755624, 434.9105444245196, 435.39410204690677
    , 422.34410844644253, 429.5531373514435, 431.4419222423643, 432.0382540928608
    , 400.9096647112953, 406.9963011366081, 417.58202193347114, 427.61835087832156, 434.67334775218734
    , 425.13964785421615, 435.42590991313597, 438.08185959366193, 439.2206478706609
    , 425.47184133060625, 435.3077795402021, 436.7916575994253, 436.6014509559256
    , 424.5252443643856, 432.7842847999346, 434.2857559218688, 433.88741732793784
    , 422.80452726493945, 429.91303461321115, 431.1844964654373, 431.10141673363245]


# ========= Tendency params =============

def drawJoint(cvImg, vInputJointX, vInputJointY):
    xLen = len(vInputJointX)
    yLen = len(vInputJointY)

    if not xLen == yLen:
        return cvImg

    for ii in range(xLen):
        cv2.circle(cvImg, (vInputJointX[ii], vInputJointY[ii]), 2, (0,255,0), -1)

    return cvImg


def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def alignSkeleton():
    global vAllX, vAllY, avgNeutralJointX, avgNeutralJointY, nNumJoint, nViewFrame

    alignedNeutralX = []
    alignedNeutralY = []

    for ff in range(nViewFrame):
        copyNeutralX = copy.deepcopy(avgNeutralJointX)
        copyNeutralY = copy.deepcopy(avgNeutralJointY)
        frameX = vAllX[ff*nNumJoint + 0:ff*nNumJoint + nNumJoint]
        frameY = vAllY[ff*nNumJoint + 0:ff*nNumJoint + nNumJoint]

        # Scale
        fActionJoint01D = euc_dist((frameX[0],frameY[0]), (frameX[1],frameY[1]))
        fAvgJoint10D = euc_dist((copyNeutralX[0],copyNeutralY[0]), (copyNeutralX[1],copyNeutralY[1]))
        fScale = fActionJoint01D / fAvgJoint10D
        for ii in range(len(copyNeutralX)):
            copyNeutralX[ii] = copyNeutralX[ii] * fScale
            copyNeutralY[ii] = copyNeutralY[ii] * fScale

        # translation
        fXOffset = frameX[1] - copyNeutralX[1]
        fYOffset = frameY[1] - copyNeutralY[1]
        
        for ii in range(len(copyNeutralX)):
            copyNeutralX[ii] = copyNeutralX[ii] + fXOffset
            copyNeutralY[ii] = copyNeutralY[ii] + fYOffset
        # copyNeutralX = copyNeutralX + fXOffset
        # copyNeutralY = copyNeutralY + fYOffset

        # alignedNeutralX = alignedNeutralX + np.ndarray.tolist(copyNeutralX)
        # alignedNeutralY = alignedNeutralY + np.ndarray.tolist(copyNeutralY)
        alignedNeutralX = alignedNeutralX + copyNeutralX
        alignedNeutralY = alignedNeutralY + copyNeutralY


    return alignedNeutralX, alignedNeutralY


def getVectorDistance(alignedNeutralX, alignedNeutralY):
    global vAllX, vAllY, nNumJoint, nViewFrame

    if len(alignedNeutralX) != len(vAllX) or len(alignedNeutralY) != len(vAllY):
        return -1

    distSum = 0.

    for ii in range(len(vAllX)):
        if vAllX[ii] == 0:
            alignedNeutralX[ii] = 0
        if vAllY[ii] == 0:
            alignedNeutralY[ii] = 0

    dist = euclidean(vAllX, alignedNeutralX)
    distSum = distSum + dist
    dist = euclidean(vAllY, alignedNeutralY)
    distSum = distSum + dist

    return distSum


def getTendencyCategory(fDistance):
    if fDistance<1500:
        return 1
    elif fDistance<4000:
        return 0
    else:
        return 2





