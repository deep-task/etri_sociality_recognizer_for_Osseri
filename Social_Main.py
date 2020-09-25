#!/usr/bin/env python
#-*- encoding: utf8 -*-
import cv2
import numpy as np
import torch
import ETRI_Action_Recognition as EAR
import time
import copy
import os
import socket
import json





# socket Interface 초기화
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def get_recog_result_json(list_ETRIFace, nBiggestIndex, nSocialActionCode):
    recog_info = {
        "encoding": "UTF-8",
        "header": {
                "content": ["human_recognitiopn"],
                "source": "ETRI",
                "target": ["UOA", "UOS"],
                "timestamp": 0
        },
        "human_recognition": [{
            "face_roi": {
                    "x1": list_ETRIFace[nBiggestIndex].rt[0].item(),
                    "x2": list_ETRIFace[nBiggestIndex].rt[2].item(),
                    "y1": list_ETRIFace[nBiggestIndex].rt[1].item(),
                    "y2": list_ETRIFace[nBiggestIndex].rt[3].item()
                },
                "gender": -1,
                "age": -1,
                "headpose": {
                    "yaw": list_ETRIFace[nBiggestIndex].fYaw,
                    "pitch": list_ETRIFace[nBiggestIndex].fPitch,
                    "roll": list_ETRIFace[nBiggestIndex].fRoll
                },
                "glasses": False,
                "social_action": nSocialActionCode,
                "gaze": -1,
                "name": "",
                "longterm_tendency": -1,
                "lognterm_habit": -1
        }]
    }

    jsonString = json.dumps(recog_info)
    return jsonString


def main():
    # HOST = "129.254.90.89"
    HOST = "192.168.0.2"
    PORT = 9999

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    message = "1"
    client_socket.send(message.encode())

    # not need to check the connection...
    # if not connected, waiting at connect function.
    bReady = True

    # 행동 정보 인식 모델 초기화
    topology, parse_objects, Skeleton_Net, BodyAction_Net, HandAction_Net = EAR.ETRI_Initialization("./models/")

    # 개인 정보 관리를 위해 기존 데이터 존재여부 확인
    nFrameCNT = 1


    # curtime, prevtime = 0, 0

    # 인식 프로세스 시작
    while(bReady):
        # 얼굴 검출 정보 / 행동인식 정보 메모리 초기화
        list_ETRIFace = []
        nMainFaceIndex = -1
        nSocialActionCode = -1
        nTendency = -1


        # receive image frame
        callString = "call"
        client_socket.send(callString.encode())
        length = recvall(client_socket, 16)
        stringData = recvall(client_socket, int(length))
        data = np.frombuffer(stringData, dtype="uint8")
        frame = cv2.imdecode(data, 1)


        # main process start
        frame = cv2.flip(frame, 1)
        img_show = copy.deepcopy(frame)

        # Action Recognition
        #######################
        # Get Skeleton data
        counts, objects, normalized_peaks = EAR.ETRI_Pose_Execute(Skeleton_Net, frame, parse_objects)
        # Estimation Facial ROI / Get Maximum face index
        list_ETRIFace, nBiggestIndex = EAR.ETRI_estimate_face_from_joint(topology, frame, counts, objects, normalized_peaks)

        if nBiggestIndex != -1:
            # Get joint array
            vInputJointX, vInputJointY = EAR.convertInputJointFormat(list_ETRIFace, nBiggestIndex)

            # update joint inform
            EAR.updateJoint(vInputJointX, vInputJointY)

            # joint 정보가 ViewFrame 만큼 쌓인 경우에만 행동인식 수행
            if len(EAR.vAllX) == EAR.nViewFrame * EAR.nNumJoint:
                # get converted feature map
                convertedImg = EAR.convertToActionArr()
                nTempAction = EAR.EAR_BodyAction_Estimation(BodyAction_Net, convertedImg)
                EAR.updateAction(nTempAction)
                sActionResult = EAR.getTopNAction(1, convertedImg)

                #### edit from here. return format / index / result string etc...
                if sActionResult.split(" ")[0] == "handaction":
                    hand_patch, _, checkVal = EAR.getHandPatch(frame, vInputJointX, vInputJointY)

                    if checkVal == 0:
                        sActionResult = EAR.getHandActionStr(HandAction_Net, hand_patch)

                else:
                    EAR.updateAction(nTempAction)
                print(sActionResult)
                nSocialActionCode = EAR.getSocialActionIndex(sActionResult)



            # update는 20frame 마다 실행 (20FPS 기준. 약 1초간격으로 업데이트)
            if nFrameCNT % 20 == 0:
                # 개인 정보에 행동 및 태도 정보 업데이트.
                if nSocialActionCode != -1:

                    # skeleton 위치 정규화
                    a, b = EAR.alignSkeleton()
                    # skeleton 움직임 기반 Tendency score 계산
                    fTendencyScore = EAR.getVectorDistance(a, b)
                    # Score 기반 성향 계산
                    nTendency = EAR.getTendencyCategory(fTendencyScore)


            jsonString = get_recog_result_json(list_ETRIFace, nBiggestIndex, nSocialActionCode)
            client_socket.send(jsonString.encode())


        cv2.imshow("TT", img_show)
        nKey = cv2.waitKey(1)

        ##########################################################################
        # # 인식 결과 출력
        # curtime = time.time()
        # sec = curtime - prevtime
        # prevtime = curtime
        # fps = 1 / sec
        # sFPS = "FPS : %.1f" % fps
        # cv2.putText(img_show, sFPS, (10, 15), 0, 0.6, (0, 255, 255), 1)
        #
        # cv2.imshow("TT", img_show)
        # nKey = cv2.waitKey(1)
        # # esc로 업데이트 된 개인정보 저장하고 프로그램 종료
        # if nKey == 27:
        #     PI.writePersonalInformation(list_PI, PI_DB_PATH)
        #     break
        # # "I" 키로 등록된 개인정보 리스트 전체 확인
        # elif nKey == 73 or nKey == 105:
        #     PI.showPersonalInformationAll(list_PI)
        # # "R" 키로 부정확하게 등록된 정보 삭제하고 남은 리스트 확인
        # elif nKey == 82 or nKey == 114 :
        #     PI.autoRemoveRarePerson(list_PI)
        #     PI.showPersonalInformationAll(list_PI)
        #
        # nFrameCNT = nFrameCNT + 1
        ##########################################################################

    client_socket.close()

if __name__ == "__main__":
    main()

