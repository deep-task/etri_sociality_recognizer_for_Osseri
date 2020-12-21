#!/bin/bash

cd

ETRI_RECOG_PATH=`find ./ -name etri_sociality_recognizer -type d`

echo $ETRI_RECOG_PATH

if [ -z $ETRI_RECOG_PATH ]
then
    echo "Cannot find path !!"
else
    cd $ETRI_RECOG_PATH'/models'
    rm *
    FILEID='12uN2Hh8XkWAgGEzplPY4F6o6EKdtHlZ-'
    FILENAME='ETRI_Osseri_Model.tar.gz'
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id="$FILEID > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id="$FILEID -o $FILENAME
    rm cookie
    tar -zxvf $FILENAME
    echo "Modelfile Download Complete!!"
fi 
