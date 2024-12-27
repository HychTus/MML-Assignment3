#!/bin/bash

# ---Task3.1 Section A Download---
# direct download

# URL="https://diskasu.pku.edu.cn:10002/Bucket/bab1d452-c23b-4d03-b766-329261bcc4d6/1257206D1CB24C49A7E99DCFED5C46E5/ED4AC93C30D14BA5B73BAA3CE13FC225?response-content-disposition=attachment%3B%20filename%2A%3Dutf-8%27%27coco%255fcaptioning.zip&AWSAccessKeyId=ASE&Expires=1735308866&Signature=c3RXQkic5DPjASWHN%2bEd8Y3P7wg%3d"
# wget -O datasets/coco_captioning.zip $URL

# ---Task3.2 Section B Download---
VAL_URL=http://images.cocodataset.org/zips/val2014.zip
TRAIN_URL=http://images.cocodataset.org/zips/train2014.zip

aria2c -x 16 -o datasets/val2014.zip $VAL_URL
aria2c -x 16 -o datasets/train2014.zip $TRAIN_URL

# wget -O val2014.zip $VAL_URL
# wget -O train2014.zip $TRAIN_URL