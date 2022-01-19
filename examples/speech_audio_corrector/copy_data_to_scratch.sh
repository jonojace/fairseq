#!/bin/sh

# copy log audio data to scratch space (on current node)
NODENAME=$(echo $HOSTNAME | cut -d. -f1)
echo we are on node: $NODENAME
SCRATCH_DISK=scratch_fast
echo copying data to $SCRATCH_DISK
mkdir -p /disk/${SCRATCH_DISK}/s1785140 #careful sometimes scratch disk is named something else!!!
rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/${SCRATCH_DISK}/s1785140
ls /disk/${SCRATCH_DISK}/s1785140
