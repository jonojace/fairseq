#!/usr/bin/env sh

for scratch_name in scratch_fast scratch_ssd scratch3 scratch2 scratch1 scratch scratch_big
do
#  echo "Checking for writability to scratch disk: $scratch_name"
  if [ -w /disk/${scratch_name} ]; then
    # move data to that scratch disk
    mkdir -p /disk/${scratch_name}/s1785140
    rsync -au /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/${scratch_name}/s1785140
    found_writable_scratch=true
    break
  fi
done

if [ "$found_writable_scratch" = true ]; then
#  echo "/disk/${scratch_name} on $HOSTNAME is writable, successfully moved data to this scratch disk"
  echo "/disk/${scratch_name}/s1785140"
else
#  echo "!!!WARNING!!! did not find a scratch disk on $HOSTNAME with write permissions."
#  echo 'Consider using srun --part=ILCC_GPU,CDT_GPU --nodelist=$NODE --pty bash to investigate what scratchs are in /disk/'
  echo "None"
fi



