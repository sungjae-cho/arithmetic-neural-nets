#!/bin/bash

i_device=0
n_devices=5
get_device_num()
{
  device_num=$(( $i_device % $n_devices ))
  echo $device_num
  i=$((i_device + 1))
}


for j in {1..60..1}
  do
    get_device_num
  done
