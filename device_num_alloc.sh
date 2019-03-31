#!/bin/bash

i=0
n_devices=5
get_device_num()
{
  device_num=$(( $i % $n_devices ))
  echo $device_num
  i=$((i + 1))
}


for j in {1..60..1}
  do
    get_device_num
  done
