#!/bin/bash
exec &> "log_jordan_cp9_add_sub_maxstep.txt"

sleep_sec=10
experiment_name="jordan_cp9_add_sub_maxstep"
operand_digits=4
hidden_units=64

start_echo()
{
  operator=$1
  rnn_type=$2
  confidence_prob=$3
  echo "==================================================================="
  echo "Operator: ${operator}"
  echo "RNN type: ${rnn_type}"
  echo "Confidence probability: ${confidence_prob}"
  echo "Hidden activation: ${activation}"
  echo "Hidden units: ${hidden_units}"
  echo "==================================================================="
}

experiment()
{
  device_num=$1

  rnn_type='jordan'
  confidence_prob=0.9

  ################################################################################
  activation='relu'

  max_steps=30

  operator='add'
  for i in {1..10..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  max_steps=20

  operator='add'
  for i in {1..10..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  max_steps=10

  operator='add'
  for i in {1..10..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

}

experiment 1 &
sleep $sleep_sec

experiment 0 &
sleep $sleep_sec

experiment 1 &
sleep $sleep_sec
