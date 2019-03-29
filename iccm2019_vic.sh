#!/bin/bash
# This experiment aims to show how mean_correct_answer_step varies dependently on
# whether two operands are ordered. This means if two operands are a and b, then
# a >= b.
# Experiment name : 'ordered_operands'
# Configurations
# - operand_digits=4
# - operator='add'
# - rnn_type='jordan'
# - activation='relu'
# - hidden_units=64
# - confidence_prob=0.9
# - max_steps=30


exec &> "log_iccm2019.txt"

sleep_sec=10
experiment_name="iccm2019"

operand_digits=4
operator='add'
rnn_type='jordan'
activation='relu'
hidden_units=48
confidence_prob=0.9
max_steps=30

start_echo()
{
  operator=$1
  echo "==================================================================="
  echo "Experiment name : ${experiment_name}"
  echo "Operand digits: ${operand_digits}"
  echo "Operator: ${operator}"
  echo "RNN type: ${rnn_type}"
  echo "Hidden activation: ${activation}"
  echo "Hidden units: ${hidden_units}"
  echo "Confidence probability: ${confidence_prob}"
  echo "Max steps: ${max_steps}"
  echo "==================================================================="
}

experiment()
{
  device_num=$1

  ################################################################################

  for i in {1..6..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done
}


for j in {1..50..1}
  do
    device_num=$(( $j % 5 ))
    experiment $device_num &
    sleep $sleep_sec
  done
