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

exec &> "log_iccm2019_test.txt"

sleep_sec=10
experiment_name="iccm2019_test"

operand_digits=4
operator='add'
rnn_type='jordan'
activation='relu'
hidden_units=24
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

  for i in {1..7..1}
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

experiment 0 &
sleep $sleep_sec
