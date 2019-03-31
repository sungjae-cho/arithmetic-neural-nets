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
  # Subtraction ###############################################################
  # 1
  operator='subtract'
  confidence_prob=0.9
  hidden_units=48
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 2
  operator='subtract'
  confidence_prob=0.9
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done
  # 3
  operator='subtract'
  confidence_prob=0.8
  hidden_units=48
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 4
  operator='subtract'
  confidence_prob=0.8
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 5
  operator='subtract'
  confidence_prob=0.7
  hidden_units=48
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 6
  operator='subtract'
  confidence_prob=0.7
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # Addition ###################################################################
  # 7
  operator='add'
  confidence_prob=0.9
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 8
  operator='add'
  confidence_prob=0.8
  hidden_units=48
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 9
  operator='add'
  confidence_prob=0.8
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 10
  operator='add'
  confidence_prob=0.7
  hidden_units=48
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done

  # 11
  operator='add'
  confidence_prob=0.7
  hidden_units=72
  for i in {1..5..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
    done
}


for j in {1..60..1}
  do
    device_num=$(( $j % 5 ))
    experiment $device_num &
    sleep $sleep_sec
  done
