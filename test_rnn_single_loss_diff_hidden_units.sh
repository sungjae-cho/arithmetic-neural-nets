#!/bin/bash
exec &> log_test_rnn_single_loss_diff_hidden_units.txt

sleep_sec=10
experiment_name='test_rnn_single_loss_diff_hidden_units'
operand_digits=4
#hidden_units=64

start_echo()
{
  operator=$1
  activation=$2
  echo "==================================================================="
  echo "Run training a single-loss RNN."
  echo "Operator: ${operator}"
  echo "Hidden activation: ${activation}"
  echo "Hidden units: ${hidden_units}"
  echo "==================================================================="
}

experiment()
{
  operator=$1
  hidden_units=$2
  device_num=$3

  start_echo $operator 'relu'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'relu' $hidden_units $device_num

  start_echo $operator 'tanh'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'tanh' $hidden_units $device_num

  start_echo $operator 'sigmoid'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'sigmoid' $hidden_units $device_num
}

experiment 'add' 64 0 &
sleep $sleep_sec
experiment 'add' 32 1 &
sleep $sleep_sec
experiment 'add' 16 1 &
sleep $sleep_sec
experiment 'add' 8 0 &

