#!/bin/bash
exec &> log_test_rnn_single_loss.txt

sleep_sec=10
experiment_name='test_rnn_single_loss'
operand_digits=4
hidden_units=64

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
  device_num=$2

  start_echo $operator 'relu'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'relu' $hidden_units $device_num

  start_echo $operator 'tanh'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'tanh' $hidden_units $device_num

  start_echo $operator 'sigmoid'
  python3 rnn_run.py $experiment_name $operand_digits $operator 'sigmoid' $hidden_units $device_num
}

experiment 'add' 0 &
sleep $sleep_sec
experiment 'subtract' 0 &
sleep $sleep_sec
experiment 'divide' 1 &
sleep $sleep_sec
experiment 'modulo' 1 &
sleep $sleep_sec
experiment 'multiply' 0 &
sleep $sleep_sec
