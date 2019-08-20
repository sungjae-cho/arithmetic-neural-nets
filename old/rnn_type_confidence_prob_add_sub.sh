#!/bin/bash
exec &> "log_rnn_type_confidence_prob_add_sub.txt"

sleep_sec=10
experiment_name="rnn_type_confidence_prob_add_sub"
operand_digits=4
hidden_units=64
activation="relu"

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
  operator=$1
  rnn_type=$2
  device_num=$3

  confidence_prob=0.7
  for i in {1..30..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $device_num
    done

  confidence_prob=0.8
  for i in {1..30..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $device_num
    done

  confidence_prob=0.9
  for i in {1..30..1}
    do
      start_echo $operator $rnn_type $confidence_prob
      python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $device_num
    done
}

experiment 'add' 'elman' 1 &
sleep $sleep_sec

experiment 'add' 'jordan' 0 &
sleep $sleep_sec

experiment 'subtract' 'elman' 1 &
sleep $sleep_sec

experiment 'subtract' 'jordan' 0 &
sleep $sleep_sec
