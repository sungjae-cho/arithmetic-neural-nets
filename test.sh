experiment_name='test'
operand_digits=4
operator='add'
rnn_type='jordan'
activation='relu'
hidden_units=64
confidence_prob=0.9
max_steps=30
device_num=1

python3 rnn_run.py $experiment_name $operand_digits $operator $rnn_type $activation $hidden_units $confidence_prob $max_steps $device_num
