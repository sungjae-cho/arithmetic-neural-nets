exec &> "log_cogsci2019_final.txt"

sleep_sec=10
experiment_name='cogsci2019_final'
operand_digits=4
hidden_units=64

start_echo()
{
  i=$1
  operator=$2
  echo "==================================================================="
  echo "Run the $i-th training of ${operand_digits} digit ${operator}"
}

experiment()
{
  device_num=$1


  operator='subtract'
  for i in {1..1..1}
    do
      python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
    done

}

for j in {1..30..1}
  do
    device_num=$(( $j % 5 ))
    experiment $device_num &
    sleep $sleep_sec
  done
