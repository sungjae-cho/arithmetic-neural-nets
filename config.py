import numpy as np
import tensorflow as tf

def operators_list():
    return ['add', 'subtract', 'multiply', 'divide', 'modulo']

def operand_digits_list():
    return [4, 6, 8]

def np_type():
    return np.int

def dir_data():
    return 'data'

def dir_saved_models():
    return 'saved_models'
    #return dir_logs()

def dir_plot_fig():
    return 'plot_figures'

def dir_logs():
    return 'tb_logs'

def dir_run_info_experiments():
    return 'run_info'

def dir_measure_log():
    return 'measure_logs'

def dir_h1_logs():
    return 'h1_logs'

def dir_sigmoid_output_seq():
    return 'sigmoid_output_seq'

def dir_result_statistics():
	return 'result_statistics'

def period_h1_log():
    # return epoch period
    return 100

def carry_dataset_statistics_name():
    return 'carry_dataset_statistics.csv'

def batch_size():
    return 32

def big_batch_size():
    return batch_size() * (2**4) # 32 * (2**4) == 512

def n_epoch():
    return 999999999999

def learning_rate():
    return 0.001

def all_correct_stop():
    return True

def big_batch_saturation():
    return False

def activation():
    return tf.nn.relu # tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu

def decreasing_dev_summary_period():
    return False

def l1_coef():
    return 0.000

def l2_coef():
    return 0.000

def dataset_ratio():
    (train_ratio, dev_ratio, test_ratio) = (1.0, 0, 0)
    #(train_ratio, dev_ratio, test_ratio) = (0.8*0.8, 0.8*0.2, 0.2)
    return train_ratio, dev_ratio, test_ratio

def on_tlu():
    return False

def on_carry_datasets_summary(dataset_type):
    if dataset_type == 'train':
        return False
    if dataset_type == 'dev':
        return False
    if dataset_type == 'test':
        return False

def on_single_loss():
    return False

#def max_time():
#    return 30

#def rnn_type():
#    return 'elman'
    #return 'jordan'

'''operators_list = ['add', 'subtract', 'multiply', 'divide', 'modulo']
operand_digits_list = [4, 6, 8]
np_type = np.int
dir_data = 'data'
dir_saved_models = 'saved_models'
dir_plot_fig = 'plot_figures'
dir_logs = 'tb_logs'
dir_run_info_experiments = 'run_info'
dir_measure_log = 'measure_logs'
carry_dataset_statistics_name = 'carry_dataset_statistics.csv'

# Hyperparameters - training
batch_size = 32
big_batch_size = batch_size * (2**4) # 32 * (2**4) == 512
n_epoch = 999999999999
learning_rate = 0.001
all_correct_stop = True
big_batch_saturation = False

activation = tf.nn.sigmoid # tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu

decreasing_dev_summary_period = False
'''
