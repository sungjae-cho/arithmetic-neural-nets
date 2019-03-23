import pickle # import_data, write_measures
import numpy as np # shuffle_np_arrays, get_2d_input
import tensorflow as tf # accuracy_vector_targets, get_fnn_model_name
import os # create_dir
import config
import run_info_utils
from os.path import join
from datetime import datetime


def shuffle_np_arrays(x, y):
    '''
    This only shuffle two numpy arrays along 0-dimension.
    Reference: https://tech.pic-collage.com/tips-of-numpy-shuffle-multiple-arrays-e4fb3e7ae2a
    '''

    # The dimension to shuffle is 0.
    dim_to_shuffle = 0

    # Generate the permutation index array.
    permutation = np.random.permutation(x.shape[dim_to_shuffle])

    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]

    return shuffled_x, shuffled_y


def get_batch(i_batch, batch_size, input_train, output_train):
    # i_bach should start with 0.
    j_start = i_batch * batch_size
    j_end = (i_batch + 1) * batch_size
    batch_input = input_train[j_start:j_end]
    batch_output = output_train[j_start:j_end]

    return batch_input, batch_output


def get_measures(targets, predictions):
    '''
    targets: true target vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    predictions: predicted vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    '''
    n_examples = tf.shape(targets)[0]
    n_dimensions = tf.shape(targets)[1]

    equal = tf.cast(tf.equal(targets, predictions), tf.int32)

    # Measure 1: (target) operation accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    tensor_op_correct = tf.equal(digits_correct, n_dimensions)
    tensor_op_correct = tf.cast(tensor_op_correct, tf.int32)

    op_correct = tf.reduce_sum(tensor_op_correct)
    op_wrong = n_examples - op_correct
    op_accuracy = tf.cast(op_correct, tf.float64) / tf.cast(n_examples, tf.float64)

    # Measure 2: digits_mean_accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    digits_wrong = (tf.ones_like(digits_correct) * n_dimensions) - digits_correct
    digits_mean_correct = tf.reduce_mean(tf.cast(digits_correct, tf.float64))
    digits_mean_wrong = tf.reduce_mean(tf.cast(digits_wrong, tf.float64))
    digits_mean_accuracy = digits_mean_correct / tf.cast(n_dimensions, tf.float64)

    # Measure 3: per_digit_accuracy
    per_digit_correct = tf.reduce_sum(equal, axis=0)
    per_digit_wrong = (tf.ones_like(per_digit_correct) * n_examples) - per_digit_correct
    per_digit_accuracy = per_digit_correct / n_examples

    return (op_accuracy, op_wrong, op_correct,
            digits_mean_accuracy, digits_mean_wrong, digits_mean_correct,
            per_digit_accuracy, per_digit_wrong, per_digit_correct)


def get_dir_sigmoid_output_seq(experiment_name, run_id):
    dir_path = join(config.dir_sigmoid_output_seq(), experiment_name)

    return dir_path


def get_accuracy(targets, predictions):
    '''
    targets: true target vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    predictions: predicted vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    '''
    n_examples = tf.shape(targets)[0]
    n_dimensions = tf.shape(targets)[1]

    equal = tf.cast(tf.equal(targets, predictions), tf.int32)

    # Measure 1: (target) operation accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    tensor_op_correct = tf.equal(digits_correct, n_dimensions)
    tensor_op_correct = tf.cast(tensor_op_correct, tf.int32)

    op_correct = tf.reduce_sum(tensor_op_correct)
    op_wrong = n_examples - op_correct
    op_accuracy = tf.cast(op_correct, tf.float64) / tf.cast(n_examples, tf.float64)

    return op_accuracy


def get_correct_seq(op_correct_stack):
    '''
    Parameters
    -----
    op_correct_stack : tf.Tensor. shape == (n_examples, max_seq_len).

    Returns
    -----
    correct_seq : tf.Tensor. shape == (n_examples)
    '''
    reduced_stack = tf.reduce_sum(op_correct_stack, axis=1)
    correct_seq = tf.cast(tf.not_equal(reduced_stack, 0), tf.int32)

    return correct_seq


def get_seq_wrong(op_correct_stack):
    '''
    Parameters
    -----
    op_correct_stack : tf.Tensor. shape == (n_examples, max_seq_len).

    Returns
    -----
    wrong_seq : tf.Tensor. shape == (n_examples)
    '''
    n_examples = tf.shape(op_correct_stack)[0]
    correct_seq = get_correct_seq(op_correct_stack)
    seq_wrong = n_examples - tf.reduce_sum(correct_seq)

    return seq_wrong


def get_seq_accuracy(op_correct_stack):
    '''
    Parameters
    -----
    op_correct_stack : tf.Tensor. shape == (n_examples, max_seq_len).

    Returns
    -----
    seq_accuracy : tf.Tensor. shape == (n_examples)
    '''
    correct_seq = get_correct_seq(op_correct_stack)
    seq_accuracy = tf.reduce_mean(tf.cast(correct_seq, tf.float32))

    return seq_accuracy

def get_op_correct(targets, predictions):
    '''
    Parameters
    -----
    targets: true target vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    predictions: predicted vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.

    Returns
    -----
    op_correct : numpy.ndarray. shape == (examples).
     - If an example is correct, the value is 1. Otherwise, 0.
     - If the first and last examples are correct, then op_correct becomes [1, ..., 1].

    '''
    n_examples = tf.shape(targets)[0]
    n_dimensions = tf.shape(targets)[1]

    equal = tf.cast(tf.equal(targets, predictions), tf.int32)

    # Measure 1: (target) operation accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    op_correct = tf.equal(digits_correct, n_dimensions)

    return op_correct


def get_correct_first_indices_stat(op_correct_stack):
    '''
    Parameters
    -----
    op_correct_stack : tf.Tensor. shape == (n_examples, max_seq_len).

    Returns
    -----
    (mean_correct_indices,
        std_correct_indices,
        min_correct_indices,
        max_correct_indices) : tf.Tensor. shape == (1)

    This function is adopted from the following link.
    https://stackoverflow.com/questions/42184663/how-to-find-an-index-of-the-first-matching-element-in-tensorflow
    '''
    correct_val = 1
    tmp_indices = tf.where(tf.equal(op_correct_stack, correct_val))
    correct_indices = tf.cast(tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0]), tf.float32)
    no_indices = tf.equal(tf.shape(correct_indices)[0], 0)

    mean_correct_indices = tf.cond(no_indices, lambda: -1.0, lambda: tf.reduce_mean(correct_indices))
    #std_correct_indices = tf.cond(no_indices, lambda: -1.0, lambda: tf.reduce_std(correct_indices)) # Above 1.13 version
    min_correct_indices = tf.cond(no_indices, lambda: -1.0, lambda: tf.reduce_min(correct_indices))
    max_correct_indices = tf.cond(no_indices, lambda: -1.0, lambda: tf.reduce_max(correct_indices))

    return (mean_correct_indices, min_correct_indices, max_correct_indices)


def get_measure_logs_path(run_id, experiment_name):
    create_dir('{}/{}'.format(config.dir_measure_log(), experiment_name))
    pickle_path = '{}/{}/run-{}.pickle'.format(config.dir_measure_log(), experiment_name, run_id)

    return pickle_path


def find_index(tensor, value=1):
    '''
    This function returns the lowest indices of elements that have `value` along
    rows, the second axis (axis=1). If there is no element having `value`, we
    set its index as -1. Indices range from 0 to (tensor.shape[1] - 1).

    Parameters
    -----
    tensor : tf.Tensor. Dimensions == 2.
    value : A particular value to find.

    Returns
    -----
    tensor_indices : tf.Tensor. shape == (tensor.shape[0]).

    I got this implementation idea from the following link.
    https://stackoverflow.com/questions/42184663/how-to-find-an-index-of-the-first-matching-element-in-tensorflow
    '''
    equal_mask = tf.cast(tf.equal(inputs, value), tf.int32)
    reduced_equal_mask = tf.reduce_sum(equal_mask, axis=1)

    no_value_mask = tf.cast(tf.equal(reduced_equal_mask, 0), tf.int32)
    no_value_indices = -no_value_mask

    value_indices = tf.argmax(equal_mask, axis=1, output_type=tf.int32)

    tensor_indices = tf.add(value_indices, no_value_indices)

    return tensor_indices


def get_fnn_model_name(run_id, tfnn_hidden_activation, list_layer_dims, str_optimizer, float_learning_rate, int_batch_size, epoch, str_acc_set, accuracy):

    if tfnn_hidden_activation == tf.nn.sigmoid:
        str_hidden_activation = 'sigm'
    if tfnn_hidden_activation == tf.nn.tanh:
        str_hidden_activation = 'tanh'
    if tfnn_hidden_activation == tf.nn.relu:
        str_hidden_activation = 'relu'

    str_model_name = run_id + '-' + 'fnn' + '-' + str_hidden_activation
    for dim in list_layer_dims:
        str_model_name = str_model_name + '-' + str(dim)

    str_model_name = '%s-%s-lr%f-bs%d-epoch%d-%sacc%.3f'%(str_model_name, str_optimizer, float_learning_rate, int_batch_size, epoch, str_acc_set, accuracy)

    return str_model_name


def dec2bin_embed(integer, n_binary_digits):
    list_embedding = list()
    str_binary = bin(integer)[2:]
    n_leading_zeros = n_binary_digits - len(str_binary)
    str_embedding = str_binary
    if n_leading_zeros > 0:
        str_embedding = '0' * n_leading_zeros + str_binary
    for str_digit in str_embedding:
        list_embedding.append(int(str_digit))
    return list_embedding


def dec2bin_np_embed(integer, n_binary_digits):
    return np.asarray(dec2bin_embed(integer, n_binary_digits), dtype=np.float).reshape(1,n_binary_digits)


def get_fcn_input(decimal_n1, decimal_n2, n_binary_digits):
    '''
    Get the binary input vector of addition of decimal_n1 and decimal_n2
    Parameters
    - decimal_n1, decimal_n1: int in the range of [0,2**(n_binary_digits)-1]
    Return
    - np.ndarry
    - shape = (1, 2 * n_binary_digits)
    - dtype = np.float
    '''
    binary_n1 = dec2bin_embed(decimal_n1, n_binary_digits)
    binary_n2 = dec2bin_embed(decimal_n2, n_binary_digits)
    input_vector = np.asarray(binary_n1 + binary_n2, dtype=np.float).reshape(1,2 * n_binary_digits)

    return input_vector


def get_fcn_target(decimal_n1, decimal_n2, n_binary_digits):
    '''
    Get the binary target vector of addition of decimal_n1 and decimal_n2
    Parameters
    - decimal_n1, decimal_n1: int in the range of [0,2**(n_binary_digits)-1]
    Return
    - np.ndarry
    - shape = (1, n_binary_digits + 1)
    - dtype = np.float
    '''
    decimal_target = decimal_n1 + decimal_n2
    binary_target = dec2bin_embed(decimal_target, n_binary_digits + 1)
    target_vector = np.asarray(binary_target, dtype=np.float).reshape(1, n_binary_digits + 1)

    return target_vector


def decode_fcn_output(np_output):
    '''
    Parameter
    - np_output: numpy.ndarray with shape=(1,n_output_binary_digits)
    '''
    n_output_binary_digits = np_output.shape[1]
    binary_string = ''
    for i in range(n_output_binary_digits):
        binary_string = binary_string + str(int(np_output[0,i]))

    return int(binary_string, base=2)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def tf_tlu(x, name=None):
    return tf.cast(tf.greater(x, 0.5), tf.float32, name=name)


def tf_confidence(x, confidence_prob=0.8, name=None):
    '''
    If predicted probability is over 0.8, then make a positive decicsion.
    If the probability is less than 0.2, then make a negative decision.
    Otherwise, make no decision.
    Say, the first and second cases are in a confident state,
    and the last case is not.

    Parameters
    -----
    x : tf.Tensor. shape == (n_examples, output_dim).
    confidence_prob : float. Probabilty threshold to make a confident answer.
    name : str.

    Returns
    ------
    in_confidence : tf.Tensor. tf.float32 elements. shape == (n_examples).
    '''
    in_upper_side = tf.greater(x, confidence_prob)
    in_lower_side = tf.less(x, 1.0 - confidence_prob)
    in_confidence = tf.cast(tf.logical_or(in_upper_side, in_lower_side), tf.int32)
    in_confidence = tf.cast(tf.reduce_prod(in_confidence, axis=1), tf.float32, name=name)

    return in_confidence


def get_2d_inputs(inputs_1d):
    '''
    inputs_1d
      - shape: (n_examples, dim_input_data_1d)
      - dim_input_data_1d should be even.
    Return: inputs_2d
      - shape: (n_examples, 2, dim_input_data_1d // 2)
    '''
    inputs_2d = np.reshape(inputs_1d, (inputs_1d.shape[0], 2, -1, 1))

    return inputs_2d


def get_seq_data(inputs, targets):
    '''
    Parameters
    -------
    inputs : numpy.ndarray
           shape: (n_input, dim_input)

    outputs : numpy.ndarray
            shape : (n_input, dim_output)
    '''

    n_seq = inputs.shape[0]

    # Inputs
    dim_input = inputs.shape[1]
    n_operand = 2
    dim_operand = dim_input // n_operand

    tmp_inputs = np.reshape(inputs, (n_seq, n_operand, dim_operand))
    tmp_inputs = np.insert(tmp_inputs, 0, 0, axis=2) # Add 0 to the highest digit.
    tmp_inputs = np.flip(tmp_inputs, axis=2) # Axes 0 and 1 do not get flipped.
    seq_inputs = np.transpose(tmp_inputs, axes=(0, 2, 1)) # dim 2 to 1, dim 1 to 2

    # Outputs
    dim_target = targets.shape[1]
    seq_targets = np.reshape(np.flip(targets, axis=1), (n_seq, dim_target, 1))

    return (seq_inputs, seq_targets)


def get_str_activation(tf_activation):
    if tf_activation == tf.nn.sigmoid:
        str_activation = 'sigmoid'
    if tf_activation == tf.nn.tanh:
        str_activation = 'tanh'
    if tf_activation == tf.nn.relu:
        str_activation = 'relu'

    return str_activation


def get_tf_activation(str_activation):
    if str_activation == 'sigmoid':
        tf_activation = tf.nn.sigmoid
    if str_activation == 'tanh':
        tf_activation = tf.nn.tanh
    if str_activation == 'relu':
        tf_activation = tf.nn.relu

    return tf_activation


def init_run_info(NN_OUTPUT_DIM):
    # Initialize an empty dictionary
    run_info = dict()

    # Training info
    run_info['dev/last_loss'] = -1
    run_info['dev/last_accuracy'] = -1
    run_info['dev/last_op_wrong'] = -1
    run_info['dev/last_tlu_loss'] = -1
    run_info['dev/last_tlu_accuracy'] = -1
    run_info['dev/last_tlu_op_wrong'] = -1
    for i in range(NN_OUTPUT_DIM):
        run_info['dev/last_digit-{}_accuracy'.format(i+1)] = -1
        run_info['dev/last_digit-{}_wrong'.format(i+1)] = -1

    ## float epochs
    run_info['last_epoch'] = -1
    run_info['dev/init_all_correct_epoch'] = -1
    for i in range(NN_OUTPUT_DIM):
        run_info['dev/init_all_correct_digit-{}_epoch'.format(i+1)] = -1
        run_info['dev/init_complete_all_correct_digit-{}_epoch'.format(i+1)] = -1

    return run_info


def write_run_info(run_info, float_epoch,
                   dev_run_outputs, dev_tlu_run_outputs,
                   test_run_outputs,
                   dev_carry_run_outputs=None, test_carry_run_outputs=None):

    if run_info['nn_model_type'] == 'mlp':
        (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
         dev_per_digit_accuracy_val, dev_per_digit_wrong_val) = dev_run_outputs
    if run_info['nn_model_type'] == 'rnn':
        (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
            dev_mean_correct_answer_step_val,
            dev_min_correct_answer_step_val,
            dev_max_correct_answer_step_val) = dev_run_outputs
        (test_loss_val, test_accuracy_val, test_op_wrong_val,
            test_mean_correct_answer_step_val,
            test_min_correct_answer_step_val,
            test_max_correct_answer_step_val) = test_run_outputs

    if dev_tlu_run_outputs != None and run_info['nn_model_type'] == 'mlp':
        (dev_loss_tlu_val, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = dev_tlu_run_outputs

    experiment_name = run_info['experiment_name']
    run_id = run_info['run_id']

    # If there is no run_info file, it returns None.
    old_run_info = run_info_utils.get_run_info(run_id, experiment_name)
    if old_run_info != None:
        run_info = old_run_info

    # loss, accuracy, n_wrong
    run_info['dev/last_loss'] = dev_loss_val
    run_info['dev/last_accuracy'] = dev_accuracy_val
    run_info['dev/last_op_wrong'] = dev_op_wrong_val
    run_info['test/last_loss'] = test_loss_val
    run_info['test/last_accuracy'] = test_accuracy_val
    run_info['test/last_op_wrong'] = test_op_wrong_val
    run_info['time/start_time'] = old_run_info['time/start_time'] if old_run_info != None else datetime.now()
    run_info['time/last_time'] = datetime.now()
    run_info['time/running_time'] = run_info['time/last_time'] - run_info['time/start_time']

    if run_info['nn_model_type'] == 'rnn':
        run_info['dev/last_mean_correct_answer_step'] = dev_mean_correct_answer_step_val
        run_info['dev/last_min_correct_answer_step'] = dev_min_correct_answer_step_val
        run_info['dev/last_max_correct_answer_step'] = dev_max_correct_answer_step_val
        run_info['test/last_mean_correct_answer_step'] = test_mean_correct_answer_step_val
        run_info['test/last_min_correct_answer_step'] = test_min_correct_answer_step_val
        run_info['test/last_max_correct_answer_step'] = test_max_correct_answer_step_val

    if dev_tlu_run_outputs != None:
        run_info['dev/last_tlu_loss'] = dev_loss_tlu_val
        run_info['dev/last_tlu_accuracy'] = dev_accuracy_tlu_val
        run_info['dev/last_tlu_op_wrong'] = dev_op_wrong_tlu_val

    if run_info['nn_model_type'] == 'mlp':
        for i in range(len(per_digit_wrong_val)):
            run_info['dev/last_digit-{}_accuracy'.format(i+1)] = dev_per_digit_accuracy_val[-(i+1)]
            run_info['dev/last_digit-{}_wrong'.format(i+1)] = dev_per_digit_wrong_val[-(i+1)]

    if dev_carry_run_outputs != None:
        for n_carries in dev_carry_run_outputs.keys():
            carry_accuracy_val = dev_carry_run_outputs[n_carries][1]
            carry_op_wrong_val = dev_carry_run_outputs[n_carries][2]
            carry_mean_correct_answer_step_val = dev_carry_run_outputs[n_carries][3]
            carry_min_correct_answer_step_val = dev_carry_run_outputs[n_carries][4]
            carry_max_correct_answer_step_val = dev_carry_run_outputs[n_carries][5]
            run_info['dev/last_carry-{}_accuracy'.format(n_carries)] = carry_accuracy_val
            run_info['dev/last_carry-{}_wrong'.format(n_carries)] = carry_op_wrong_val
            run_info['dev/last_carry-{}_mean_correct_answer_step'.format(n_carries)] = carry_mean_correct_answer_step_val
            run_info['dev/last_carry-{}_min_correct_answer_step'.format(n_carries)] = carry_min_correct_answer_step_val
            run_info['dev/last_carry-{}_max_correct_answer_step'.format(n_carries)] = carry_max_correct_answer_step_val

    # float epochs
    run_info['last_epoch'] = float_epoch

    # The float epoch of all correct operation float epoch
    if dev_op_wrong_val == 0 and run_info['dev/init_all_correct_epoch'] == -1:
        run_info['dev/init_all_correct_epoch'] = float_epoch

    # The float epoch of all correct digit
    if run_info['nn_model_type'] == 'mlp':
        for i in range(len(per_digit_wrong_val)):
            # init_all_correct: the initial time to attain all correct digit outputs.
            init_all_correct_key = 'dev/init_all_correct_digit-{}_epoch'.format(i+1)
            # init_complete_all_correct: the last initial time to attain all correct digit outputs.
            init_complete_all_correct_key = 'dev/init_complete_all_correct_digit-{}_epoch'.format(i+1)

            if per_digit_wrong_val[-(i+1)] == 0 and run_info[init_all_correct_key] == -1:
                run_info[init_all_correct_key] = float_epoch
            if per_digit_wrong_val[-(i+1)] == 0 and run_info[init_complete_all_correct_key] == -1:
                run_info[init_complete_all_correct_key] = float_epoch
            if per_digit_wrong_val[-(i+1)] != 0 and run_info[init_complete_all_correct_key] != -1:
                run_info[init_complete_all_correct_key] = -1

    # The float epoch of all carry datasets
    if dev_carry_run_outputs != None:
        for n_carries in dev_carry_run_outputs.keys():
            carry_op_wrong_val = dev_carry_run_outputs[n_carries][2]
            # init_all_correct: the initial time to attain all correct output for `n_carries` dataset.
            init_all_correct_key = 'dev/init_all_correct_carry-{}_epoch'.format(n_carries)
            # init_complete_all_correct: the last initial time to attain all correct output for `n_carries` dataset.
            init_complete_all_correct_key = 'dev/init_complete_all_correct_carry-{}_epoch'.format(n_carries)

            # Initialization step
            if init_all_correct_key not in run_info:
                run_info[init_all_correct_key] = -1
            if init_complete_all_correct_key not in run_info:
                run_info[init_complete_all_correct_key] = -1

            if carry_op_wrong_val == 0 and run_info[init_all_correct_key] == -1:
                run_info[init_all_correct_key] = float_epoch
            if carry_op_wrong_val == 0 and run_info[init_complete_all_correct_key] == -1:
                run_info[init_complete_all_correct_key] = float_epoch
            if carry_op_wrong_val != 0 and run_info[init_complete_all_correct_key] != -1:
                run_info[init_complete_all_correct_key] = -1

    # Recordig for early stopping
    if old_run_info == None:
        run_info['dev/max_accuracy'] = dev_accuracy_val
        run_info['dev/max_accuracy_epoch'] = float_epoch
    else:
        # Candidate early stopping phase.
        if old_run_info['dev/max_accuracy'] < dev_accuracy_val:
            run_info['dev/max_accuracy'] = dev_accuracy_val
            run_info['dev/max_accuracy_epoch'] = float_epoch
            run_info['test/early_stopping/accuracy'] = test_accuracy_val
            if dev_carry_run_outputs != None:
                for n_carries in dev_carry_run_outputs.keys():
                    run_info['dev/carry-{}/early_stopping/accuracy'.format(n_carries)] = dev_carry_run_outputs[n_carries][1]
                    run_info['test/carry-{}/early_stopping/accuracy'.format(n_carries)] = test_carry_run_outputs[n_carries][1]

            if run_info['nn_model_type'] == 'rnn':
                run_info['dev/early_stopping/mean_correct_answer_step'] = dev_mean_correct_answer_step_val
                run_info['dev/early_stopping/min_correct_answer_step'] = dev_min_correct_answer_step_val
                run_info['dev/early_stopping/max_correct_answer_step'] = dev_max_correct_answer_step_val
                run_info['test/early_stopping/mean_correct_answer_step'] = test_mean_correct_answer_step_val
                run_info['test/early_stopping/min_correct_answer_step'] = test_min_correct_answer_step_val
                run_info['test/early_stopping/max_correct_answer_step'] = test_max_correct_answer_step_val

                if dev_carry_run_outputs != None:
                    for n_carries in dev_carry_run_outputs.keys():
                        run_info['test/carry-{}/early_stopping/mean_correct_answer_step'.format(n_carries)] = test_carry_run_outputs[n_carries][3]
                        run_info['test/carry-{}/early_stopping/min_correct_answer_step'.format(n_carries)] = test_carry_run_outputs[n_carries][4]
                        run_info['test/carry-{}/early_stopping/max_correct_answer_step'.format(n_carries)] = test_carry_run_outputs[n_carries][5]


    # Save run_info
    create_dir('{}/{}'.format(config.dir_run_info_experiments(), experiment_name))
    with open('{}/{}/run-{}.pickle'.format(config.dir_run_info_experiments(), experiment_name, run_id), 'wb') as f:
        pickle.dump(run_info, f)

def create_measure_logs(run_info):
    # Create a new measure log dictionary
    measure_logs = dict()
    measure_logs['float_epoch'] = list()
    measure_logs['dev/loss'] = list()
    measure_logs['dev/accuracy'] = list()
    measure_logs['dev/op_wrong'] = list()
    measure_logs['test/loss'] = list()
    measure_logs['test/accuracy'] = list()
    measure_logs['test/op_wrong'] = list()
    for n_carries in run_info['carry_list']:
        measure_logs['dev/carry-{}/accuracy'.format(n_carries)] = list()
        measure_logs['test/carry-{}/accuracy'.format(n_carries)] = list()

    if run_info['nn_model_type'] == 'rnn':
        measure_logs['dev/mean_correct_answer_step'] = list()
        measure_logs['test/mean_correct_answer_step'] = list()
        for n_carries in run_info['carry_list']:
            measure_logs['dev/carry-{}/mean_correct_answer_step'.format(n_carries)] = list()
            measure_logs['test/carry-{}/mean_correct_answer_step'.format(n_carries)] = list()

    if run_info['nn_model_type'] == 'mlp':
        for i in range(len(per_digit_wrong_val)):
            measure_logs['dev/last_digit-{}_accuracy'.format(i+1)] = list()
            measure_logs['dev/digit-{}_op_wrong'.format(i+1)] = list()

    measure_logs['dev/tlu_test_loss'] = list()
    measure_logs['dev/tlu_test_accuracy'] = list()
    measure_logs['dev/tlu_op_wrong'] = list()

    return measure_logs


def write_measures(measure_logs, run_info, float_epoch,
                   dev_run_outputs, dev_tlu_run_outputs,
                   test_run_outputs,
                   dev_carry_run_outputs=None, test_carry_run_outputs=None):

    if run_info['nn_model_type'] == 'mlp':
        (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
         per_digit_accuracy_val, per_digit_wrong_val) = dev_run_outputs
    if run_info['nn_model_type'] == 'rnn':
        (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
            dev_mean_correct_answer_step_val,
            dev_min_correct_answer_step_val,
            dev_max_correct_answer_step_val) = dev_run_outputs
        (test_loss_val, test_accuracy_val, test_op_wrong_val,
            test_mean_correct_answer_step_val,
            test_min_correct_answer_step_val,
            test_max_correct_answer_step_val) = test_run_outputs

    if dev_tlu_run_outputs != None:
        (dev_loss_tlu_val, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = dev_tlu_run_outputs

    run_id = run_info['run_id']
    experiment_name = run_info['experiment_name']

    pickle_path = get_measure_logs_path(run_id, experiment_name)

    if not os.path.exists(pickle_path):
        # Create a new measure log dictionary
        measure_logs = dict()
        measure_logs['float_epoch'] = list()
        measure_logs['dev/loss'] = list()
        measure_logs['dev/accuracy'] = list()
        measure_logs['dev/op_wrong'] = list()
        measure_logs['test/loss'] = list()
        measure_logs['test/accuracy'] = list()
        measure_logs['test/op_wrong'] = list()
        for n_carries in dev_carry_run_outputs.keys():
            measure_logs['dev/carry-{}/accuracy'.format(n_carries)] = list()
            measure_logs['test/carry-{}/accuracy'.format(n_carries)] = list()

        if run_info['nn_model_type'] == 'rnn':
            measure_logs['dev/mean_correct_answer_step'] = list()
            measure_logs['test/mean_correct_answer_step'] = list()
            if dev_carry_run_outputs != None:
                for n_carries in dev_carry_run_outputs.keys():
                    measure_logs['dev/carry-{}/mean_correct_answer_step'.format(n_carries)] = list()
                    measure_logs['test/carry-{}/mean_correct_answer_step'.format(n_carries)] = list()

        if run_info['nn_model_type'] == 'mlp':
            for i in range(len(per_digit_wrong_val)):
                measure_logs['dev/last_digit-{}_accuracy'.format(i+1)] = list()
                measure_logs['dev/digit-{}_op_wrong'.format(i+1)] = list()

        if dev_tlu_run_outputs != None:
            measure_logs['dev/tlu_test_loss'] = list()
            measure_logs['dev/tlu_test_accuracy'] = list()
            measure_logs['dev/tlu_op_wrong'] = list()

    else:
        # Import the measure log dictionary from the pickle file.
        with open(pickle_path, 'rb') as f:
            measure_logs = pickle.load(f)

    # Append a new set of measures
    measure_logs['float_epoch'].append(float_epoch)
    measure_logs['dev/loss'].append(dev_loss_val)
    measure_logs['dev/accuracy'].append(dev_accuracy_val)
    measure_logs['dev/op_wrong'].append(dev_op_wrong_val)
    measure_logs['test/loss'].append(test_loss_val)
    measure_logs['test/accuracy'].append(test_accuracy_val)
    measure_logs['test/op_wrong'].append(test_op_wrong_val)
    for n_carries in dev_carry_run_outputs.keys():
        measure_logs['dev/carry-{}/accuracy'.format(n_carries)].append(dev_carry_run_outputs[1])
        measure_logs['test/carry-{}/accuracy'.format(n_carries)].append(test_carry_run_outputs[1])

    if run_info['nn_model_type'] == 'rnn':
        measure_logs['dev/mean_correct_answer_step'].append(dev_mean_correct_answer_step_val)
        measure_logs['test/mean_correct_answer_step'].append(test_mean_correct_answer_step_val)
        if dev_carry_run_outputs != None:
            for n_carries in dev_carry_run_outputs.keys():
                measure_logs['dev/carry-{}/mean_correct_answer_step'.format(n_carries)].append(dev_carry_run_outputs[3])
                measure_logs['test/carry-{}/mean_correct_answer_step'.format(n_carries)].append(test_carry_run_outputs[3])

    if run_info['nn_model_type'] == 'mlp':
        for i in range(len(per_digit_wrong_val)):
            measure_logs['dev/last_digit-{}_accuracy'.format(i+1)].append(per_digit_accuracy_val[-(i+1)])
            measure_logs['dev/digit-{}_op_wrong'.format(i+1)].append(per_digit_wrong_val[-(i+1)])

    if dev_tlu_run_outputs != None:
        measure_logs['dev/tlu_test_loss'].append(dev_loss_tlu_val)
        measure_logs['dev/tlu_test_accuracy'].append(dev_accuracy_tlu_val)
        measure_logs['dev/tlu_op_wrong'].append(dev_op_wrong_tlu_val)

def save_measure_logs(measure_logs, run_id, experiment_name):
    pickle_path = get_measure_logs_path(run_id, experiment_name)
    # Write measure_logs
    with open(pickle_path, 'wb') as f:
        pickle.dump(measure_logs, f)


def save_sigmoid_output_seq(seq_dict, run_info):
    experiment_name = run_info['experiment_name']
    run_id = run_info['run_id']
    dir_path = get_dir_sigmoid_output_seq(experiment_name, run_id)
    create_dir(dir_path)
    pickle_path = join(dir_path, run_id) + '.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(seq_dict, f)
    print('The final output sequence has been saved in {}.'format(pickle_path))


def read_measure_logs(experiment_name, run_id):
    pickle_path = '{}/{}/run-{}.pickle'.format(config.dir_measure_log(), experiment_name, run_id)
    with open(pickle_path, 'rb') as f:
        measure_logs = pickle.load(f)

    return measure_logs


def read_sigmoid_output_seq(experiment_name, run_id):
    dir_path = get_dir_sigmoid_output_seq(experiment_name, run_id)
    pickle_path = join(dir_path, run_id) + '.pickle'
    with open(pickle_path, 'rb') as f:
        seq_dict = pickle.load(f)

    return seq_dict
