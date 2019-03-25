import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import utils
import data_utils
from datetime import datetime
import os
import pickle
import sys
import config
import gc # garbage collector interface
import tracemalloc

def main():
    experiment_name = sys.argv[1]
    operand_bits =  int(sys.argv[2])
    operator =  sys.argv[3]
    rnn_type = sys.argv[4]
    str_activation = sys.argv[5]
    hidden_units =  int(sys.argv[6])
    confidence_prob = float(sys.argv[7])
    max_steps = int(sys.argv[8])
    str_device_num = str(int(sys.argv[9]))
    nn_model_type = 'rnn'
    on_tlu = config.on_tlu()
    mlp_run(experiment_name, operand_bits, operator, rnn_type, str_activation,
        hidden_units, confidence_prob, max_steps, str_device_num, nn_model_type, on_tlu)


def mlp_run(experiment_name, operand_bits, operator, rnn_type, str_activation,
    hidden_units, confidence_prob, max_steps, str_device_num, nn_model_type, on_tlu):

    def train(sess, batch_input, batch_target, float_epoch, all_correct_val):
        _, _, _ = sess.run([loss, op_accuracy, train_op],
                            feed_dict={inputs:batch_input, targets:batch_target,
                                       condition_tlu:False,
                                       training_epoch:float_epoch,
                                       big_batch_training:big_batch_training_val,
                                       all_correct_epoch:(all_correct_val * float_epoch),
                                       all_correct:all_correct_val})

    def write_train_summary(sess, compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step):
        # Run computing train loss, accuracy
        train_loss, train_accuracy, merged_summary_op_val = sess.run(
            compute_nodes,
            feed_dict={inputs:batch_input, targets:batch_target,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("epoch: {}, step: {}, train_loss: {}, train_accuracy: {}".format(epoch, step, train_loss, train_accuracy))
        train_summary_writer.add_summary(merged_summary_op_val, step)

        return (train_loss, train_accuracy)

    def write_dev_summary(sess, compute_nodes, float_epoch, all_correct_val, step):

        [dev_loss, dev_accuracy, merged_summary_op_val, dev_op_wrong_val,
            dev_mean_answer_step_val,
            dev_min_answer_step_val,
            dev_max_answer_step_val] = sess.run(
            compute_nodes,
            feed_dict={inputs:input_dev, targets:target_dev,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("└ epoch: {}, step: {}, dev_loss: {}, dev_accuracy: {}, op_wrong: {}".format(epoch, step, dev_loss, dev_accuracy, op_wrong_val))
        dev_summary_writer.add_summary(merged_summary_op_val, step)

        return (dev_loss, dev_accuracy, dev_op_wrong_val,
            dev_mean_answer_step_val,
            dev_min_answer_step_val,
            dev_max_answer_step_val)

    def write_tlu_dev_summary(sess, compute_nodes, float_epoch, all_correct_val, step):
        [dev_loss_tlu, dev_accuracy_tlu, merged_summary_op_val, dev_op_wrong_val_tlu,
            _, _, _] = sess.run(
            compute_nodes,
            feed_dict={inputs:input_dev, targets:target_dev,
                       condition_tlu:True,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("└ [TLU] epoch: {}, step: {}, dev_loss: {}, dev_accuracy: {}, op_wrong: {}".format(epoch, step, dev_loss_tlu, dev_accuracy_tlu, op_wrong_val_tlu))
        tlu_summary_writer.add_summary(merged_summary_op_val, step)

        return (dev_loss_tlu, dev_accuracy_tlu, dev_op_wrong_val_tlu)

    def write_test_summary(sess, compute_nodes, float_epoch, all_correct_val, step):
        [test_loss, test_accuracy, merged_summary_op_val, op_wrong_val,
            test_mean_answer_step_val,
            test_min_answer_step_val,
            test_max_answer_step_val] = sess.run(
            compute_nodes,
            feed_dict={inputs:input_test, targets:target_test,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})
        #print("└ epoch: {}, step: {}, test_loss: {}, test_accuracy: {}, op_wrong: {}".format(epoch, step, test_loss, test_accuracy, op_wrong_val))
        test_summary_writer.add_summary(merged_summary_op_val, step)

        return (test_loss, test_accuracy, op_wrong_val,
            test_mean_answer_step_val,
            test_min_answer_step_val,
            test_max_answer_step_val)

    def write_carry_datasets_summary(sess, compute_nodes, float_epoch, all_correct_val, step, dataset_type='dev'):
        value_dict = dict()
        for n_carries in splited_carry_datasets.keys():
            carry_dataset_input = splited_carry_datasets[n_carries]['input'][dataset_type]
            carry_dataset_output = splited_carry_datasets[n_carries]['output'][dataset_type]

            [carry_loss_val, carry_accuracy_val, merged_summary_op_val, carry_op_wrong_val,
                carry_mean_answer_step_val,
                carry_min_answer_step_val,
                carry_max_answer_step_val] = sess.run(
                compute_nodes,
                feed_dict={inputs:carry_dataset_input, targets:carry_dataset_output,
                           condition_tlu:False,
                           training_epoch:float_epoch,
                           big_batch_training:big_batch_training_val,
                           all_correct_epoch:(all_correct_val * float_epoch),
                           all_correct:all_correct_val})

            value_dict[n_carries] = (carry_loss_val, carry_accuracy_val, carry_op_wrong_val,
                carry_mean_answer_step_val,
                carry_min_answer_step_val,
                carry_max_answer_step_val)
            if config.on_carry_datasets_summary(dataset_type):
                carry_datasets_summary_writers[n_carries][dataset_type].add_summary(merged_summary_op_val, step)

        return value_dict

    def write_h1_summary(sess, h1, run_id, float_epoch, init_all_correct=False):
        dir_h1_logs = os.path.join(config.dir_h1_logs(), experiment_name)
        utils.create_dir(dir_h1_logs)

        carry_datasets = data_utils.import_carry_datasets(operand_bits, operator)
        input_arrays = list()
        output_arrays = list()
        carry_arrays = list()

        for carries in carry_datasets.keys():
            input_array = carry_datasets[carries]['input']
            output_array = carry_datasets[carries]['output']
            n_examples = input_array.shape[0]
            input_arrays.append(input_array)
            output_arrays.append(output_array)
            carry_arrays.append(np.full((n_examples), carries, dtype=np.int))

        np_inputs = np.concatenate(input_arrays, axis=0)
        np_outputs = np.concatenate(output_arrays, axis=0)
        np_carry_labels = np.concatenate(carry_arrays, axis=0)

        # Get h1 values.
        [h1_val] = sess.run([h1],
            feed_dict={inputs:np_inputs,
                       condition_tlu:False})

        return_dict = dict()
        return_dict['input'] = np_inputs
        return_dict['carry'] = np_carry_labels
        return_dict['output'] = np_outputs
        return_dict['h1'] = h1_val
        return_dict['operator'] = operator

        if init_all_correct:
            file_name = '{}_init_all_correct.pickle'.format(run_id, int(float_epoch))
        else:
            file_name = '{}_ep{}.pickle'.format(run_id, int(float_epoch))
        with open(os.path.join(dir_h1_logs, file_name), 'wb') as f:
            pickle.dump(return_dict, f)


    def create_carry_datasets_summary_writers(logdir, carry_datasets):
        carry_datasets_summary_writers = dict()
        for n_carries in carry_datasets.keys():
            carry_datasets_summary_writers[n_carries] = dict()
            carry_datasets_summary_writers[n_carries]['train'] =  tf.summary.FileWriter(logdir + '/train/carry-{}'.format(n_carries))
            carry_datasets_summary_writers[n_carries]['dev'] =  tf.summary.FileWriter(logdir + '/dev/carry-{}'.format(n_carries))
            carry_datasets_summary_writers[n_carries]['test'] =  tf.summary.FileWriter(logdir + '/test/carry-{}'.format(n_carries))
        return carry_datasets_summary_writers

    def close_carry_datasets_summary_writers(carry_datasets_summary_writers):
        for n_carries in carry_datasets_summary_writers.keys():
            carry_datasets_summary_writers[n_carries]['train'].close()
            carry_datasets_summary_writers[n_carries]['dev'].close()
            carry_datasets_summary_writers[n_carries]['test'].close()

    def get_all_correct_val(op_wrong_val):
        if op_wrong_val == 0:
            return True
        else:
            return False

    def is_last_batch(i_batch):
        if i_batch == (n_batch - 1):
            return True
        else:
            return False

    def decrease_dev_summary_period(dev_accuracy_val, op_wrong_val):
        # Preconditions
        if not decreasing_dev_summary_period:
            return
        if dev_accuracy_val < 0.999:
            return

        # If the preconditions are satisfied, ...
        if op_wrong_val <= 8:
            dev_summary_period = int(init_dev_summary_period // 128)
        elif op_wrong_val <= 16:
            dev_summary_period = int(init_dev_summary_period // 64)
        if op_wrong_val <= 32:
            dev_summary_period = int(init_dev_summary_period // 32)
        elif op_wrong_val <= 64:
            dev_summary_period = int(init_dev_summary_period // 16)
        elif op_wrong_val <= 128:
            dev_summary_period = int(init_dev_summary_period // 8)

        if op_wrong_val > 512:
            dev_summary_period = init_dev_summary_period

    def compute_sigmoid_output_seq(sess, run_info, sigmoid_outputs_series, float_epoch, all_correct_val):
        seq_dict = dict()
        for n_carries in splited_carry_datasets.keys():
            carry_dataset_input = splited_carry_datasets[n_carries]['input']['test']
            carry_dataset_output = splited_carry_datasets[n_carries]['output']['test']
            sigmoid_outputs_series_val = sess.run(
                sigmoid_outputs_series,
                feed_dict={inputs:carry_dataset_input, targets:carry_dataset_output,
                           condition_tlu:False,
                           training_epoch:float_epoch,
                           big_batch_training:big_batch_training_val,
                           all_correct_epoch:(all_correct_val * float_epoch),
                           all_correct:all_correct_val})
            #print("└ epoch: {}, step: {}, test_loss: {}, test_accuracy: {}, op_wrong: {}".format(epoch, step, test_loss, test_accuracy, op_wrong_val))
            seq_dict[n_carries] = dict()
            seq_dict[n_carries]['output_seq'] = sigmoid_outputs_series_val
            seq_dict[n_carries]['input'] = carry_dataset_input
            seq_dict[n_carries]['output'] = carry_dataset_output

        utils.save_sigmoid_output_seq(seq_dict, run_info)

    ############################################################################
    # Running point.

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]= str_device_num # 0, 1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs: Unable to display GPU info when running on the bash

    # Import datasets
    (train_ratio, dev_ratio, test_ratio) = config.dataset_ratio()
    (input_train, input_dev, input_test,
        target_train, target_dev, target_test,
        splited_carry_datasets
    ) = data_utils.import_op_dataset(operator, operand_bits,
            train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio)

    # Contants
    if nn_model_type == 'mlp':
        NN_INPUT_DIM = input_train.shape[1]
    if nn_model_type == 'rnn':
        if rnn_type == 'jordan':
            NN_INPUT_DIM = input_train.shape[1] + target_train.shape[1]
        if rnn_type == 'elman':
            NN_INPUT_DIM = input_train.shape[1] + hidden_units
    NN_OUTPUT_DIM = target_train.shape[1]

    # Hyperparameters - training
    batch_size = config.batch_size()
    big_batch_size = config.big_batch_size()
    n_epoch = config.n_epoch()
    learning_rate = config.learning_rate()
    all_correct_stop = config.all_correct_stop()
    big_batch_saturation = config.big_batch_saturation()
    if big_batch_saturation:
        all_correct_stop = False

    # Hyperparameters - model
    #activation = config.activation() # tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu
    #str_activation = utils.get_str_activation(activation)
    activation = utils.get_tf_activation(str_activation)
    h_layer_dims = [hidden_units] # h_layer_dims[0]: dim of h1 layer
    last_size = NN_OUTPUT_DIM

    # Variables determined by other variables
    train_size = input_train.shape[0]
    n_batch = train_size // batch_size

    # Print periods
    if (n_batch // 4) != 0:
        train_summary_period = n_batch // 4 # 4 times per epoch
    else:
        train_summary_period = 1
    init_dev_summary_period = n_batch # n_batch: print at every epoch
    dev_summary_period = init_dev_summary_period
    decreasing_dev_summary_period = config.decreasing_dev_summary_period()

    # Weight initialization
    ## https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer
    if activation == tf.nn.relu:
        init_factor = 2.0
    if activation == tf.nn.sigmoid:
        init_factor = 1.0
    if activation == tf.nn.tanh:
        init_factor = 1.0

    fan_in_1 = NN_INPUT_DIM
    fan_in_2 = h_layer_dims[0]

    ############################################################################
    # Creating a computational graph.

    # Initializing paraters to learn.
    with tf.name_scope('parameter'):
        W1 = tf.Variable(tf.truncated_normal((NN_INPUT_DIM, h_layer_dims[0]), stddev=np.sqrt(init_factor / fan_in_1)), name="W1")
        b1 = tf.Variable(tf.zeros((h_layer_dims[0])), name="b1")
        W2 = tf.Variable(tf.truncated_normal((h_layer_dims[0], NN_OUTPUT_DIM), stddev=np.sqrt(init_factor / fan_in_2)), name="W2")
        b2 = tf.Variable(tf.zeros((NN_OUTPUT_DIM)), name="b2")

    # Setting the input and target output.
    inputs = tf.placeholder(tf.float32, shape=(None, input_train.shape[1]), name='inputs') # None for batch_size. This is variable because of different size of train and test sets.
    targets = tf.placeholder(tf.float32, shape=(None, target_train.shape[1]), name='targets')

    condition_tlu = tf.placeholder(tf.int32, shape=(), name="tlu_condition")
    is_tlu_hidden = tf.greater(condition_tlu, tf.constant(0, tf.int32))
    #is_tlu_hidden = tf.constant(condition_tlu == True, dtype=tf.bool) # https://github.com/pkmital/tensorflow_tutorials/issues/36

    # Creating a graph for a MLP ###############################################
    if nn_model_type == 'mlp':

        # NN structure
        with tf.name_scope('layer1'):
            h1_logits = tf.add(tf.matmul(inputs,  W1), b1)
            h1 = tf.cond(is_tlu_hidden, lambda: utils.tf_tlu(h1_logits, name='h1_tlu'), lambda: activation(h1_logits, name='h1')) # https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph / https://www.tensorflow.org/versions/r1.7/api_docs/python/tf/cond
        with tf.name_scope('layer2'):
            last_logits = tf.add(tf.matmul(h1,  W2), b2)
            sigmoid_outputs = tf.sigmoid(last_logits)
        predictions = utils.tf_tlu(sigmoid_outputs, name='predictions')

        # Loss: objective function
        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=last_logits) # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
            loss = tf.reduce_mean(loss)

        # Get measures:
        # [1] operation measures (accuracy, n_wrong, n_correct)
        # [2] mean digits accuracy (mean_digits_accuracy)
        # [3] per digit accuracy (per_digit_accuracy)
        (op_accuracy, op_wrong, op_correct,
         digits_mean_accuracy, digits_mean_wrong, digits_mean_correct,
         per_digit_accuracy, per_digit_wrong, per_digit_correct
        ) = utils.get_measures(targets, predictions)

    # Creating a graph for a MLP ###############################################

    # Creating a graph for a Jordan RNN ###############################################
    if nn_model_type == 'rnn':
        init_output_val = 0.5 # 0.5 means being uncertain about decision of 0 or 1.
        if rnn_type == 'jordan':
            sigmoid_outputs = tf.fill(tf.shape(targets), init_output_val, name="sigmoid_outputs")
        if rnn_type == 'elman':
            h1 = tf.zeros(shape=[tf.shape(targets)[0], hidden_units])
        # confidence_mask stands for
        # whether the network has faced any confident prediction at the previous steps.
        # 0 means it has faced a confident prediction, and 1 does not.
        confidence_mask = tf.ones(tf.shape(targets)[0])

        # Forward pass
        last_logits_series = []
        answer_mask_series = [] # To make answer_step_indices
        answer_masked_last_logits_series = []
        opt_masked_last_logits_series = []
        sigmoid_outputs_series = []

        # Sequential computation
        for t in range(max_steps):
            # t varies from 0 to (max_time - 1)
            # RNN at step t.
            if rnn_type == 'jordan':
                input_and_prob_concat = tf.concat([inputs, sigmoid_outputs], axis=1)  # Increasing number of columns
                input_to_h1 = input_and_prob_concat
            if rnn_type == 'elman':
                input_and_h1_concat = tf.concat([inputs, h1], axis=1) # Increasing number of columns
                input_to_h1 = input_and_h1_concat

            with tf.name_scope('layer1'):
                h1 = activation(tf.add(tf.matmul(input_to_h1, W1), b1))  # Broadcasted addition

            with tf.name_scope('layer2'):
                last_logits = tf.add(tf.matmul(h1,  W2), b2)
                last_logits_series.append(last_logits)
                sigmoid_outputs = tf.sigmoid(last_logits, name='sigmoid_outputs_step_{}'.format(t))
                sigmoid_outputs_series.append(sigmoid_outputs)
            ##### Jordan RNN at step t.

            # Compute answer_mask. #####
            if t < max_steps - 1:
                # All steps except the last step.
                # confidence : whether the network is confident at the current step.
                confidence = utils.tf_confidence(sigmoid_outputs, confidence_prob=confidence_prob)
                # answer_mask : whether the network answers at the current step.
                answer_mask = confidence_mask * confidence
            if t == max_steps - 1:
                if config.on_single_loss():
                    # answer_mask : whether the network answers at the current step.
                    # The last last step
                    # If there is no confident prediction until the step right before the last step,
                    # answer at the last step.
                    answer_mask = confidence_mask
                else:
                    # answer_mask : whether the network answers at the current step.
                    answer_mask = confidence_mask * confidence
                answer_mask = confidence_mask * confidence
            # confidence_mask : whether the network has been confident.
            # 1 for not being answered. 0 for being answered.
            confidence_mask = tf.cast(tf.not_equal(confidence_mask, answer_mask), tf.float32)

            answer_mask_series.append(answer_mask)

            # answer_mask_2d : the 2-dimensional tensor of answer_mask.
            answer_mask_2d = tf.reshape(answer_mask, (tf.shape(answer_mask)[0], -1))
            # answer_mask_2d is element-wise producted with the current last_logits.
            answer_masked_last_logits = answer_mask_2d * last_logits
            if t < max_steps - 1:
                opt_masked_last_logits = answer_mask_2d * last_logits
            else:
                last_confidence_mask = confidence_mask
                confidence_mask_2d = tf.reshape(last_confidence_mask, (tf.shape(confidence_mask)[0], -1))
                opt_masked_last_logits = (answer_mask_2d + confidence_mask_2d) * last_logits
            answer_masked_last_logits_series.append(answer_masked_last_logits)
            opt_masked_last_logits_series.append(opt_masked_last_logits)

        # Make answer_last_logits that contains last_logits of all answers.
        answer_masked_last_logits_stack = tf.stack(answer_masked_last_logits_series, axis=0)
        opt_masked_last_logits_stack= tf.stack(opt_masked_last_logits_series, axis=0)
        # reduce_sum in the direction of time steps (axis=0).
        # answer_last_logits.shape == [n_examples, output_dim]
        answer_last_logits = tf.reduce_sum(answer_masked_last_logits_stack, axis=0)
        opt_last_logits = tf.reduce_sum(opt_masked_last_logits_stack, axis=0)
        # Get predictions of all last_logits
        answer_sigmoid_outputs = tf.sigmoid(answer_last_logits)
        opt_sigmoid_outputs = tf.sigmoid(opt_last_logits)
        answer_predictions = utils.tf_tlu(answer_sigmoid_outputs, name='answer_predictions')
        opt_predictions = utils.tf_tlu(opt_sigmoid_outputs, name='opt_predictions')

        # Make answer_step_indices. #####
        # answer_mask_stack : shape == [max_steps, n_examples].
        ## 1 means being answered and 0 means not being answered, at a certain step.
        answer_mask_stack = tf.stack(answer_mask_series, axis=0)
        # total_answer_mask : shape = [n_examples].
        ## 1 means being answered and 0 means not being answered, throughout all steps.
        total_answer_mask = tf.reduce_sum(answer_mask_stack, axis=0)
        answer_step_indices = tf.cast(tf.argmax(answer_mask_stack, axis=0), tf.float32) + total_answer_mask - tf.ones(tf.shape(targets)[0])
        # Get correct_answer_step_indices.
        answer_correctness = utils.get_op_correct(targets, answer_predictions, total_answer_mask)
        correct_answer_step_indices = tf.boolean_mask(answer_step_indices, answer_correctness)
        # Get statistics of answer_step_indices.
        mean_correct_answer_step_indices = tf.reduce_mean(correct_answer_step_indices)
        min_correct_answer_step_indices = tf.reduce_min(correct_answer_step_indices)
        max_correct_answer_step_indices = tf.reduce_max(correct_answer_step_indices)

        (op_accuracy, op_wrong, op_correct,
         digits_mean_accuracy, digits_mean_wrong, digits_mean_correct,
         per_digit_accuracy, per_digit_wrong, per_digit_correct
        ) = utils.get_answered_measures(targets, opt_predictions, total_answer_mask)

        # Loss: objective function
        with tf.name_scope('loss'):
            if config.on_single_loss():
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=opt_last_logits) # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
            else:
                losses = [tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits) for logits in last_logits_series]
                loss = tf.stack(losses, axis=0)
            loss = tf.reduce_mean(loss)
    # Creating a graph for a Jordan RNN ###############################################

    # Weight regularization part
    with tf.name_scope('loss'):
        if config.l1_coef() != 0:
            loss = loss \
                + config.l1_coef() / (2 * batch_size) * (tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)))
            #    + config.l1_coef() / (2 * batch_size) * (tf.reduce_sum(tf.abs(tf.abs(W1) - 1)) + tf.reduce_sum(tf.abs(tf.abs(W2) - 1)))
        if config.l2_coef() != 0:
            loss = loss \
                + config.l2_coef() / (2 * batch_size) * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))

    # Training, optimization
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    training_epoch = tf.placeholder(tf.float32, shape=None)
    all_correct_epoch = tf.placeholder(tf.float32, shape=None)
    big_batch_training = tf.placeholder(tf.int32, shape=None)
    all_correct = tf.placeholder(tf.int32, shape=None)

    # Summary: Scalar
    ## Measures
    tf.summary.scalar('loss', loss)

    with tf.name_scope('operation'):
        tf.summary.scalar('accuracy', op_accuracy)
        tf.summary.scalar('wrong', op_wrong)

    tf.summary.scalar('epoch', training_epoch)
    tf.summary.scalar('all_correct_epoch', all_correct_epoch)
    tf.summary.scalar('big_batch_training', big_batch_training)
    tf.summary.scalar('all_correct', all_correct)
    tf.summary.scalar('condition_tlu', condition_tlu)

    # Summary: Histogram
    with tf.name_scope('layer1'):
        tf.summary.histogram('weight', W1)
        tf.summary.histogram('bias', b1)
        tf.summary.histogram('activation', h1)
    with tf.name_scope('layer2'):
        tf.summary.histogram('weight', W2)
        tf.summary.histogram('bias', b2)
        tf.summary.histogram('activation', sigmoid_outputs)

    if nn_model_type == 'mlp':
        with tf.name_scope('digits'):
            tf.summary.scalar('mean_accuracy', digits_mean_accuracy)
            tf.summary.scalar('mean_wrong', digits_mean_wrong)

        with tf.name_scope('per_digit'):
            for i in range(NN_OUTPUT_DIM):
                tf.summary.scalar('digit-{}/accuracy'.format(i+1), per_digit_accuracy[-(i+1)])
                tf.summary.scalar('digit-{}/wrong'.format(i+1), per_digit_wrong[-(i+1)])
                # add per_digit_correct

    if nn_model_type == 'rnn':
        with tf.name_scope('correct_answer_step_indices'):
            tf.summary.scalar('mean', mean_correct_answer_step_indices)
            #tf.summary.scalar('std', std_correct_indices)
            tf.summary.scalar('min', min_correct_answer_step_indices)
            tf.summary.scalar('max', max_correct_answer_step_indices)

    # Merge summary operations
    merged_summary_op = tf.summary.merge_all()

    run_info = utils.init_run_info(NN_OUTPUT_DIM)

    # Experiment info
    run_info['experiment_name'] = experiment_name

    # Problem info
    run_info['operator'] = operator
    run_info['operand_bits'] = operand_bits
    run_info['result_bits'] = target_train.shape[1]

    # Network info
    run_info['nn_model_type'] = nn_model_type
    if nn_model_type == 'rnn':
        run_info['rnn_type'] = rnn_type
        run_info['confidence_prob'] = confidence_prob
    run_info['network_input_dimension'] = input_train.shape[1]
    run_info['network_output_dimension'] = target_train.shape[1]
    run_info['hidden_activation'] = str_activation
    run_info['hidden_dimensions'] = h_layer_dims
    run_info['max_steps'] = max_steps


    # Dataset info
    run_info['train_dev_test_ratio'] = config.dataset_ratio()
    run_info['train_set_size'] = input_train.shape[0]
    run_info['dev_set_size'] = input_dev.shape[0]
    run_info['test_set_size'] = input_test.shape[0]
    for carries in splited_carry_datasets.keys():
        run_info['train_set_size/carry-{}'.format(carries)] = splited_carry_datasets[carries]['input']['train'].shape[0]
        run_info['dev_set_size/carry-{}'.format(carries)] = splited_carry_datasets[carries]['input']['dev'].shape[0]
        run_info['test_set_size/carry-{}'.format(carries)] = splited_carry_datasets[carries]['input']['test'].shape[0]
    run_info['carry_list'] = list(splited_carry_datasets.keys())

    # Optimizer info
    run_info['batch_size'] = batch_size
    run_info['optimizer'] = train_op.name
    run_info['learning_rate'] = learning_rate
    run_info['all_correct_stop'] = all_correct_stop

    run_id = datetime.now().strftime('%Y%m%d%H%M%S')
    run_info['run_id'] = run_id

    # Train logging
    logdir = '{}/{}/{}_{}bit_{}_{}_h{}_run-{}/'.format(
        config.dir_logs(), experiment_name, operator, operand_bits, nn_model_type, str_activation, h_layer_dims, run_id)

    train_summary_writer = tf.summary.FileWriter(logdir + '/train', graph=tf.get_default_graph())
    dev_summary_writer = tf.summary.FileWriter(logdir + '/dev')
    if on_tlu:
        tlu_summary_writer = tf.summary.FileWriter(logdir + '/tlu')
    test_summary_writer = tf.summary.FileWriter(logdir + '/test')
    if operator in config.operators_list():
        carry_datasets_summary_writers = create_carry_datasets_summary_writers(logdir, splited_carry_datasets)

    # Model saving
    dir_saved_model = '{}/{}/{}_{}bit_{}_{}_h{}/run-{}/'.format(
        config.dir_saved_models(), experiment_name, operator, operand_bits, nn_model_type, str_activation, h_layer_dims, run_id)
    utils.create_dir(dir_saved_model)

    model_saver = tf.train.Saver()
    init_all_correct_model_saver = tf.train.Saver()

    # Compute nodes
    train_compute_nodes = [loss, op_accuracy, merged_summary_op]
    #dev_compute_nodes = [loss, op_accuracy, merged_summary_op, op_wrong, per_digit_accuracy, per_digit_wrong]
    dev_compute_nodes = [loss, op_accuracy, merged_summary_op, op_wrong,
        mean_correct_answer_step_indices, min_correct_answer_step_indices, max_correct_answer_step_indices]
    test_compute_nodes = [loss, op_accuracy, merged_summary_op, op_wrong,
        mean_correct_answer_step_indices, min_correct_answer_step_indices, max_correct_answer_step_indices]

    # Session configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    print("\nRun ID: {}".format(run_id))
    print(logdir)
    print(dir_saved_model)

    with tf.Session(config=tf_config) as sess:
        sess.run(init)

        float_epoch = 0.0
        all_correct_val = False
        big_batch_training_val = False
        init_all_correct_model_saved = False

        measure_logs = utils.create_measure_logs(run_info)
        #tracemalloc.start()
        #snapshot_now = tracemalloc.take_snapshot()
        for epoch in range(n_epoch):
            '''if epoch % 100 == 0:
                snapshot_before = snapshot_now
                snapshot_now = tracemalloc.take_snapshot()
                diff_stats = snapshot_now.compare_to(snapshot_before, 'lineno')
                for stat in diff_stats[:10]:
                    print(stat)
                print('===================================================')'''

            input_train, target_train = utils.shuffle_np_arrays(input_train, target_train)

            if big_batch_saturation and all_correct_val:
                big_batch_training_val = True
                batch_size = big_batch_size

            for i_batch in range(n_batch):
                # Get mini-batch
                batch_input, batch_target = utils.get_batch(i_batch, batch_size, input_train, target_train)

                # Initial state evalutation: No training
                if epoch == 0 and i_batch == 0:
                    step = 0
                    float_epoch = 0.0

                    write_train_summary(sess, train_compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step)
                    write_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)
                    write_test_summary(sess, test_compute_nodes, float_epoch, all_correct_val, step)
                    #write_h1_summary(sess, h1, run_id, float_epoch)
                    if on_tlu:
                        write_tlu_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)


                # Set step, float_epoch
                ## 1 <= (i_batch + 1) <= n_batch
                step = n_batch * epoch + (i_batch + 1)
                float_epoch = epoch + float(i_batch + 1) / n_batch

                # Training operation ##################################################################
                train(sess, batch_input, batch_target, float_epoch, all_correct_val)

                # training set summary writer###########################################################
                if step % train_summary_period == 0:
                    (train_loss, train_accuracy) = write_train_summary(sess, train_compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step)

                #if float_epoch % config.period_h1_log() == 0:
                #    write_h1_summary(sess, h1, run_id, float_epoch)

                # Development loss evalution
                # After dev_summary_period batches are trained
                if (step % dev_summary_period == 0) or is_last_batch(i_batch):
                    # dev set summary writer#############################################################
                    dev_run_outputs = write_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)
                    (_, dev_accuracy_val, dev_op_wrong_val, _, _, _) = dev_run_outputs
                    test_run_outputs = write_test_summary(sess, test_compute_nodes, float_epoch, all_correct_val, step)

                    # carry datasets summary writer #####################################################
                    dev_carry_run_outputs = write_carry_datasets_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step, 'dev')
                    test_carry_run_outputs = write_carry_datasets_summary(sess, test_compute_nodes, float_epoch, all_correct_val, step, 'test')
                    write_carry_datasets_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step, 'train')


                    # TLU-dev summary writer#############################################################
                    # on_tlu
                    if on_tlu:
                        dev_tlu_run_outputs = (_, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = write_tlu_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)
                    else:
                        dev_tlu_run_outputs = None

                    # Write running information################################
                    # Write the logs of measures################################

                    utils.write_run_info(run_info, float_epoch,
                                        dev_run_outputs, dev_tlu_run_outputs,
                                        test_run_outputs,
                                        dev_carry_run_outputs, test_carry_run_outputs)

                    utils.write_measures(measure_logs, run_info, float_epoch,
                                        dev_run_outputs, dev_tlu_run_outputs,
                                        test_run_outputs,
                                        dev_carry_run_outputs, test_carry_run_outputs)

                    if is_last_batch(i_batch):
                        # After one epoch is trained
                        # Save the trained model ################################################
                        model_saver.save(sess, '{}/dev-{}.ckpt'.format(dir_saved_model, run_id))
                        ##print("Model saved.")
                        # decrease_dev_summary_period

                    #decrease_dev_summary_period(dev_accuracy_val, dev_op_wrong_val)

                    # If there is no wrong operation, then ...
                    all_correct_val = get_all_correct_val(dev_op_wrong_val)

                    # If the model is  trained with 100% accuracy,
                    if all_correct_val and (not init_all_correct_model_saved):
                        # Save the model.
                        model_name = 'epoch{}-batch{}'.format(float_epoch, i_batch)
                        init_all_correct_model_saver.save(sess, '{}/{}-init-all-correct.ckpt'.format(
                            dir_saved_model, model_name))
                        #write_h1_summary(sess, h1, run_id, float_epoch, True)
                        init_all_correct_model_saved = True

                    if all_correct_val and all_correct_stop:
                        break # Break the batch for-loop

            # End of one epoch
            if all_correct_val and all_correct_stop:
                break # Break the epoch for-loop
            gc.collect()

        # End of all epochs

        # Test loss evalution
        # Run computing test loss, accuracy
        # test set summary writer#############################################################
        (test_loss, test_accuracy, test_op_wrong_val,
            test_mean_answer_step_val,
            test_min_answer_step_val,
            test_max_answer_step_val
            ) = write_test_summary(sess, test_compute_nodes, float_epoch, all_correct_val, step)

        # carry datasets summary writer #####################################################
        #carry_run_outputs = write_carry_datasets_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step, 'test')

        compute_sigmoid_output_seq(sess, run_info, sigmoid_outputs_series, float_epoch, all_correct_val)

        model_saver.save(sess, '{}/{}.ckpt'.format(dir_saved_model, run_id))
        print("Model saved.")

    utils.save_measure_logs(measure_logs, run_id, experiment_name)

    train_summary_writer.close()
    dev_summary_writer.close()
    if on_tlu:
        tlu_summary_writer.close()
    test_summary_writer.close()
    if operator in config.operators_list():
        close_carry_datasets_summary_writers(carry_datasets_summary_writers)

    print("The training is over.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
