'''
This file is to archive deprecated code.
'''

def write_embeddings_summary(sess, h1):
    # Reference: https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector
    dir_logs = os.path.join(config.dir_saved_models(), experiment_name)
    metadata = os.path.join(dir_logs, 'metadata.tsv')
    carry_datasets = data_utils.import_carry_datasets(operand_bits, operator)
    input_arrays = list()
    with open(metadata, 'w') as f:
        for carries in carry_datasets.keys():
            input_arrays.append(carry_datasets[carries]['input'])
            f.write('{}\n'.format(carries))

    carry_inputs = np.concatenate(input_arrays, axis=0)

    [h1_val] = sess.run([h1],
        feed_dict={inputs:carry_inputs,
                   condition_tlu:False})

    h1_var = tf.Variable(h1_val, name='h1_var')
    saver = tf.train.Saver([h1_var])
    sess.run(h1_var.initializer)
    saver.save(sess, os.path.join(dir_logs, 'h1_var.ckpt'))

    pconfig = projector.ProjectorConfig()
    pconfig.model_checkpoint_path = os.path.join(dir_logs, 'h1_var.ckpt')
    embedding = pconfig.embeddings.add()
    embedding.tensor_name = h1_var.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.summary.FileWriter(dir_logs), pconfig)
