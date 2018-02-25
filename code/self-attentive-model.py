import os
import re
import argparse
import numpy as np
import pickle as pl
import pandas as pd
import tensorflow as tf
from components import sequence, get_sentence

def build_graph(
    inputs,
    hidden_size = 64,
    da = 128,
    r = 30,
    atten_size = 50,
    nclasses = 2,
    embeddings = None
    ):

    print("inputs shape: "+str(inputs.shape))
    # inputs shape: (?, 200)
    
    rev_length = (int)(inputs.shape[1])
    print("review length: "+str(rev_length))
    # review length: 200
    
    _, embedding_size = embeddings.shape
    print("embedding size: "+str(embedding_size))
    # embedding size: 50
    
    word_rnn_inputs = tf.nn.embedding_lookup( tf.convert_to_tensor(embeddings, np.float32), inputs)
    print("word rnn inputs: "+str(word_rnn_inputs))
    # word rnn inputs: Tensor("embedding_lookup:0", shape=(?, 200, 50), dtype=float32)
    
    reuse_value = None
    
    with tf.variable_scope("word_rnn", reuse=reuse_value):
        word_rnn_outputs = sequence(word_rnn_inputs, hidden_size, None)
    # rnn outputs: (<tf.Tensor 'word_rnn/bidirectional_rnn/fw/fw/transpose:0' shape=(?, 200, 64) dtype=float32>, <tf.Tensor 'word_rnn/ReverseV2:0' shape=(?, 200, 64) dtype=float32>)
    # final state: (<tf.Tensor 'word_rnn/bidirectional_rnn/fw/fw/while/Exit_2:0' shape=(?, 64) dtype=float32>, <tf.Tensor 'word_rnn/bidirectional_rnn/bw/bw/while/Exit_2:0' shape=(?, 64) dtype=float32>)
    print("word rnn ouputs: "+str(word_rnn_outputs))
    # word rnn ouputs: (<tf.Tensor 'word_rnn/bidirectional_rnn/fw/fw/transpose:0' shape=(?, 200, 64) dtype=float32>, <tf.Tensor 'word_rnn/ReverseV2:0' shape=(?, 200, 64) dtype=float32>)
    
    #now combine the outputs:
    atten_inputs = tf.concat(word_rnn_outputs, 2)
    print("attention inputs: "+str(atten_inputs))
    # attention inputs: Tensor("concat:0", shape=(?, 200, 128), dtype=float32)

    with tf.variable_scope("weights", reuse=reuse_value):
        Ws1 = tf.get_variable(name="Ws1", dtype=tf.float32, shape=[da, hidden_size*2])
        Ws2 = tf.get_variable(name="Ws2", dtype=tf.float32, shape=[r, da])
        
    print("Ws1: "+str(Ws1))
    # Ws1: <tf.Variable 'weights/Ws1:0' shape=(128, 128) dtype=float32_ref>
    
    print("Ws2: "+str(Ws2)) 
    # Ws2: <tf.Variable 'weights/Ws2:0' shape=(128, 30) dtype=float32_ref>
    
    Ws1H = tf.map_fn(lambda x: tf.matmul(Ws1, tf.transpose(x)), atten_inputs)
    print("Ws1H: "+str(Ws1H))
    # Ws1H: Tensor("map_2/TensorArrayStack/TensorArrayGatherV3:0", shape=(?, 128, 200), dtype=float32)
   
    A = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(Ws2, tf.tanh(x)), Ws1H))
    print("attention matrix: "+str(A))
    # attention matrix: Tensor("Reshape_1:0", shape=(?, 30, 200), dtype=float32)

    M = tf.matmul(A, atten_inputs)
    print("attended matrix M: "+str(M))
    # attended matrix M: Tensor("MatMul:0", shape=(?, 30, 128), dtype=float32)
    
    M = tf.reshape(M, [-1, r*2*hidden_size])
    print("attended matrix M after reshape: "+str(M))
    
    d1 = 20
    with tf.variable_scope("MLP", reuse=reuse_value):
        W1 = tf.get_variable(name="W1", dtype=tf.float32, shape=[r*2*hidden_size, d1])
        b1 = tf.get_variable(name="b1", dtype=tf.float32, shape=[d1])
    
    o1 = tf.nn.relu(tf.matmul(M,  W1) + b1)
    
    with tf.variable_scope("out_weights", reuse=reuse_value) as out:
        weights_out = tf.get_variable(name="output_w", dtype=tf.float32, shape=[d1, nclasses])
        biases_out = tf.get_variable(name="output_bias", dtype=tf.float32, shape=[nclasses])
    dense = tf.matmul(o1, weights_out) + biases_out
    print("dense: "+str(dense))
        
    # attention matrix after reshape: Tensor("Reshape_1:0", shape=(?, 30, 200), dtype=float32)
    # attended matrix before reshape: Tensor("MatMul_2:0", shape=(?, 30, 128), dtype=float32)
    # attended matrix after reshape: Tensor("Reshape_3:0", shape=(?, 3840), dtype=float32)
    # dense: Tensor("add_2:0", shape=(?, 2), dtype=float32)
    return dense, A

def initialize(working_dir, log_dir):
    train_filename = os.path.join(working_dir, "train_df_file")
    test_filename = os.path.join(working_dir, "test_df_file")
    emb_filename = os.path.join(working_dir, "emb_matrix")
    print("load dataframe for training...")
    df_train = pd.read_pickle(train_filename)
    df_train = df_train.dropna(axis=0, how='any')
    print("df train shape: "+str(df_train.shape))
    print('review len: '+str(len(df_train['review'].iloc[0])))
    rev_length = len(df_train['review'].iloc[0])
    print("load dataframe for testing...")
    df_test = pd.read_pickle(test_filename)
    df_test = df_test.dropna(axis=0, how='any')
    print("df test shape: "+str(df_test.shape))
    print("load embedding matrix...")
    (emb_matrix, word2index, index2word) = pl.load(open(emb_filename, "rb"))
    return df_train, df_test, emb_matrix, word2index, index2word

if __name__ == '__main__':
    train_batch_size = 2048
    resume = False
    epochs = 1
    nclasses = 2
    working_dir = "../data/aclImdb"
    log_dir = "../logs"
    df_train, df_test, emb_matrix, word2index, index2word = initialize(working_dir, log_dir)

    rev_length = len(df_train['review'].iloc[0])
    y_ = tf.placeholder(tf.int32, shape=[None, nclasses])
    inputs = tf.placeholder(tf.int32, [None, rev_length])

    dense, A = build_graph(inputs, embeddings=emb_matrix, nclasses=nclasses)

    penalty_weight = 0.0
    r = int(A.shape[1])
    cross_entropy_n = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=dense)
    penalty_matrix = tf.matmul(A, tf.transpose(A, perm=[0, 2, 1])) - tf.eye(r)
    penalty_n = penalty_weight * tf.square(tf.norm(penalty_matrix, ord='fro', axis=(1,2)))
    loss = tf.reduce_mean(cross_entropy_n + penalty_n)

    with tf.variable_scope('optimizers', reuse=None):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    y_predict = tf.argmax(dense, 1)
    correct_prediction = tf.equal(y_predict, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: "+str(accuracy))
    saver = tf.train.Saver()
    tf.summary.scalar("cost", loss)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()
    on_value = 1
    off_value = 0

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_batch = (int)(df_train.values.shape[0]/train_batch_size)
        for epoch in range(0, 12):
            avg_cost = 0.0
            print("epoch {}".format(epoch))
            for i in range(total_batch):
                train_sample = df_train.sample(train_batch_size)
                batch_data = np.asarray(train_sample['review'].values.tolist()).reshape(train_batch_size, rev_length)
    #                 print("batch data shape: "+str(batch_data.shape))
                batch_label = train_sample['label'].tolist()
                
                batch_label_formatted = tf.one_hot(indices=batch_label, depth=nclasses, on_value=on_value, off_value=off_value, axis=-1)
            
                batch_labels = sess.run(batch_label_formatted)
                    
                feed = {inputs: batch_data, y_: batch_labels}
                _, c, summary_in_batch_train = sess.run([optimizer, loss, summary_op], feed_dict=feed)
                avg_cost += c/total_batch
    #                 train_writer.add_summary(summary_in_batch_train, epoch*total_batch + i)
    #             saver.save(sess, os.path.join(log_dir, "model.ckpt"), epoch, write_meta_graph=False)
            print("avg cost in the training phase epoch {}: {}".format(epoch, avg_cost))
    
        print("evaluating...")
    
        x_test = np.asarray(df_test['review'].tolist())
        y_test = df_test['label'].values.tolist()
        test_batch_size = 1000
        total_batch2 = int(len(x_test)/(test_batch_size))
        avg_accu = 0.0
    
        for i in range(total_batch2):
            batch_x = x_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_y = y_test[i*test_batch_size:(i+1)*test_batch_size]
            batch_label_formatted2 =tf.one_hot(indices=batch_y, depth=nclasses, on_value=on_value, off_value=off_value, axis=-1)
        
            batch_labels2 = sess.run(batch_label_formatted2)
            feed = {inputs: batch_x, y_: batch_labels2}
            accu  = sess.run(accuracy, feed_dict=feed)
            avg_accu += 1.0*accu/total_batch2
            print("avg accuracy: "+str(avg_accu))
