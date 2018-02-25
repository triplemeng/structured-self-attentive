import os, re
import tensorflow as tf
import numpy as np

def get_sentence(vocabulary_inv, sen_index):
    return ' '.join([vocabulary_inv[index] for index in sen_index])

def sequence(rnn_inputs, hidden_size, seq_lens):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build fw cell: '+str(cell_fw))
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build bw cell: '+str(cell_bw))
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw,
                                                               inputs=rnn_inputs,
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32
                                                               )
    print('rnn outputs: '+str(rnn_outputs))
    print('final state: '+str(final_state))

    return rnn_outputs
   
if __name__ == '__main__':
    pass 
