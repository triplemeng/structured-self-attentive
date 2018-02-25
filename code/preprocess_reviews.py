import os, re
import argparse
import numpy as np
import pickle as pl
from os import walk
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from tensorflow.contrib.keras import preprocessing

def build_emb_matrix_and_vocab(embedding_model, keep_in_dict=10000, embedding_size=50):
    # 0 th element is the default vector for unknowns.
    emb_matrix = np.zeros((keep_in_dict+2, embedding_size))
    word2index = {}
    index2word = {}
    for k in  range(1, keep_in_dict+1):
        word = embedding_model.wv.index2word[k-1]
        emb_matrix[k] = embedding_model[word]
        word2index[word] = k
        index2word[k] = word
    word2index['UNK'] = 0
    index2word[0] = 'UNK'
    word2index['STOP'] = keep_in_dict+1 
    index2word[keep_in_dict+1] = 'STOP'
    return emb_matrix, word2index, index2word

def sent2index(sent, word2index):
    words = sent.strip().split(' ')
    sent_index = [word2index[word] if word in word2index else 0 for word in words]
    return sent_index

def get_sentence(index2word, sen_index):
    return ' '.join([index2word[index] for index in sen_index])

def gen_data(data_dir, word2index):
    data = []
    tokenizer = RegexpTokenizer(r'\w+')
    for  filename in os.listdir(data_dir):
        file = os.path.join(data_dir, filename)
        with open(file) as f:
            content = f.readline().lower()
            sent = ' '.join(tokenizer.tokenize(content))
            sent_index = sent2index(sent, word2index)
            data.append(sent_index)
    return data

def preprocess_review(data, required_rev_len, keep_in_dict=10000):
    ## As the result, each review will be composed of required_rev_len words. If the original review is longer than that, we truncate it, and if shorter than that, we append STOP to it. 
    data_formatted = []
    review_lens = []
    for review in data:
        review_len = len(review)
        lack_len = required_rev_len - review_len
        if lack_len > 0:
            review_formatted_right_len = review.extend([0]*lack_len)
        elif lack_len < 0:
            review_formatted_right_len = review[:required_rev_len]
        data_formatted.append(review_formatted_right_len)
    return data_formatted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-r', '--required_rev_length', type=int, default=200,
                   help='fix the maximum review length')

    args = parser.parse_args()
    required_rev_length = args.required_rev_length

    print('rev length is set as {}'.format(required_rev_length))

    working_dir = "../data/aclImdb"
    fname = os.path.join(working_dir, "imdb_embedding")
    train_dir = os.path.join(working_dir, "train")
    train_pos_dir = os.path.join(train_dir, "pos")
    train_neg_dir = os.path.join(train_dir, "neg")
    test_dir = os.path.join(working_dir, "test")
    test_pos_dir = os.path.join(test_dir, "pos")
    test_neg_dir = os.path.join(test_dir, "neg")

    if os.path.isfile(fname):
        embedding_model = Word2Vec.load(fname)
    else:
        print("please run gen_word_embeddings.py first to generate embeddings!")
        exit(1)
    print("generate word to index dictionary and inverse dictionary...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)
    print("format each review into sentences, and also represent each word by index...")
    train_pos_data = gen_data(train_pos_dir, word2index)
    train_neg_data = gen_data(train_neg_dir, word2index)
    train_data = train_neg_data + train_pos_data
    test_pos_data = gen_data(test_pos_dir, word2index)
    test_neg_data = gen_data(test_neg_dir, word2index)
    test_data = test_neg_data + test_pos_data

    print("preprocess each review...")
    x_train = preprocess_review(train_data, required_rev_length)
    print(len(x_train[1]))
    x_test = preprocess_review(test_data, required_rev_length)
    y_train = [0]*len(train_neg_data)+[1]*len(train_pos_data)
    y_test = [0]*len(test_neg_data)+[1]*len(test_pos_data)

    print("save word embedding matrix ...")
    emb_filename = os.path.join(working_dir, "emb_matrix")
#     #emb_matrix.dump(emb_filename)
    pl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb"))

    print("save review data for training...")
    df_train = pd.DataFrame({'review':x_train, 'label':y_train})
    train_filename = os.path.join(working_dir, "train_df_file")
    df_train.to_pickle(train_filename)

    print("save review data for testing...")
    df_test = pd.DataFrame({'review':x_test, 'label':y_test})
    test_filename = os.path.join(working_dir, "test_df_file")
    df_test.to_pickle(test_filename)
