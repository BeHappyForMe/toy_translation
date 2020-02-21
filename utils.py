import nltk
import pkuseg
import math
import random

def read_corpus(file_path):
    """读取语料
    :param file_path:
    :param type:
    :return:
    """
    src_data = []
    tgt_data = []
    seg = pkuseg.pkuseg()
    with open(file_path,'r') as fout:
        for line in fout.readlines():
            pair = line.strip().split('\t')
            src_data.append(nltk.word_tokenize(pair[0].lower()))
            tgt_data.append(['<BOS>'] + seg.cut(pair[1]) + ['<EOS>'])
    return (src_data, tgt_data)

def pad_sents(sents,pad_token):
    """pad句子"""
    sents_padded = []
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    for sent in sents:
        sent_padded = sent + [pad_token] * (max_len - len(sent))
        sents_padded.append(sent_padded)
    return sents_padded

def batch_iter(data, batch_size, shuffle=False):
    """
        batch数据,同时使每个batch里的senten按长度倒序
    :param data: list of tuple
    :param batch_size:
    :param shuffle:
    :return:
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i*batch_size:(i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples,key=lambda x: len(x[0]),reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents