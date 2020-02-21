from utils import read_corpus,pad_sents

from typing import List
from collections import Counter
from itertools import chain
import json

import torch

class VocabEntry(object):
    def __init__(self,word2id=None):
        """
        初始化vocabEntry
        :param word2id: mapping word to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0
            self.word2id['<BOS>'] = 1
            self.word2id['<EOS>'] = 2
            self.word2id['<UNK>'] = 3
        self.unk_id = self.word2id['<UNK>']
        self.id2word = {v:k for k,v in self.word2id.items()}

    def __getitem__(self,word):
        """获取word的idx"""
        return self.word2id.get(word,self.unk_id)

    def __contains__(self,word):
        return word in self.word2id

    def __setitem__(self,key,value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)
    def __repr__(self):

        return 'Vocabulary[size=%d]' % (len(self.word2id))

    def add(self,word):
        """增加word"""
        if word not in self.word2id:
            wid = self.word2id[word] = len(self.word2id)
            self.id2word[wid] = word
            return wid
        else:
            return self.word2id[word]

    def words2indices(self,sents):
        """
        将sents转为number index
        :param sents: list(word) or list(list(wod))
        :return:
        """
        if type(sents[0]) == list:
            return [[self.word2id.get(w,self.unk_id) for w in s] for s in sents]
        else:
            return [self.word2id.get(s,self.unk_id) for s in sents]

    def indices2words(self,idxs):
        return [self.id2word[id] for id in idxs]

    def to_input_tensor(self,sents: List[List[str]], device: torch.device):
        """
        将原始句子list转为tensor,同时将句子PAD成max_len
        :param sents: list of list<str>
        :param device:
        :return:
        """
        sents = self.words2indices(sents)
        sents = pad_sents(sents,self.word2id['<PAD>'])
        sents_var = torch.tensor(sents,device=device)
        return sents_var

    @staticmethod
    def from_corpus(corpus,size,min_feq = 3):
        """从给定语料中创建VocabEntry"""
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = word_freq.most_common(size-4)
        valid_words = [word for word, value in valid_words if value >= min_feq]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), min_feq, len(valid_words)))
        for word in valid_words:
            vocab_entry.add(word)
        return vocab_entry

class Vocab(object):
    """src、tgt的词汇类"""
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, min_feq):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents,vocab_size,min_feq)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents,vocab_size,min_feq)

        return Vocab(src,tgt)

    def save(self,file_path):
        with open(file_path,'w') as fint:
            json.dump(dict(src_word2id=self.src.word2id,tgt_word2id=self.tgt.word2id),fint,indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path,'r') as fout:
            entry = json.load(fout)
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id),VocabEntry(tgt_word2id))
    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))

if __name__ == '__main__':


    src_sents,tgt_sents = read_corpus('./data/train.txt')

    vocab = Vocab.build(src_sents, tgt_sents, 50000, 2)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save('./vocab.json')



