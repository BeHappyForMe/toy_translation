import random
import numpy as np
import pkuseg
import nltk
from nltk.translate.bleu_score import corpus_bleu
import argparse
import os
import math

from tqdm import trange,tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW,get_linear_schedule_with_warmup

from utils import read_corpus, batch_iter
from vocab import Vocab,VocabEntry
from nmt import Encoder,Decoder,Seq2Seq

def setseed():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)


def train(args,model, train_data,dev_data,vocab):
    LOG_FILE = args.output_file
    tb_writer = SummaryWriter('./runs')

    t_total = args.num_epoch * (math.ceil(len(train_data) / args.batch_size))
    optimizer = AdamW(model.parameters(), lr=args.learnning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    global_step = 0
    total_num_words = total_loss = 0.
    logg_loss = 0.
    logg_num_words = 0.
    val_losses = []
    train_epoch = trange(args.num_epoch,desc='train_epoch')
    for epoch in train_epoch:
        model.train()

        for src_sents,tgt_sents in batch_iter(train_data,args.batch_size,shuffle=True):
            batch_size = len(src_sents)
            global_step += 1
            optimizer.zero_grad()

            example_losses = -model(src_sents,tgt_sents).sum()

            # 计算每个单词的loss
            tgt_lengths = [len(sen)-1 for sen in tgt_sents]
            num_words = np.sum(tgt_lengths)
            loss = example_losses / num_words
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * num_words
            total_num_words += num_words

            if global_step % 10 == 0:
                loss_scalar = (total_loss - logg_loss) / (total_num_words-logg_num_words)
                logg_num_words = total_num_words
                logg_loss = total_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, global_step, loss_scalar,
                                                                                       scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, global_step, loss_scalar,
                                                                                scheduler.get_lr()[0]))
                tb_writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if (epoch+1) % 2 == 0:
            eval_loss = evaluate(args,model, dev_data)  # 评估模型
            with open(LOG_FILE, "a") as fout:
                fout.write("EVALUATE: epoch: {}, loss: {}\n".format(epoch,eval_loss))
            if len(val_losses) == 0 or eval_loss < min(val_losses):
                # 如果比之前的loss要小，就保存模型
                print("best model on epoch: {}, val loss: {}".format(epoch,eval_loss))
                torch.save(model.state_dict(), "translate-best.th")
            val_losses.append(eval_loss)


def evaluate(args,model, dev_data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():#不需要更新模型，不需要梯度
        for src_sents,tgt_sents in batch_iter(dev_data,args.batch_size):

            loss = -model(src_sents,tgt_sents).sum()
            num_words = np.sum([len(sen)-1 for sen in tgt_sents])
            total_loss += loss.item()
            total_num_words += num_words

    print("Evaluation loss", total_loss/total_num_words)
    return total_loss/total_num_words

def build_vocab(args):
    if not os.path.exists(args.vocab_path):
        src_sents, tgt_sents = read_corpus(args.train_data_dir)
        vocab = Vocab.build(src_sents, tgt_sents, args.max_vocab_size, args.min_freq)
        vocab.save(args.vocab_path)
    else:
        vocab = Vocab.load(args.vocab_path)
    return vocab

def decode(args,model):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args.test_data_dir))
    test_data_src,test_data_tgt = read_corpus(args.test_data_dir)

    print("load model from {}".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args.beam_size),
                             max_decoding_time_step=args.max_decoding_time_step)

    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score * 100))

    with open(args.args.output_file, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model : Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding'):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    return hypotheses

def compute_corpus_level_bleu_score(references, hypotheses):
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<BOS>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score

def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_dir",default='./data/train.txt',type=str,required=False)
    parse.add_argument("--dev_data_dir", default='./data/dev.txt', type=str, required=False)
    parse.add_argument("--test_data_dir", default='./data/test.txt', type=str, required=False)
    parse.add_argument("--output_file", default='translation_model.log', type=str, required=False)
    parse.add_argument("--batch_size", default=16, type=int)
    parse.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_translate", action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--num_epoch", default=10, type=int)
    parse.add_argument("--max_vocab_size",default=50000,type=int)
    parse.add_argument("--min_freq", default=2, type=int)
    parse.add_argument("--embed_size",default=300,type=int)
    parse.add_argument("--hidden_size", default=512, type=int)
    parse.add_argument("--dropout_rate", default=0.2, type=float)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    parse.add_argument("--vocab_path",default='./vocab.json',type=str)
    parse.add_argument("--model_path", default='translate-best.th', type=str, required=False)
    parse.add_argument("--beam_size", default=5, type=int)
    parse.add_argument("--max_decoding_time_step", default=60, type=int)

    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    setseed()

    train_data = read_corpus(args.train_data_dir)
    train_data = [(src,tgt) for src,tgt in zip(*train_data)]
    dev_data = read_corpus(args.dev_data_dir)
    dev_data = [(src,tgt) for src,tgt in zip(*dev_data)]

    vocab = build_vocab(args)

    encoder = Encoder(vocab,args.embed_size,args.hidden_size,dropout_rate=args.dropout_rate)
    decoder = Decoder(vocab,args.embed_size,args.hidden_size,device,dropout_rate=args.dropout_rate)
    model = Seq2Seq(encoder,decoder,vocab,device,args.hidden_size)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    if args.do_train:
        train(args,model,train_data,dev_data,vocab)

    if args.do_test:
        decode(args,model)

    if args.do_translate:
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)
        while True:
            title = input("请输入要翻译的英文句子:\n")
            if len(title.strip()) == 0:
                continue
            title = nltk.word_tokenize(title.lower())
            title = [title]

            hypotheses = beam_search(model,title,args.beam_size,args.max_decoding_time_step)

            print("翻译后的中文结果为:{}".format(hypotheses))


if __name__ == '__main__':
    main()



