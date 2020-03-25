import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
class Encoder(nn.Module):
    """seq2seq编码器，双向GRU"""
    def __init__(self,vocab,embed_size,hidden_size,dropout_rate=0.2):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(len(vocab.src.word2id),embed_size,vocab.src.word2id['<PAD>'])
        self.rnn = nn.GRU(embed_size,self.hidden_size,bidirectional=True,batch_first=True)
        # 将encodert_t时刻的hidden state变换为decoder初始时刻hidden state的输入
        self.ht_projection = nn.Linear(2*hidden_size,hidden_size,bias=False)

    def forward(self,x,x_lengths):
        """
            encoder编码过程,pad前输入要求是按长度降序的
        :param x:  batch, src_len
        :param x_lengths: batch
        :return:
        """
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,x_lengths,batch_first=True)
        packed_out, last_hidden = self.rnn(packed_embedded)
        enc_hiddens = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True,total_length=max(x_lengths))[0]

        last_hidden = torch.cat([last_hidden[-1],last_hidden[-2]],dim=1)
        dec_init_state = self.ht_projection(last_hidden)

        # batch, src_len, 2h
        # batch, h
        return enc_hiddens,dec_init_state

class Decoder(nn.Module):
    """解码器"""
    def __init__(self,vocab,embed_size,hidden_size,device,dropout_rate=0.2):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(vocab.tgt.word2id),embed_size,vocab.tgt.word2id['<PAD>'])

        # 使用乘性attention,即 enc_hidden * W_atten * dec_cell
        # 即decode的每一步的hidden都与整个输入的hidden做乘性(双线性)变换
        self.att_projection = nn.Linear(2*hidden_size,hidden_size,bias=False)

        self.rnn = nn.GRUCell(embed_size+hidden_size,hidden_size)
        self.combined_output_projection = nn.Linear(3*hidden_size,hidden_size,bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    def forward(self,enc_hiddens,enc_masks,dec_init_state,y):
        """
            解码步骤:
                    1、上个时间步的o_t-1与当前输入做concate
                    2、做一个时间步的rnn，输出当前时间步的h_t
                    3、当前时间步的h_t与encoder的所有hiddens做双线性注意力e_t
                    4、一般encoders有pad的需要mask，替换为较大的负数
                    5、e_t做softmax，概率归一化alpha_t
                    6、当前时间步的hidden state与alpha_t点积，输出当前时间步的注意力输出a_t
                    7、当前注意力输出a_t与当前hidden state做concate后经过linear层、tanh激活、dropout输出当前o_t
                    8、当前的o_t与h_t作为下一个时间步的输入，重复1-8
                    9、所有tgt_len的o_t输出堆叠起来即为decoder的输出
        :param enc_hiddens: encoder 输出的每一个hidden state  【batch,src_len,2h】
        :param enc_masks: 需mask的encoder         【batch,src_len】
        :param dec_init_state: decoder输入的hidden state的初始值 【batch,h】
        :param y: target  【batch, tgt_len】
        :return:
        """
        # 去除最长senten的EOS token
        y = y[:,:-1]
        dec_state = dec_init_state
        batch_size = enc_hiddens.shape[0]
        src_len = enc_hiddens.shape[1]
        # 上一个时间步的combine输出
        o_prev = torch.zeros(batch_size,self.hidden_size,device=self.device)

        combined_outputs = []

        # 双线性atten的前部分  【enc_hiddens  *   w_atten 】 *  dec_hidden_cell
        enc_hiddens_proj = self.att_projection(enc_hiddens.reshape(batch_size*src_len,-1))
        # 拉长变换后再变换回来 【batch,src_len,h】
        enc_hiddens_proj = enc_hiddens_proj.view(batch_size,src_len,-1).contiguous()

        # 【batch, tgt_len, embed_size】
        Y = self.embedding(y)
        # 一步一步迭代
        for Y_t in torch.split(Y,1,dim=1):
            # 【batch,embed_size】
            Y_t = Y_t.squeeze(1)
            # 前一个时间步和当前步tgt输入concat
            Ybar_t = torch.cat((o_prev,Y_t),dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj,enc_masks)
            combined_outputs.append(o_t)

            # 当前时间步的dec_state与o_t作为下一step的输入
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs).transpose(0,1)

        # 【batch, tgt_len, h】
        return combined_outputs


    def step(self,Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj,enc_masks):
        """
            单个step的decode
        :param Ybar_t: 前一个时间步输出及当前时间步输入的concat 【batch, embed+h】
        :param dec_state: 前一个时间步的hidden state 【batch,h】
        :param enc_hiddens: encoder的hiddens  【batch, src_len, 2h】
        :param enc_hiddens_proj: 双线性atten的前部分 enc_hiddens * W_atten * dec_cell 【batch, src_len, h】
        :param enc_masks: 【batch, src_len】
        :return:
        """
        combined_output = None

        # 解码一步
        dec_state = self.rnn(Ybar_t,dec_state)
        # 双线性atten的后半部 enc_hiddens * W_atten * dec_cell  【batch,src_len】
        e_t = torch.bmm(enc_hiddens_proj,dec_state.unsqueeze(2)).squeeze(2)

        if enc_masks is not None:
            # 将需要mask的地方fill为负无穷大。后面接softmax
            e_t.masked_fill_(enc_masks.bool(),-float('inf'))
        #【batch,src_len】注意力score分布
        alpha_t = F.softmax(e_t,dim=1)

        # 点乘enc_hiddens，得到当前时间步的注意力输出 【batch,2h】
        a_t = torch.bmm(alpha_t.unsqueeze(1),enc_hiddens).squeeze(1)
        # 【batch,3*h】
        U_t = torch.cat((a_t,dec_state),dim=1)
        # combine注意力输出与hidden state后加激活及dropout，当前时间步输出。
        # 【batch, tgt_vocab_size】
        O_t = self.dropout(torch.tanh(self.combined_output_projection(U_t)))

        combined_output = O_t

        return dec_state, combined_output, e_t


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,vocab,device,hidden_size):
        super(Seq2Seq,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device

        self.hidden_size = hidden_size
        self.target_vocab_projection = nn.Linear(self.hidden_size,len(vocab.tgt.word2id),bias=False)


    def forward(self,x,y):
        x_lengths = [len(sen) for sen in x]
        x_padded = self.vocab.src.to_input_tensor(x,self.device)    #   【batch,src_len】
        y_padded = self.vocab.tgt.to_input_tensor(y, self.device)   #   【batch,tgt_len】

        x_masks = self.generate_sent_masks(x_padded,x_lengths).to(self.device)
        enc_hiddens, hidden_state = self.encoder(x_padded,x_lengths)

        # 【batch, tgt_len, h】
        combined_outputs = self.decoder(enc_hiddens,x_masks,hidden_state,y_padded)

        # 输出翻译后的单词概率分布矩阵 【batch, tgt, vocab_size】
        P = F.log_softmax(self.target_vocab_projection(combined_outputs),dim=-1)

        # 对翻译输出的pad做mask 【batch,tgt_len】
        y_masks = (y_padded != self.vocab.tgt.word2id['<PAD>']).float()

        true_tgt_y = y_padded[:,1:]  # 翻译需要错位
        # 【batch,tgt_len】 负对数似然即交叉熵损失
        tgt_log_prob = torch.gather(P,index=true_tgt_y.unsqueeze(2),dim=-1).squeeze(2) * y_masks[:,1:]
        score = torch.sum(tgt_log_prob,dim=0)

        return score

    def beam_search(self,src_sent,beam_size,max_decoding_time_step = 100):
        """
        beam_search解码单个src sent
        :param src_sent: list of word
        :param beam_size:
        :param max_decoding_time_step: 最长解码长度
        :return: 返回一个list of 候选译文，每个译文包含一下:
                value: List[str]: 译文句子内容
                score: float: 译文的对数似然概率
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent],self.device)
        src_hid_encodings,dec_init_hid = self.encoder(src_sents_var,[len(src_sent)])
        # 双线性att的前半部 [batch=1, src_len, h]
        src_encod_att_linear = self.decoder.att_projection(src_hid_encodings)

        h_tm1 = dec_init_hid
        EOS_id = self.vocab.tgt.word2id['<EOS>']
        hypotheses = [['<BOS>']]
        hyp_scores = torch.zeros(len(hypotheses),dtype=torch.float,device=self.device)

        # dec阶段第一个初始化的输入,与输入的word embedding concate后输入
        att_tm1 = torch.zeros(len(hypotheses),self.hidden_size,device=self.device)
        h_tm1 = dec_init_hid

        completed_hypotheses = []

        t = 0
        # beam seach第一步留下topk，其余步k^2，留下topk
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t+=1
            hyp_num = len(hypotheses)
            # 扩展成batch
            exp_src_encodings = src_hid_encodings.expand(hyp_num,
                                                         src_hid_encodings.shape[1],
                                                         src_hid_encodings.shape[2])
            exp_src_encod_att_linear = src_encod_att_linear.expand(hyp_num,
                                                                   src_encod_att_linear.shape[1],
                                                                   src_encod_att_linear.shape[2])
            # 取前一步解码出的word 【hyp_num】
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses],dtype=torch.long,device=self.device)
            # 【hyp_num, embed_size】
            y_t_embed = self.decoder.embedding(y_tm1)
            # 【hyp_num, embed+h】
            Ybar_t = torch.cat((y_t_embed,att_tm1),dim=-1)
            # 单个句子/解码无需
            h_t, att_t, _ = self.decoder.step(Ybar_t,h_tm1,exp_src_encodings,exp_src_encod_att_linear,None)

            # 译文对数似然概率分布 【hyp_num, vocab_size】
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t),dim=-1)
            # 减去已经beam search结束的
            live_hyp_num = beam_size - len(completed_hypotheses)
            #
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores,k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '<EOS>':
                    # 搜寻终止
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = h_t[live_hyp_ids]
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


    def generate_sent_masks(self,x_padded,x_lengths):
        """对encoder中的pad做mask处理"""
        enc_masks = torch.zeros(x_padded.shape[0],x_padded.shape[1],dtype=torch.float)
        for idx,length in enumerate(x_lengths):
            enc_masks[idx,length:] = 1
        return enc_masks


