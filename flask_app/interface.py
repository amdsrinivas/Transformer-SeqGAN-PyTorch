import sys
sys.path.insert(0, '../seq_gan_with_attention')
sys.path.insert(0, '../seq_gan')
sys.path.insert(0, '../core') 

from helper import *
from generator_attention import Generator_attention as GenAttention
from generator import Generator as GenSeqgan
from data_iter import GenDataIter, DisDataIter
import test_attention_only

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# from nltk.translate.bleu_score import corpus_bleu


class interactive_demo:
    def __init__(self, test_sentence=None):
        self.CHECKPOINT_PATH = './checkpoints/'
        self.TEST_FILE       = './raw_test.data'
        BATCH_SIZE           = 10
        self.cuda            = True
        # self.idx_to_word, self.word_to_idx, self.VOCAB_SIZE, g_emb_dim, g_hidden_dim, g_sequence_len = load_vocab(self.CHECKPOINT_PATH)
        self.gen_attention_metadata = load_vocab(self.CHECKPOINT_PATH+'/seq_gan_with_attention/checkpoint_2/metadata.data')

        # Seq GAN with Attention
        self.gen_attention = GenAttention(self.gen_attention_metadata['vocab_size'], self.gen_attention_metadata['g_emb_dim'],
                    self.gen_attention_metadata['g_hidden_dim'], self.gen_attention_metadata['g_sequence_len'], BATCH_SIZE, self.cuda, test_mode = True)
        self.gen_attention = self.gen_attention.cuda()
        self.gen_attention.load_state_dict(torch.load(self.CHECKPOINT_PATH+'/seq_gan_with_attention/checkpoint_2/generator.model', map_location={'cuda:1': 'cpu'}))
        self.gen_attention = self.gen_attention.cuda()

        # Seq GAN based generator
        self.seq_gan_metadata = load_vocab(self.CHECKPOINT_PATH + '/seq_gan/metadata.data')
        self.seq_gan = GenSeqgan(self.seq_gan_metadata['vocab_size'], self.seq_gan_metadata['g_emb_dim'],
                    self.seq_gan_metadata['g_hidden_dim'], self.cuda)

        self.seq_gan = self.seq_gan.cuda()
        self.seq_gan.load_state_dict(torch.load(self.CHECKPOINT_PATH+'/seq_gan/generator_seqgan.model', map_location={'cuda:1': 'cpu'}))
        self.seq_gan = self.seq_gan.cuda()

    def predict_for_all(self, test_sentence = None):
        attention_out = self.demo(self.gen_attention, self.gen_attention_metadata, test_sentence)[0]
        print('Output of Attention based' , attention_out)
        seqgan_out = self.demo(self.seq_gan, self.seq_gan_metadata, test_sentence)[0]
        print(len(seqgan_out))
        print('Output of Seq based' , seqgan_out)
        return attention_out, seqgan_out
        #  return attention_out
 
    def demo(self, model, metadata, test_sentence):
        # idx_to_word, word_to_idx, VOCAB_SIZE = load_vocab(CHECKPOINT_PATH)
        self.test_sentence = test_sentence
        self.max_test_sentence_len = 10

        # Get padded sentencels
        self.padded_sentence = pad_sentences(self.test_sentence, self.max_test_sentence_len)
        # print(self.padded_sentence)

        # Get ids of padded sentence
        padded_sent_ids = get_ids(self.padded_sentence, metadata['idx_to_word'], metadata['word_to_idx'], metadata['vocab_size'])
        # print(padded_sent_ids)

        # print("&"*80, padded_sent_ids)
        # return [["NONE"]]
        # Write to temporary file
        out_file = self.TEST_FILE
        fp = open(out_file, "w")
        fp.writelines(["%s " % item  for item in padded_sent_ids])
        fp.close()

        test_iter = GenDataIter(self.TEST_FILE, 1)
        return self.test_predict(model, test_iter, metadata)

    def test_predict(self, model, data_iter, metadata):
        data_iter.reset()
        ret_pred = []
        for (data, target) in data_iter:
            data = Variable(data, volatile=True)
            target = Variable(target, volatile=True)
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            prob = model.forward(data)
            # print('PROBS SH:',  prob.shape)
            # mini_batch = prob.shape[0]
            if len(prob.shape) > 2:
                prob = torch.reshape(prob, (prob.shape[0] * prob.shape[1], -1))
            predictions = torch.max(prob, dim=1)[1]
            # print(predictions)
            predictions = predictions.view(1, -1)
            # print(predictions)
            # print('PRED SHAPE:' , predictions.shape)
            for each_sen in list(predictions):
                # print(each_sen)
                sent_from_id = generate_sentence_from_id(metadata['idx_to_word'], each_sen)
                # print('Sample Output:', sent_from_id)
                ret_pred.append(sent_from_id)
            sys.stdout.flush()
             
        data_iter.reset()
        return ret_pred


    def get_references(self, data_iter):
        ret_pred= []
        for (data, target) in data_iter:
            data = data.view(1, -1)
            for each_sen in list(data):
                
                sent_from_id = generate_sentence_from_id(self.seq_gan_metadata['idx_to_word'], each_sen)
                # print('Sample Output:', sent_from_id)
                ret_pred.append(sent_from_id)
        return ret_pred

    # Helper for calculating BLEU score
    def calc_bleu(self):
        data_file = '../seq_gan_with_attention/real.data'
        
        # metadata = load_vocab(self.CHECKPOINT_PATH + '/seq_gan/metadata.data')
        # idx_to_word = metadata['idx_to_word']
        train_data_iter = GenDataIter(data_file, 1)
        reference = self.get_references(train_data_iter)

        
        # pred = self.test_predict(self.seq_gan, train_data_iter, self.seq_gan_metadata)
        train_data_iter = GenDataIter(data_file, 1)
        candidates_ga = self.test_predict(self.gen_attention, train_data_iter, self.seq_gan_metadata)

        train_data_iter = GenDataIter(data_file, 1)
        candidates_sg = self.test_predict(self.seq_gan, train_data_iter, self.seq_gan_metadata)
        
        # train_data_iter = GenDataIter(data_file, 1)

        candidates_ao = []
        for sentence in reference:
            s = " ".join(sentence)
            
            # s = "kindergarten is great"
            sent = test_attention_only.test(s)
            candidates_ao.append(sent)

            
        # candidates_sg = self.test_predict(self.seq_gan, train_data_iter, self.seq_gan_metadata)

        # CALC BLEU

        # references = 
        # candidates_words = []
        # for id in candidates:
        #     candidates_words.append(idx_to_word[])
        # load 
        # candidatates_id to words 
        # [0, 12] => [['a', 'aaa']]

        sen_score = 0
        for sent in candidates_ga:
            sen_score += sentence_bleu(reference, sent)
        print("@@@@@@@@@@@@@@@@", sen_score)# / len(candidates_ga))

        # print('Individual 3-gram: %f' % corpus_bleu(reference, candidates_ga, weights=(1, 0, 0, 0)))
        print('Individual 3-gram: %f' % sen_score)
        # print('Individual 4-gram: %f' % corpus_bleu(reference, candidates_ga, weights=(1, 0, 0, 0)))
        print('Individual 3-gram: %f' % corpus_bleu(reference, candidates_sg, weights=(1, 0, 0, 0)))
        # print('Individual 4-gram: %f' % corpus_bleu(reference, candidates_sg, weights=(1, 0, 0, 0)))
        print('Individual 3-gram: %f' % corpus_bleu(reference, candidates_ao, weights=(1, 0, 0, 0)))
        # print('Individual 4-gram: %f' % corpus_bleu(reference, candidates_ao, weights=(1, 0, 0, 0)))
        


        # print('Individual 3-gram: %f' % sentence_bleu(reference, candidates_ao, weights=(0, 0, 1, 0)))
        # print('Individual 4-gram: %f' % sentence_bleu(reference, candidates_ao, weights=(0, 0, 0, 1)))


        # references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
        # candidates = [['this', 'is', 'a', 'test']]
        # score = corpus_bleu(references, candidates)
        # print(score)