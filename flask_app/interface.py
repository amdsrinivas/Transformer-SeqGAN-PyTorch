import sys
sys.path.insert(0, '../seq_gan_with_attention')
sys.path.insert(0, '../seq_gan')
sys.path.insert(0, '../core') 

from helper import *
from generator_attention import Generator_attention as GenAttention
from generator import Generator as GenSeqgan
from data_iter import GenDataIter, DisDataIter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class interactive_demo:
    def __init__(self, test_sentence=None):
        self.CHECKPOINT_PATH = './checkpoints/'
        self.TEST_FILE       = './raw_test.data'
        BATCH_SIZE           = 10
        self.cuda            = True
        # self.idx_to_word, self.word_to_idx, self.VOCAB_SIZE, g_emb_dim, g_hidden_dim, g_sequence_len = load_vocab(self.CHECKPOINT_PATH)
        self.gen_attention_metadata = load_vocab(self.CHECKPOINT_PATH+'metadata_attention.data')

        # Seq GAN with Attention
        self.gen_attention = GenAttention(self.gen_attention_metadata['vocab_size'], self.gen_attention_metadata['g_emb_dim'],
                    self.gen_attention_metadata['g_hidden_dim'], self.gen_attention_metadata['g_sequence_len'], BATCH_SIZE, self.cuda, test_mode = True)
        self.gen_attention = self.gen_attention.cuda()
        self.gen_attention.load_state_dict(torch.load(self.CHECKPOINT_PATH+'generator_attention.model', map_location={'cuda:1': 'cpu'}))

        # Seq GAN based generator
        # generator = GenSeqgan(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
        # generator = generator.cuda()
        # generator.load_state_dict(torch.load(CHECKPOINT_PATH+'generator_seqgan.model', map_location={'cuda:1': 'cpu'}))
    
    def predict_for_all(self, test_sentence = None):
        attention_out = self.demo(self.gen_attention, self.gen_attention_metadata)
        return attention_out
 
    def demo(self, model, metadata):
        # idx_to_word, word_to_idx, VOCAB_SIZE = load_vocab(CHECKPOINT_PATH)
        self.test_sentence = "This is a sample sentence"
        self.max_test_sentence_len = 10

        # Get padded sentencels
        self.padded_sentence = pad_sentences(self.test_sentence, self.max_test_sentence_len)
        # print(self.padded_sentence)

        # Get ids of padded sentence
        padded_sent_ids = get_ids(self.padded_sentence, metadata['idx_to_word'], metadata['word_to_idx'], metadata['vocab_size'])
        # print(padded_sent_ids)

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
            mini_batch = prob.shape[0]
            prob = torch.reshape(prob, (prob.shape[0] * prob.shape[1], -1))
            predictions = torch.max(prob, dim=1)[1]
            predictions = predictions.view(mini_batch, -1)
            # print('PRED SHAPE:' , predictions.shape)
            for each_sen in list(predictions):
                sent_from_id = generate_sentence_from_id(metadata['idx_to_word'], each_sen)
                print('Sample Output:', sent_from_id)
                ret_pred.append(sent_from_id)
            sys.stdout.flush()
             
        data_iter.reset()
        return ret_pred
