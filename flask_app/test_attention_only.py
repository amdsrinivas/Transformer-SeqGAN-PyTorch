# coding: utf-8

import glob
from sklearn.model_selection import train_test_split
import numpy as np 

import collections
import itertools
import typing

import copy
import torch
import transformer

from torch import nn
from torch import optim

Token = collections.namedtuple("Token", ["index", "word"])

# DATA_GERMAN = s_train
# DATA_ENGLISH = s_train
# DATA_GERMAN2 = s_test
SOS = Token(0, "<sos>")
EOS = Token(1, "<eos>")
PAD = Token(2, "<pad>")

EMBEDDING_SIZE = 300
GPU = True
NUM_EPOCHS = 1

def fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2):
    """Determines the vocabulary, and provides mappings from indices to words and vice versa.
    
    Returns:
        tuple: A pair of mappings, index-to-word and word-to-index.
    """
    # gather all (lower-cased) words that appear in the data
    all_words = set()
    for sentence in itertools.chain(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2):
        all_words.update(word.lower() for word in sentence.split(" "))
    
    # create mapping from index to word
    idx_to_word = [SOS.word, EOS.word, PAD.word] + list(sorted(all_words))
    # create mapping from word to index
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return idx_to_word, word_to_idx


def prepare_data(DATA_GERMAN, DATA_ENGLISH, word_to_idx: typing.Dict[str, int]):
    """Prepares the data as PyTorch ``LongTensor``s.
    
    Args:
        word_to_idx (dict[str, int]): A dictionary that maps words to indices in the vocabulary.
    
    Returns:
        tuple: A pair of ``LongTensor``s, the first representing the input and the second the target sequence.
    """
    # break sentences into word tokens
    german = []
    
    for sentence in DATA_GERMAN:
        german.append([SOS.word] + sentence.split(" ") + [EOS.word])
    english = []
    for sentence in DATA_ENGLISH:
        english.append([SOS.word] + sentence.split(" ") + [EOS.word])
    
    # pad all sentences to equal length
    len_german = max(len(sentence) for sentence in german)
    for sentence in german:
        sentence.extend([PAD.word] * (len_german - len(sentence)))
    len_english = max(len(sentence) for sentence in english)
    for sentence in english:
        sentence.extend([PAD.word] * (len_english - len(sentence)))
    
    # map words to indices in the vocabulary
    
    g = []
    for sentence in german:
        for word in sentence:
            if word.lower() in word_to_idx:
                g.append(word_to_idx[word.lower()])
            else:
                print(word)
                g.append(PAD.index)
    german = g

    # german = [[word_to_idx[word.lower()] for word in sentence if word.lower() in word_to_idx] for sentence in german]
    # english = [[word_to_idx[word.lower()] for word in sentence if word.lower() in word_to_idx] for sentence in english]
    
    # create according LongTensors
    german = torch.LongTensor(german)
    # english = torch.LongTensor(english)
    
    return german, german


# Helper for interactive demo
def pad_sentences(sentence):
    words = sentence.split(" ")
    if len(words) > 10:
        # keep only 10 words
        words = words[:10]
    else:
        for i in range(10-len(words)):
            words.append(PAD.word)

    return words

# Convert the sentence to word ids
def get_ids(sentence, idx_to_word, word_to_idx):
    sentence_ids = []
    for word in sentence:
        if word.lower() not in word_to_idx:
            # PAD when unknown word found
            sentence_ids.append(PAD.index)
        elif word.lower():
            sentence_ids.append(word_to_idx[word.lower()])
    
    return sentence_ids


def test(input_sent):
    # fetch vocabulary + prepare data
    metadata = torch.load("./checkpoints/attention_only/checkpoint_0/attention_only_metadata.data")
    idx_to_word = metadata['idx_to_word']
    word_to_idx = metadata['word_to_idx']
    emb = metadata['emb']
    input_seq_size = metadata['input_seq_size']
    
    
    # input_sent_padded = pad_sentences(input_sent)
    # input_sent_ids = get_ids(input_sent_padded, idx_to_word, word_to_idx)
    # print("$$input_sent_ids len: ",len(input_sent_ids))

    # idx_to_word, word_to_idx = fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2)
    # input_seq, target_seq = prepare_data(input_sent, input_sent, word_to_idx)
    
    
    # create embeddings to use
    # emb = nn.Embedding(len(idx_to_word), EMBEDDING_SIZE)
    # emb.reset_parameters()
    
    # create transformer model
    model = transformer.Transformer(
            emb,
            PAD.index,
            emb.num_embeddings,
            max_seq_len=input_seq_size
    )
    
    model = torch.load("./checkpoints/attention_only/checkpoint_0/attention_only_model.model")

    if GPU:
        model.cuda()
        # input_seq = torch.LongTensor(input_sent_ids)
        # target_seq = torch.LongTensor(input_sent_ids)
        # input_seq = input_seq.cuda()
        # target_seq = target_seq.cuda()

    # train the model
    # test_input_seq_padded = pad_sentences(input_sent)
    # test_input_seq_ids = get_ids(test_input_seq_padded, idx_to_word, word_to_idx)
    # test_input_seq = torch.LongTensor(test_input_seq_ids)
    # test_target_seq = copy.deepcopy(test_input_seq)

    test_input_seq, test_target_seq = prepare_data([input_sent], [input_sent], word_to_idx)
    # print(test_target_seq.shape)
    # print(test_input_seq)
    test_input_seq = test_input_seq.cuda()
    test_target_seq = test_target_seq.cuda()

    test_input_seq = test_input_seq.view(1, -1)
    test_target_seq = test_target_seq.view(1, -1)
    # print(test_target_seq.shape)
    # print("$$test_target_seq: ",test_target_seq.shape)

    # print(test_input_seq.shape, test_target_seq.shape)
    sampled_output = transformer.sample_output(model, test_input_seq, EOS.index, PAD.index, test_target_seq.size(1))
    
    # print("$$$ sample_output", sampled_output)

    output_arr = []
    for sample_idx in range(test_input_seq.size(0)):

        for token_idx in range(sampled_output.size(1)):
            temp = idx_to_word[sampled_output[sample_idx, token_idx].item()]
            output_arr.append(temp)

    return output_arr