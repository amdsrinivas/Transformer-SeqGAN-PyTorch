# coding: utf-8

import glob
from sklearn.model_selection import train_test_split
import numpy as np 

import collections
import itertools
import typing

import torch
import transformer

from torch import nn
from torch import optim
import sys

def get_sentences():
    s = []
    word_limit = 10
    for file in list(glob.glob('../data/rev-split/*.new')):                                       
        with open(file) as f:
            word_limit = 10
            lines = f.readlines()
            for line in lines:
                if word_limit == 0:
                    break
                line = line.strip()
                s.append(line)
                word_limit -= 1
        

    s_train, s_test= train_test_split(s, test_size=0.33, random_state=42)
    # np.save("s_train2.npy", s_train)
    # np.save("s_test2.npy", s_test)
    return s_train, s_test

def load_from_big_file():
    file = "../data/train_data_obama.txt"
    s = []
    
    with open(file) as f:
        lines = f.readlines()
    
        for line in lines[:]:
            line = line.strip()
            line = line.rstrip(".")
            words = line.split()
            if len(words) >= 10:
                sent = " ".join(words[:10])
            else:
                sent = " ".join(words)
            sent += " ."
            s.append(sent)
    
    
    s_train, s_test= train_test_split(s, shuffle = True, test_size=0.1, random_state=42)
    return s_train, s_test[:10]


# Uncomment to get train sentences
# s_train, s_test = get_sentences()
# s_train = np.load("s_train.npy")
# s_test = np.load("s_test.npy")

s_train, s_test = load_from_big_file()

# s_train = s_train[:5]
# s_test = s_test[:5]

print("*"*80,"generated sentences")
Token = collections.namedtuple("Token", ["index", "word"])


DATA_GERMAN = s_train
DATA_ENGLISH = s_train
DATA_GERMAN2 = s_test
SOS = Token(0, "<sos>")
EOS = Token(1, "<eos>")
PAD = Token(2, "<pad>")

EMBEDDING_SIZE = 300
GPU = True
NUM_EPOCHS = 500


def eval_model(model: transformer.Transformer, input_seq: torch.LongTensor, target_seq: torch.LongTensor) -> None:
    """Evaluates the the provided model on the given data, and prints the probabilities of the desired translations.
    
    Args:
        model (:class:`transformer.Transformer`): The model to evaluate.
        input_seq (torch.LongTensor): The input sequences, as (batch-size x max-input-seq-len) tensor.
        target_seq (torch.LongTensor): The target sequences, as (batch-size x max-target-seq-len) tensor.
    """
    probs = transformer.eval_probability(model, input_seq, target_seq, pad_index=PAD.index).detach().numpy().tolist()
    
    print("sample       " + ("{}         " * len(probs)).format(*range(len(probs))))
    print("probability  " + ("{:.6f}  " * len(probs)).format(*probs))



def fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
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

'''
def fetch_vocab2() -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    """Determines the vocabulary, and provides mappings from indices to words and vice versa.
    
    Returns:
        tuple: A pair of mappings, index-to-word and word-to-index.
    """
    # gather all (lower-cased) words that appear in the data
    all_words = set()
    for sentence in itertools.chain(DATA_GERMAN2, DATA_GERMAN2):
        all_words.update(word.lower() for word in sentence.split(" "))
    
    # create mapping from index to word
    idx_to_word = [SOS.word, EOS.word, PAD.word] + list(sorted(all_words))
    
    # create mapping from word to index
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    
    return idx_to_word, word_to_idx
'''


'''
def prepare_data2(word_to_idx: typing.Dict[str, int]) -> typing.Tuple[torch.LongTensor, torch.LongTensor]:
    """Prepares the data as PyTorch ``LongTensor``s.
    
    Args:
        word_to_idx (dict[str, int]): A dictionary that maps words to indices in the vocabulary.
    
    Returns:
        tuple: A pair of ``LongTensor``s, the first representing the input and the second the target sequence.
    """
    # break sentences into word tokens
    german = []
    for sentence in DATA_GERMAN2:
        german.append([SOS.word] + sentence.split(" ") + [EOS.word])
    english = []
    for sentence in DATA_GERMAN2:
        english.append([SOS.word] + sentence.split(" ") + [EOS.word])
    
    # pad all sentences to equal length
    len_german = max(len(sentence) for sentence in german)
    for sentence in german:
        sentence.extend([PAD.word] * (len_german - len(sentence)))
    len_english = max(len(sentence) for sentence in english)
    for sentence in english:
        sentence.extend([PAD.word] * (len_english - len(sentence)))
    
    # map words to indices in the vocabulary
    german = [[word_to_idx[word.lower()] for word in sentence] for sentence in german]
    english = [[word_to_idx[word.lower()] for word in sentence] for sentence in english]
    
    # create according LongTensors
    german = torch.LongTensor(german)
    english = torch.LongTensor(english)
    
    return german, english
'''

def prepare_data(DATA_GERMAN, DATA_ENGLISH, word_to_idx: typing.Dict[str, int]) -> typing.Tuple[torch.LongTensor, torch.LongTensor]:
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
    german = [[word_to_idx[word.lower()] for word in sentence] for sentence in german]
    english = [[word_to_idx[word.lower()] for word in sentence] for sentence in english]
    
    # create according LongTensors
    german = torch.cuda.LongTensor(german)
    english = torch.cuda.LongTensor(english)
    
    return german, english

def manual_eval(model):
    idx_to_word, word_to_idx = fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2)
    train_seq, train_seq = prepare_data(DATA_GERMAN, DATA_ENGLISH, word_to_idx)
 
    s_output = transformer.sample_output(model, train_seq[:10], EOS.index, PAD.index, train_seq.size(1))

    print()
    print("Sampled Outputs on Train:")
    print("----------------")
    for sample_idx in range(10):
        for token_idx in range(10):
            print(idx_to_word[train_seq[sample_idx, token_idx].item()], end=" ")
        print(" => ", end=" ")
        for token_idx in range(s_output.size(1)):
            print(idx_to_word[s_output[sample_idx, token_idx].item()], end=" ")
        print()


    test_input_seq, test_target_seq = prepare_data(DATA_GERMAN2, DATA_GERMAN2, word_to_idx)

    s_output = transformer.sample_output(model, test_input_seq, EOS.index, PAD.index, test_target_seq.size(1))

    print()
    print("Sampled Outputs on Test:")
    print("----------------")
    for sample_idx in range(test_input_seq.size(0)):
        for token_idx in range(test_input_seq.size(1)):
            print(idx_to_word[test_input_seq[sample_idx, token_idx].item()], end=" ")
        print(" => ", end=" ")
        for token_idx in range(s_output.size(1)):
            print(idx_to_word[s_output[sample_idx, token_idx].item()], end=" ")
        print()



def main():
    # fetch vocabulary + prepare data
    idx_to_word, word_to_idx = fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2)
    input_seq, target_seq = prepare_data(DATA_GERMAN, DATA_ENGLISH, word_to_idx)

    # create embeddings to use
    emb = nn.Embedding(len(idx_to_word), EMBEDDING_SIZE)
    emb.reset_parameters()
    print("*"*80,"generated emb")
    print(emb)
    # create transformer model
    model = transformer.Transformer(
            emb,
            PAD.index,
            emb.num_embeddings,
            max_seq_len=max(input_seq.size(1), target_seq.size(1))
    )

    # create an optimizer for training the model + a X-entropy loss
    optimizer = optim.Adam((param for param in model.parameters() if param.requires_grad), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    
    print("Initial Probabilities of Translations:")
    print("--------------------------------------")
    # eval_model(model, input_seq, target_seq)
    # print()
    
    # move model + data on the GPU (if possible)
    if GPU:
        model.cuda()
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()

    # train the model
    for epoch in range(NUM_EPOCHS):
        
        for i in range(0, len(input_seq), 10):
            b_input_seq = input_seq[i:i+10]
            b_target_seq = target_seq[i:i+10]
            # print("training epoch {}...".format(epoch + 1), end=" ")
        
            predictions = model(b_input_seq, b_target_seq)
            optimizer.zero_grad()
            current_loss = loss(
                    predictions.view(predictions.size(0) * predictions.size(1), predictions.size(2)),
                    b_target_seq.view(-1)
            )
            current_loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            manual_eval(model)
            torch.save(model.state_dict(), 'experiments/2_obama/saved_model.pkl')
        print("Iteration {}: loss: {:.6f}".format(epoch, current_loss.item()))
        sys.stdout.flush()

        

    
    # put model in evaluation mode
    # model.eval()

    # print()
    # print("Final Probabilities of Translations:")
    # print("------------------------------------")
    # eval_model(model, input_seq, target_seq)
    
    # randomly sample outputs from the input sequences based on the probabilities computed by the trained model
    # idx_to_word, word_to_idx = fetch_vocab2()
    # idx_to_word, word_to_idx = fetch_vocab(DATA_GERMAN2, DATA_GERMAN2)
    # test_input_seq, test_target_seq = prepare_data2(word_to_idx)
    test_input_seq, test_target_seq = prepare_data(DATA_GERMAN2, DATA_GERMAN2, word_to_idx)
    test_input_seq = test_input_seq.cuda()
    test_target_seq = test_target_seq.cuda()
    print(test_input_seq.shape, test_target_seq.shape)
    sampled_output = transformer.sample_output(model, test_input_seq, EOS.index, PAD.index, test_target_seq.size(1))
    print(sampled_output.shape)
    print()
    print("Sampled Outputs:")
    
    print("----------------")
    for sample_idx in range(test_input_seq.size(0)):

        for token_idx in range(test_input_seq.size(1)):
            print(idx_to_word[test_input_seq[sample_idx, token_idx].item()], end=" ")
        print(" => ", end=" ")
        for token_idx in range(sampled_output.size(1)):
            # print("89898",sampled_output[sample_idx, token_idx].item())

            print(idx_to_word[sampled_output[sample_idx, token_idx].item()], end=" ")
        print()


if __name__ == "__main__":
    main()
