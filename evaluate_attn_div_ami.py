import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random

from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

from data_meeting import TopicSegment, Utterance, bert_tokenizer
from data import cnndm
from data.cnndm import ProcessedDocument, ProcessedSummary
from models.hierarchical_rnn import EncoderDecoder

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool

START_TOKEN = '[CLS]'
SEP_TOKEN   = '[SEP]'
STOP_TOKEN  = '[MASK]'

START_TOKEN_ID = bert_tokenizer.convert_tokens_to_ids(START_TOKEN)
SEP_TOKEN_ID   = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
STOP_TOKEN_ID  = bert_tokenizer.convert_tokens_to_ids(STOP_TOKEN)

teacherforcing  = True
decoding_method = 'teacherforcing'
attn_score_path = "lib/attn_scores_u/HGRUV5_AMI_APR/{}/".format(decoding_method)

def evaluate_attn_div(start_idx, end_idx):
    print("Start training hierarchical RNN model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['num_utterances'] = 1500  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 300   # max no. words in a summary
    args['summary_type']   = 'short'   # long or short summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.1
    args['num_layers_enc'] = 2    # in total it's num_layers_enc*2 (word/utt)
    args['num_layers_dec'] = 1

    args['batch_size']      = 1

    args['memory_utt'] = False


    args['model_save_dir'] = "lib/trained_models/"
    args['load_model']  = "lib/trained_models/MODEL_0.pt"
    load_option         = 2 # 1=old | 2=new
    # ---------------------------------------------------------------------------------- #
    # print_config(args)

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack... 1 GPU')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '0' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    train_data = load_ami_data('test')

    model = EncoderDecoder(args, device=device)
    # print(model)

    # Load model if specified (path to pytorch .pt)
    if args['load_model'] != None:
        trained_model = args['load_model']
        if device == 'cuda':
            try:
                state = torch.load(trained_model)
                if load_option == 1:
                    model.load_state_dict(state)
                elif load_option == 2:
                    model_state_dict = state['model']
                    model.load_state_dict(model_state_dict)
            except RuntimeError: # need to remove module
                # Main model
                model_state_dict = torch.load(trained_model)
                new_model_state_dict = OrderedDict()
                for key in model_state_dict.keys():
                    new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
                model.load_state_dict(new_model_state_dict)
        else:
            try:
                state = torch.load(trained_model, map_location=torch.device('cpu'))
                if load_option == 1:
                    model.load_state_dict(state)
                elif load_option == 2:
                    model_state_dict = state['model']
                    model.load_state_dict(model_state_dict)
            except:
                model_state_dict = torch.load(trained_model, map_location=torch.device('cpu'))
                new_model_state_dict = OrderedDict()
                for key in model_state_dict.keys():
                    new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
                model.load_state_dict(new_model_state_dict)
        model.eval()
        print("Loaded model from {}".format(args['load_model']))
    else:
        print("Train a new model")

    print("Train a new model")

    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    if BATCH_SIZE != 1: raise ValueError("Batch Size must be 1")

    num_train_data = len(train_data)
    num_batches = int(num_train_data/BATCH_SIZE)
    print("num_batches = {}".format(num_batches))

    idx = 0

    decode_dict = {
        'batch_size': BATCH_SIZE,
        'k': 10, 'search_method': 'argmax',
        'time_step': args['summary_length'], 'vocab_size': 30522,
        'device': device, 'start_token_id': START_TOKEN_ID,
        'stop_token_id': STOP_TOKEN_ID,
        'alpha': 2.0, 'length_offset': 5,
        'penalty_ug': 0.0,
        'keypadmask_dtype': KEYPADMASK_DTYPE,
        'memory_utt': args['memory_utt']
    }

    print("DECODING: {}".format(decoding_method))

    with torch.no_grad():
        for bn in range(start_idx, end_idx):

            if check_if_id_exists(bn): continue

            input, u_len, w_len, target, tgt_len = get_a_batch(
                    train_data, bn, BATCH_SIZE,
                    args['num_utterances'], args['num_words'],
                    args['summary_length'], args['summary_type'], device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            if teacherforcing == True:
                try:
                    # decoder_output = model(input, u_len, w_len, target)
                    decoder_output, _, attn_scores, _, u_attn_scores = model(input, u_len, w_len, target)
                except IndexError:
                    print("there is an IndexError --- likely from if segment_indices[bn][-1] == u_len[bn]-1:")
                    print("for now just skip this batch!")
                    idx += BATCH_SIZE # previously I forget to add this line!!!
                    continue

                output = torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist()[0]
                max_l  = decoder_output.size(1)

            else:
                output, attn_score, attn_score_u = model.decode_beamsearch(input, u_len, w_len, decode_dict)

                # shift decoder_output by one
                output = output.tolist()
                max_l = len(output)
                u_attn_scores = attn_score_u.unsqueeze(0)

            try:
                dec_len = output.index(103)
            except ValueError:
                dec_len = max_l
            dec_sep_pos = []
            for i, v in enumerate(output):
                if i == dec_len: break
                if v == 102: dec_sep_pos.append(i)

            if len(dec_sep_pos) == 0:
                dec_sep_pos.append(max_l)

            enc_len = u_len[0]
            dec_start_pos = [0] + [x+1 for x in dec_sep_pos[:-1]]
            this_attn = u_attn_scores[0, :dec_len, :enc_len].cpu()

            mean_div_within_sentence, mean_div_between_sentences = diversity1_sent(this_attn, dec_start_pos, dec_sep_pos)
            write_attn_scores(bn, mean_div_within_sentence, mean_div_between_sentences)

def check_if_id_exists(id):
    exist = os.path.isfile(attn_score_path + '{}_x1.txt'.format(id))
    # if exist: print("id already exists")
    return exist

def write_attn_scores(id, x1, x2):
    path_x1 = attn_score_path + '{}_x1.txt'.format(id)
    path_x2 = attn_score_path + '{}_x2.txt'.format(id)

    with open(path_x1, 'w') as f: f.write(str(x1))
    with open(path_x2, 'w') as f: f.write(str(x2))

    print("wrote: id = {}".format(id))

    return

def diversity1_sent(attn_score, dec_start_pos, dec_sep_pos):
    # BATCH_SIZE is 1
    num_dec_sentences = len(dec_sep_pos)
    diverisity = []
    avg_attn = [None for _ in range(num_dec_sentences)]
    for j in range(num_dec_sentences):
        _t1 = dec_start_pos[j]
        _t2 = dec_sep_pos[j] + 1
        attn_in_this_dec_sent = attn_score[_t1:_t2]

        T, N = attn_in_this_dec_sent.size()
        # ----------- Within the same decoder sentence ------------ #
        count = 0
        sum_div = 0
        for t1 in range(T-1):
            for t2 in range(t1+1, T):
                p1 = attn_in_this_dec_sent[t1,:N]
                p2 = attn_in_this_dec_sent[t2,:N]
                sum_div += torch.sqrt(((p1 - p2)**2).mean()).item()
                count += 1
        if count == 0 and T == 1:
            d = 0
        else:
            d = sum_div / count
        diverisity.append(d)


        # ----------- between the decoder sentences ------------ #
        avg_attn[j] = attn_score[_t1:_t2].mean(dim=0)

    count = 0
    sum_div = 0
    for j1 in range(num_dec_sentences-1):
        for j2 in range(j1+1, num_dec_sentences):
            p1 = avg_attn[j1]
            p2 = avg_attn[j2]
            sum_div += torch.sqrt(((p1 - p2)**2).mean()).item()
            count += 1
    mean_div_within_sentence = sum(diverisity) / len(diverisity)
    if count > 0:
        mean_div_between_sentences = sum_div / count
    else:
        mean_div_between_sentences = 0

    return mean_div_within_sentence, mean_div_between_sentences

def shift_decoder_target(target, tgt_len, device, mask_offset=False):
    # MASK_TOKEN_ID = 103
    batch_size = target.size(0)
    max_len = target.size(1)
    dtype0  = target.dtype

    decoder_target = torch.zeros((batch_size, max_len), dtype=dtype0, device=device)
    decoder_target[:,:-1] = target.clone().detach()[:,1:]
    # decoder_target[:,-1:] = 103 # MASK_TOKEN_ID = 103
    # decoder_target[:,-1:] = 0 # add padding id instead of MASK

    # mask for shifted decoder target
    decoder_mask = torch.zeros((batch_size, max_len), dtype=torch.float, device=device)
    if mask_offset:
        offset = 10
        for bn, l in enumerate(tgt_len):
            # decoder_mask[bn,:l-1].fill_(1.0)
            # to accommodate like 10 more [MASK] [MASK] [MASK] [MASK],...
            if l-1+offset < max_len: decoder_mask[bn,:l-1+offset].fill_(1.0)
            else: decoder_mask[bn,:].fill_(1.0)
    else:
        for bn, l in enumerate(tgt_len):
            decoder_mask[bn,:l-1].fill_(1.0)

    return decoder_target, decoder_mask

def get_a_batch(ami_data, idx, batch_size, num_utterances, num_words, summary_length, sum_type, device):
    if sum_type not in ['long', 'short']:
        raise Exception("summary type long/short only")

    input   = torch.zeros((batch_size, num_utterances, num_words), dtype=torch.long)
    summary = torch.zeros((batch_size, summary_length), dtype=torch.long)
    summary.fill_(103)

    utt_lengths  = np.zeros((batch_size), dtype=np.int)
    word_lengths = np.zeros((batch_size, num_utterances), dtype=np.int)

    # summary lengths
    summary_lengths = np.zeros((batch_size), dtype=np.int)

    for bn in range(batch_size):
        topic_segments  = ami_data[idx+bn][0]
        if sum_type == 'long':
            encoded_summary = ami_data[idx+bn][1]
        elif sum_type == 'short':
            encoded_summary = ami_data[idx+bn][2]
        # input
        utt_id = 0
        for segment in topic_segments:
            utterances = segment.utterances
            for utterance in utterances:
                encoded_words = utterance.encoded_words
                l = len(encoded_words)
                if l > num_words:
                    encoded_words = encoded_words[:num_words]
                    l = num_words
                input[bn,utt_id,:l] = torch.tensor(encoded_words)
                # word_lengths[bn,utt_id] = torch.tensor(l)
                word_lengths[bn,utt_id] = l
                utt_id += 1
                if utt_id == num_utterances: break

            if utt_id == num_utterances: break

        # utt_lengths[bn] = torch.tensor(utt_id)
        utt_lengths[bn] = utt_id

        # summary
        l = len(encoded_summary)
        if l > summary_length:
            encoded_summary = encoded_summary[:summary_length]
            l = summary_length
        summary_lengths[bn] = l
        summary[bn, :l] = torch.tensor(encoded_summary)

    input   = input.to(device)
    summary = summary.to(device)

    # covert numpy to torch tensor (for multiple GPUs purpose)
    utt_lengths = torch.from_numpy(utt_lengths)
    word_lengths = torch.from_numpy(word_lengths)
    summary_lengths = torch.from_numpy(summary_lengths)

    return input, utt_lengths, word_lengths, summary, summary_lengths

def load_ami_data(data_type):
    path = "lib/model_data/ami-191209.{}.pk.bin".format(data_type)
    with open(path, 'rb') as f:
        ami_data = pickle.load(f, encoding="bytes")
    return ami_data

def load_cnndm_data(args, data_type, dump=False):
    if dump:
        data    = cnndm.load_data(args, data_type)
        summary = cnndm.load_summary(args, data_type)
        articles = []
        for encoded_words in data['encoded_articles']:
            # encoded_sentences = []
            article = TopicSegment()
            l = len(encoded_words) - 1
            for i, x in enumerate(encoded_words):
                if x == 101: # CLS
                    sentence = []
                elif x == 102: # SEP
                    utt = Utterance(sentence, -1, -1, -1)
                    article.add_utterance(utt)
                elif x == 100: # UNK
                    break
                else:
                    sentence.append(x)
                    if i == l:
                        utt = Utterance(sentence, -1, -1, -1)
                        article.add_utterance(utt)
            articles.append([article])
        abstracts = []
        for encoded_abstract in summary['encoded_abstracts']:
            if 103 in encoded_abstract:
                last_idx = encoded_abstract.index(103)
                encoded_abstract = encoded_abstract[:last_idx]
            encoded_abstract.append(102)
            encoded_abstract.append(103)
            abstracts.append(encoded_abstract)
        cnndm_data = []
        for x, y in zip(articles, abstracts):
            cnndm_data.append((x,y,y))
    else:
        path = "lib/model_data/cnndm-191216.{}.pk.bin".format(data_type)
        with open(path, 'rb') as f:
            cnndm_data = pickle.load(f, encoding="bytes")

    return cnndm_data

def print_config(args):
    print("============================= CONFIGURATION =============================")
    for x in args:
        print('{}={}'.format(x, args[x]))
    print("=========================================================================")

if __name__ == "__main__":
    if(len(sys.argv) == 3):
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])

    elif(len(sys.argv) == 2):
        start_idx = int(sys.argv[1])
        end_idx = start_idx + 1
    else:
        raise Exception("argv error")

    if end_idx > 20: end_idx = 20

    evaluate_attn_div(start_idx, end_idx)
