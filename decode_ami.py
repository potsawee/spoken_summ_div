"""Inference time script for the abstractive task"""
import os
import sys
import torch
import numpy as np
import random
import pickle
from collections import OrderedDict
from datetime import datetime

from models.hierarchical_rnn import EncoderDecoder
from train_ami import get_a_batch
from data_meeting import TopicSegment, Utterance, bert_tokenizer

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else:
    print("source ~/anaconda3/bin/activate torch12-cuda10")
    raise Exception("Torch Version not supported")

START_TOKEN = '[CLS]'
SEP_TOKEN   = '[SEP]'
STOP_TOKEN  = '[MASK]'

START_TOKEN_ID = bert_tokenizer.convert_tokens_to_ids(START_TOKEN)
SEP_TOKEN_ID   = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
STOP_TOKEN_ID  = bert_tokenizer.convert_tokens_to_ids(STOP_TOKEN)

TEST_DATA_SIZE = 20
VOCAB_SIZE     = 30522

def decoding(model, data, args, start_idx, batch_size, num_batches, k,
            search_method, alpha, penalty_ug, memory_utt):
    device = args['device']
    max_summary_length = args['summary_length']
    time_step = max_summary_length
    idx = 0
    summary_out_dir = args['summary_out_dir']

    alpha = alpha
    length_offset = 5

    decode_dict = {
        'k': k, 'search_method': search_method,
        'time_step': time_step, 'vocab_size': VOCAB_SIZE,
        'device': device, 'start_token_id': START_TOKEN_ID,
        'stop_token_id': STOP_TOKEN_ID,
        'alpha': alpha, 'length_offset': length_offset,
        'penalty_ug': penalty_ug,
        'keypadmask_dtype': KEYPADMASK_DTYPE,
        'memory_utt': memory_utt
    }

    if batch_size != 1: raise ValueError("batch size must be 1")

    for bn in range(num_batches):
        decode_dict['batch_size'] = batch_size

        input, u_len, w_len, target, tgt_len, _, _, _  = get_a_batch(
            data, start_idx+idx, batch_size, args['num_utterances'],
            args['num_words'], args['summary_length'], args['summary_type'], device)

        if args['decode_method'] == 'beamsearch':
            if search_method == 'argmax':
                summaries_id, _, _ = model.decode_beamsearch(input, u_len, w_len, decode_dict)
            else:
                raise ValueError("search_method not supported!")
        else:
            raise ValueError("decode_method not supported!")

        # finish t = 0,...,max_summary_length
        summaries = [None for _ in range(batch_size)]
        summaries[0] = tgtids2summary(summaries_id)

        write_summary_files(summary_out_dir, summaries, start_idx+idx)

        print("[{}] batch {}/{} --- idx [{},{})".format(
                str(datetime.now()), bn+1, num_batches,
                start_idx+idx, start_idx+idx+batch_size))

        sys.stdout.flush()
        idx += batch_size

def decoding_teacher_forcing(model, data, args, start_idx, batch_size, num_batches):
    device = args['device']
    max_summary_length = args['summary_length']
    time_step = max_summary_length
    idx = 0
    summary_out_dir = args['summary_out_dir']

    for bn in range(num_batches):
        input, u_len, w_len, target, tgt_len, _, _, _  = get_a_batch(
            data, start_idx+idx, batch_size, args['num_utterances'],
            args['num_words'], args['summary_length'], args['summary_type'], device)

        decoder_output, _, attn_scores, _ = model(input, u_len, w_len, target)
        import pdb; pdb.set_trace()

        summaries_id = [None for _ in range(batch_size)]
        for j in range(batch_size):
            summaries_id[j] = torch.argmax(decoder_output[j], dim=-1).cpu().numpy()
            summaries_id[j] = np.insert(summaries_id[j], 0, 101)
            # print(bert_tokenizer.decode(summaries_id[j]))

        # finish t = 0,...,max_summary_length
        summaries = [None for _ in range(batch_size)]
        for j in range(batch_size):
            summaries[j] = tgtids2summary(summaries_id[j])

        write_summary_files(summary_out_dir, summaries, start_idx+idx)

        print("[{}] batch {}/{} --- idx [{},{})".format(
                str(datetime.now()), bn+1, num_batches,
                start_idx+idx, start_idx+idx+batch_size))

        sys.stdout.flush()
        idx += batch_size

def write_summary_files(dir, summaries, start_idx):
    if not os.path.exists(dir): os.makedirs(dir)
    num_data = len(summaries)
    for idx in range(num_data):
        filepath = dir + 'file.{}.txt'.format(idx+start_idx)
        line = '\n'.join(summaries[idx])
        with open(filepath, 'w') as f:
            f.write(line)

def tgtids2summary(tgt_ids):
    # tgt_ids = a row of numpy array containing token ids
    bert_decoded = bert_tokenizer.decode(tgt_ids)
    # truncate START_TOKEN & part after STOP_TOKEN
    stop_idx = bert_decoded.find(STOP_TOKEN)
    processed_bert_decoded = bert_decoded[5:stop_idx]
    summary = [s.strip() for s in processed_bert_decoded.split(SEP_TOKEN)]
    return summary

def load_ami_data(fversion, data_type):
    path = "lib/model_data/ami-{}.{}.pk.bin".format(fversion, data_type)
    with open(path, 'rb') as f:
        ami_data = pickle.load(f, encoding="bytes")
    return ami_data

def decode(start_idx):
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['num_utterances'] = 1500  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 300   # max no. words in a summary
    args['summary_type']   = 'short'   # max no. words in a summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.0
    args['num_layers_enc'] = 2
    args['num_layers_dec'] = 1

    args['model_save_dir'] = "lib/trained_models/"
    args['model_data_dir'] = "lib/model_data/"

    args['memory_utt']  = False
    args['model_name']  = "MODEL_1"
    args['model_epoch'] = 0

    # ---------------------------------------------------------------------------------- #
    load_option         = 2 # 1=old | 2=new
    # ---------------------------------------------------------------------------------- #
    testfile = 'asrx'   # manual | asr2 (linlin's revised version) | asrx (official release v200124)
    datatype = 'test'    # train/test

    start_idx   = start_idx
    batch_size  = 1
    num_batches = 20

    teacher_forcing = False

    args['decode_method'] = 'beamsearch'
    search_method = 'argmax'
    beam_width  = 10
    alpha       = 2.0
    penalty_ug  = 0.0
    random_seed = 28
    # ---------------------------------------------------------------------------------- #
    if args['decode_method'] == 'beamsearch':
        if 'asr' not in testfile:
            args['summary_out_dir'] = \
            'out_summary/model-{}-ep{}-len{}/{}-width{}-{}-alpha{}-penalty{}/' \
                .format(args['model_name'], args['model_epoch'], args['summary_length'],
                        datatype, beam_width, search_method, alpha, penalty_ug)
        else:
            args['summary_out_dir'] = \
            'out_summary/model-{}-ep{}-len{}/{}/{}-width{}-{}-alpha{}-penalty{}/' \
                .format(args['model_name'], args['model_epoch'], args['summary_length'],
                        testfile, datatype, beam_width, search_method, alpha, penalty_ug)
    # ---------------------------------------------------------------------------------- #

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    args['device'] = device
    print("device = {}".format(device))

    # random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Define and Load the model
    model = EncoderDecoder(args, device)

    trained_model = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'],args['model_epoch'])
    if device == 'cuda':
        try:
            state = torch.load(trained_model)
            if load_option == 1:
                model.load_state_dict(state)
            elif load_option == 2:
                model_state_dict = state['model']
                model.load_state_dict(model_state_dict)
        except:
            model_state_dict = torch.load(trained_model)
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            model.load_state_dict(new_model_state_dict)
    elif device == 'cpu':
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
        # model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))

    model.eval() # switch it to eval mode
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    if testfile == 'manual':
        print("load manual trainscriptions")
        test_data = load_ami_data('191209', datatype)
    elif testfile == 'asr2':
        print("load ASR2 trainscriptions (Linlin's revised version)")
        test_data = load_ami_data('asr-200214', datatype)
    elif testfile == 'asrx':
        print("load ASRX trainscriptions (Official Release)")
        test_data = load_ami_data('asr-200124', datatype)
    elif 'asr_err' in testfile:
        print("load ASR1 trainscriptions with artificial error")
        test_data = load_ami_data('asr-200128.err03', datatype)
    else:
        raise ValueError("test file not exist!")

    print("========================================================")
    print("start decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))
    print("========================================================")

    with torch.no_grad():
        print("beam_width = {}".format(beam_width))
        if not teacher_forcing:
            decoding(model, test_data, args, start_idx, batch_size, num_batches,
                     k=beam_width, search_method=search_method,
                     alpha=alpha, penalty_ug=penalty_ug, memory_utt=args['memory_utt'])
        else:
            if 'asr' not in testfile:
                args['summary_out_dir'] = \
                'out_summary/model-{}-ep{}-len{}/{}-teacherf/' \
                    .format(args['model_name'], args['model_epoch'], args['summary_length'], datatype)
            else:
                args['summary_out_dir'] = \
                'out_summary/model-{}-ep{}-len{}/{}/{}-teacherf/' \
                    .format(args['model_name'], args['model_epoch'], args['summary_length'], testfile, datatype)
            print("Teaching Forcing Decoding!")
            decoding_teacher_forcing(model, test_data, args, start_idx, batch_size, num_batches)
    print("finish decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python decode.py start_idx")
        raise Exception("argv error")

    start_idx = int(sys.argv[1])
    decode(start_idx)
