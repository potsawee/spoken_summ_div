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

from data.meeting import TopicSegment, Utterance, bert_tokenizer, DA_MAPPING
from data import cnndm
from data.cnndm import ProcessedDocument, ProcessedSummary
from models.hierarchical_rnn import EncoderDecoder, DALabeller, EXTLabeller
from models.neural import LabelSmoothingLoss
from train_ami import print_config, load_ami_data, get_a_batch, shift_decoder_target, length2mask

def evaluate_label_task(model_name, epoch, multitask):
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

    args['dropout']        = 0.5
    args['num_layers_enc'] = 2    # in total it's num_layers_enc*3 (word/utt/seg)
    args['num_layers_dec'] = 1
    args['memory_utt']  = False

    args['batch_size']  = 2
    args['random_seed'] = 444

    args['model_save_dir'] = "lib/trained_models/"
    args['load_model'] = "lib/trained_models/model-{}-ep{}.pt".format(model_name, epoch)
    # ---------------------------------------------------------------------------------- #

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

    test_data  = load_ami_data('test')
    valid_data = load_ami_data('valid')

    # random seed
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    model = EncoderDecoder(args, device=device)

    NUM_DA_TYPES = len(DA_MAPPING)
    da_labeller = DALabeller(args['rnn_hidden_size'], NUM_DA_TYPES, device)
    ext_labeller = EXTLabeller(args['rnn_hidden_size'], device)


    if device == 'cuda': state = torch.load(args['load_model'])
    else: state = torch.load(args['load_model'],  map_location=torch.device('cpu'))
    model_state_dict = state['model']
    model.load_state_dict(model_state_dict)
    if multitask:
        da_labeller.load_state_dict(state['da_labeller'])
        ext_labeller.load_state_dict(state['ext_labeller'])

    BATCH_SIZE = args['batch_size']
    num_test_data = len(test_data)

    num_batches = int(num_test_data/BATCH_SIZE)

    idx = 0

    model = model.eval()
    da_labeller = da_labeller.eval()
    ext_labeller = ext_labeller.eval()

    da_true = 0
    da_total = 0

    ext_tp = 0
    ext_tn = 0
    ext_fp = 0
    ext_fn = 0

    for bn in range(num_batches):

        input, u_len, w_len, target, tgt_len, topic_boundary_label, dialogue_acts, extractive_label = get_a_batch(
                test_data, idx, BATCH_SIZE,
                args['num_utterances'], args['num_words'],
                args['summary_length'], args['summary_type'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        decoder_output, u_output, _, _, u_attn_scores = model(input, u_len, w_len, target)

        # multitask(1): dialogue act prediction
        da_output = da_labeller(u_output)

        # multitask(2): extractive label prediction
        ext_output = ext_labeller(u_output).squeeze(-1)

        multitask_mask = length2mask(u_len, BATCH_SIZE, args['num_utterances'], device)

        t, total = multiclass_eval(torch.argmax(da_output,dim=-1), dialogue_acts, multitask_mask)
        da_true += t
        da_total += total

        tp, tn, fp ,fn = labelling_eval(ext_output, extractive_label, multitask_mask, threshold=0.5)
        ext_tp += tp
        ext_tn += tn
        ext_fp += fp
        ext_fn += fn
        idx += BATCH_SIZE
        print("#",end='')
        sys.stdout.flush()

    print()

    print("Model:", args['load_model'])
    print("[2] ======= Dialogue Act Prediction Task =======")
    try:
        accuracy = da_true / da_total
        print("Acc: {:.4f}".format(accuracy))
    except ZeroDivisionError:
        print("zerodivision")
    print("[3] ============== Extractive Task ==============")
    accuracy = (ext_tp+ext_tn)/(ext_tp+ext_tn+ext_fp+ext_fn)
    if ext_tp+ext_fp > 0:
        precision = ext_tp/(ext_tp+ext_fp)
    else:
        precision = 0
    recall = ext_tp/(ext_tp+ext_fn)
    if precision != 0 and recall != 0:
        f1 = 2 * precision*recall / (precision+recall)
    else:
        f1 = 0
    print("Acc: {:.4f} | Pre: {:.4f} | Rec: {:.4f} | F-1: {:.4f}".format(accuracy,precision,recall,f1))
def multiclass_eval(output, target, mask):
    # evaluate accuracy
    match_arr = (output == target).type(torch.FloatTensor)
    match = (match_arr * mask.cpu()).sum().item()
    total = mask.sum().item()
    return match, total


def labelling_eval(output, label, mask, threshold):
    # evaluate P, R, F1-score, accuracy
    output = output.view(-1)
    label = label.view(-1)
    mask = mask.view(-1)
    size = output.size(0)
    pred = torch.zeros((size), dtype=torch.float)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(size):
        if mask[i] == 0.0: continue
        if label[i] == 1.0: # positive
            if output[i] > threshold: tp += 1
            else: fn += 1
        else: # negative
            if output[i] > threshold: fp += 1
            else: tn += 1
    return tp, tn, fp ,fn

def main():
    model_names = ["MODEL_1"]
    multitask   = True
    epoch = 0
    for model_name in model_names:
        evaluate_label_task(model_name, epoch, multitask)

main()
