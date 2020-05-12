import os
import pickle
import pdb

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# -------------------------------------------------------------------------------------------- #
# get this official partition from http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
# split = 98+20+20 e.g. 25*4-2 , 5*4, 5*4
SCENARIO_TRAIN = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009',
'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002',
'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009',
'TS3010', 'TS3011', 'TS3012'] ### --- IS1002 (no a), IS1005 (no d)
                              ### --- ES2006c does not have topic segmentation
                              ### --- ES2015(b,c) does not have topic segmentation
                              ### --- TS3012c does have have extsumm
SCENARIO_VALID = ['ES2003', 'ES2011', 'IS1008', 'TS3004', 'TS3006']
SCENARIO_TEST  = ['ES2004', 'ES2014', 'IS1009', 'TS3003', 'TS3007']
# -------------------------------------------------------------------------------------------- #
CLS_TOKEN  = '[CLS]'
SEP_TOKEN  = '[SEP]'
MASK_TOKEN = '[MASK]'

DA_MAPPING = {
    'bck': 0, 'stl': 1, 'fra': 2,
    'inf': 3, 'sug': 4, 'ass': 5,
    'el.inf': 6, 'el.sug': 7, 'el.ass': 8, 'el.und': 9,
    'off': 10, 'und': 11, 'be.pos': 12, 'be.neg': 13, 'oth': 14,
    'na': -1
}

class TopicSegment(object):
    def __init__(self):
        self.utterances = []
        self.topic_labels = None
        self.topic_description = None

    def add_utterance(self, utterance):
        self.utterances.append(utterance)

class Utterance(object):
    def __init__(self, encoded_words, dialogueact, speakerid, extsum_label):
        self.encoded_words = encoded_words
        self.dialogueact   = dialogueact
        self.speakerid     = speakerid
        self.extsum_label  = extsum_label

def process_input(path):
    with open(path) as f:
        lines = f.readlines()

    topic_segment = None
    topic_segments = []

    last_line_idx = len(lines) - 1

    for ln, line in enumerate(lines):
        items = line.strip().split('\t')
        # topic segmentation line
        if items[0] == 'topic_segment':
            if topic_segment != None:
                topic_segments.append(topic_segment)

            topic_labels = [int(x) for x in items[1].split(',')]
            topic_description = items[2]

            topic_segment = TopicSegment()
            topic_segment.topic_labels = topic_labels
            topic_segment.topic_description = topic_description

        # utterance line
        else:
            utterance_id = items[0]
            starttime    = float(items[1])
            endtime      = float(items[2])
            speakerid    = items[3]
            dialogueact  = items[4]
            w_startid    = int(items[5])
            w_endid      = int(items[6])
            extsum_label = int(items[7])
            text         = items[-1]

            encoded_words = bert_tokenizer.encode(text)
            utterance = Utterance(encoded_words, dialogueact, speakerid, extsum_label)
            topic_segment.add_utterance(utterance)

            if ln == last_line_idx:
                topic_segments.append(topic_segment)

    return topic_segments

def process_summary(path):
    with open(path) as f:
        lines = f.readlines()
    encoded_summary = []
    for line in lines:
        line = line.strip()
        encoded_summary += bert_tokenizer.encode(line)

        encoded_summary += bert_tokenizer.encode(SEP_TOKEN)

    encoded_summary = bert_tokenizer.encode(CLS_TOKEN) + encoded_summary
    encoded_summary += bert_tokenizer.encode(MASK_TOKEN)
    return encoded_summary

def get_ami_data(dir, data_type, style):
    if data_type not in ['train', 'valid', 'test']: raise ValueError("train/valid/test only")

    if data_type == 'train':   scenarios = SCENARIO_TRAIN
    elif data_type == 'valid': scenarios = SCENARIO_VALID
    elif data_type == 'test':  scenarios = SCENARIO_TEST

    ami_data = [] # [(topic_segments, encoded_summary), (),...]
    for scenario in scenarios:
        for part in ['a','b','c','d']:
            if scenario == "IS1002" and part == "a": continue
            if scenario == "IS1005" and part == "d": continue
            if scenario == "ES2006" and part == "c": continue # no topic segmentation
            if scenario == "ES2015" and part == "b": continue # no topic segmentation
            if scenario == "ES2015" and part == "c": continue # no topic segmentation
            if scenario == "TS3012" and part == "c": continue # no extractive summary

            inputtsvpath = "{}/{}{}.{}.tsv".format(dir,scenario,part,style)
            longsummpath = "{}/{}{}.abslong.txt".format(dir,scenario,part)
            shortsummpath = "{}/{}{}.absshort.txt".format(dir,scenario,part)
            topic_segments  = process_input(inputtsvpath)
            encoded_summary_long  = process_summary(longsummpath)
            encoded_summary_short = process_summary(shortsummpath)
            ami_data.append((topic_segments, encoded_summary_long, encoded_summary_short))
            print("loaded: {}{}".format(scenario,part))

    return ami_data

def rouge_reference(dir_in, dir_out, style, sumtype):
    idx = 0
    for scenario in SCENARIO_VALID:
        for part in ['a','b','c','d']:
            if scenario == "IS1002" and part == "a": continue
            if scenario == "IS1005" and part == "d": continue
            if scenario == "ES2006" and part == "c": continue # no topic segmentation
            if scenario == "ES2015" and part == "b": continue # no topic segmentation
            if scenario == "ES2015" and part == "c": continue # no topic segmentation
            if scenario == "TS3012" and part == "c": continue # no extractive summary

            if sumtype == 'long':
                summpath = "{}/{}{}.abslong.txt".format(dir_in,scenario,part)
            elif sumtype == 'short':
                summpath = "{}/{}{}.absshort.txt".format(dir_in,scenario,part)

            with open(summpath, 'r') as f:
                lines = f.readlines()

            path_out = "{}/file.{}.txt".format(dir_out, idx)
            with open(path_out, 'w') as f:
                f.write("".join(lines))
            idx += 1

def rouge_reference_go():
    dir_in = "/home/alta/summary/pm574/data/amicorpus/summary_work/v-191203"
    dir_out = "/home/alta/summary/pm574/summariser1/out_summary/reference/valid/long"
    style = 'manual'
    sumtype = 'long'
    rouge_reference(dir_in, dir_out, style, sumtype)

def main():
    ami_dir = "/home/alta/summary/pm574/data/amicorpus/summary_work/v-200214-asr"
    train_data = get_ami_data(ami_dir, data_type='train', style='asr') # len = 94
    valid_data = get_ami_data(ami_dir, data_type='valid', style='asr') # len = 20
    test_data  = get_ami_data(ami_dir, data_type='test',  style='asr') # len = 20

    with open("lib/model_data/ami-asr-200214.train.pk.bin", "wb") as f: pickle.dump(train_data, f)
    with open("lib/model_data/ami-asr-200214.valid.pk.bin", "wb") as f: pickle.dump(valid_data, f)
    with open("lib/model_data/ami-asr-200214.test.pk.bin", "wb") as f: pickle.dump(test_data, f)

    print("process data finished")

if __name__ == "__main__":
    main()
    # rouge_reference_go()
