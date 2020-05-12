import pickle
import pdb

class ProcessedDocument(object):
    def __init__(self, encoded_article, attention_mask, token_type_ids, cls_pos):
        self.encoded_article = encoded_article
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_pos = cls_pos

class ProcessedSummary(object):
    def __init__(self, encoded_abstract, length):
        self.encoded_abstract = encoded_abstract
        self.length = length


# copied from summariser0/train.py
def load_data(args, data_type):
    if data_type not in ['train', 'val', 'test', 'trainx']:
        raise ValueError('train/val/test only')

    path_data        = args['model_data_dir'] + "{}-{}.dat.nltk.pk.bin".format(data_type, args['max_pos_embed'])
    path_target      = args['model_data_dir'] + "target.{}-{}.pk.bin".format(data_type, args['max_num_sentences'])
    with open(path_data, "rb") as f: data = pickle.load(f)
    with open(path_target, "rb") as f: target_pos = pickle.load(f)

    assert len(data) == len(target_pos), "len(data) != len(target_pos)"

    num_data = len(data)
    encoded_articles   = [None for i in range(num_data)]
    attention_masks    = [None for i in range(num_data)]
    token_type_ids_arr = [None for i in range(num_data)]
    cls_pos_arr        = [None for i in range(num_data)]

    for i, doc in enumerate(data):
        encoded_articles[i]   = doc.encoded_article
        attention_masks[i]    = doc.attention_mask
        token_type_ids_arr[i] = doc.token_type_ids
        cls_pos_arr[i]        = doc.cls_pos

    data_dict = {
        'num_data': num_data,
        'encoded_articles': encoded_articles,
        'attention_masks': attention_masks,
        'token_type_ids_arr': token_type_ids_arr,
        'cls_pos_arr': cls_pos_arr,
        'target_pos': target_pos
    }
    return data_dict

# copied from summariser0/train_abstractive.py
def load_summary(args, data_type):
    if data_type not in ['train', 'val', 'test', 'trainx']:
        raise ValueError('train/val/test only')

    path = args['model_data_dir'] + "abstract.{}-{}.pk.bin".format(data_type, args['max_summary_length'])
    with open(path, "rb") as f: summaries = pickle.load(f)

    N = len(summaries)
    encoded_abstracts = [None for i in range(N)]
    abs_lengths       = [None for i in range(N)]

    for i, sum in enumerate(summaries):
        encoded_abstracts[i] = sum.encoded_abstract
        abs_lengths[i]       = sum.length

    summary_dict = {
        'num_data': N,
        'encoded_abstracts': encoded_abstracts,
        'abs_lengths': abs_lengths
    }
    return summary_dict
