# spoken_summ_div
------------------------------------------------
## Requirements (tested)
- python 3.7
- torch 1.2
- CUDA 10
## 1. Data Preparation
- download (Manual/ASR transcripts): http://groups.inf.ed.ac.uk/ami/download/
- process the downloaded files into this format:
```
utt0000.A.1     0.37    1.76    A       bck     0       3       0       hmm hmm hmm .
utt0001.B.2     10.99   12.13   B       fra     1       2       0       are we
utt0002.B.3     12.13   14.53   B       el.inf  4       19      0       we're not allowed to dim the lights so people can see that a bit better ?
```
- use data/meeting.py to convert the processed files into pickle binaries
## 2. Model Training
```
python train_ami.py
```
- set the hyperparameters in args['...']
- choose GPU(s) by setting os.environ["CUDA_VISIBLE_DEVICES"] = '0'
## 3. Model Decoding
```
python decode_ami.py start_idx
```
- set the decoding parameters in args['...']
## 4. Model Evaluation (ROUGE summarisation)
PyROUGE: https://pypi.org/project/pyrouge/
## 5. Model Evaluation (multitask learning tasks)
```
python evaluate_attn_div_ami.py start_idx end_idx
```
