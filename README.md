# RoLASER: A Robust LASER Encoder for English User-Generated Content

From the paper Making Sentence Embeddings Robust to User-Generated Content.

## Table of Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Reproduction](#reproduction)
4. [Citation](#citation)


## 1. Introduction <a name="introduction"></a>

<p align="center">
  <img alt="Teacher-Student distillation of LASER" width="500" src="./img/robust-laser.png">
</p>
      
RoLASER is a sentence embedding model trained by distillation of LASER to be robust to user-generated content (UGC). Examples of such content are social media posts, which are known to present a lot of lexical variations (spelling errors, internet slang, abbreviations, ...). RoLASER maps non-standard UGC sentences close to their standard versions in the LASER embedding space, just as the original LASER encoder maps paraphrases and translations close to each other.

### Examples

Example cosine distance between std and ugc sentence for LASER and Rolaser

add plot of rocsmt sentences

### Results

tables?

## 2. Usage <a name="usage"></a>

### Installation

Environment:
- Python >= 3.7
- Pythorch 1.10.1+cu102
- GCC 11.2

Dependencies:

- Fairseq: `git clone https://github.com/lydianish/fairseq.git`


- LASER: `git clone https://github.com/lydianish/LASER.git`

### Examples

#### 1. Sentence Similarity

Computing pairwise cosine distances between sentence embeddings in Python:

```python
import sys, os

from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize

# Set the path to the fairseq and LASER local repos:
sys.path.append('/path/to/fairseq')
os.environ['LASER'] = '/path/to/LASER' # required
sys.path.append(f"{os.environ['LASER']}/source")

from rolaser import RoLaserEncoder

# Set the model and vocab paths, and pick the corresponding tokenizer type 
# (spm for LASER, roberta for RoLASER, and char for c-RoLASER ):
model = "/path/to/model"
vocab = "/path/to/vocab"
tokenizer = "spm | roberta | char"

ugc_sentences = [
    'c u 2moro',
    'I love cheese!',
    'eye wud liek 2 aply 4 vilage idot'
]

std_sentences = [
    'see you tomorrow',
    'I love cheese!',
    'I would like to apply for village idiot.'
]

model = RoLaserEncoder(model_path=model, vocab=vocab, tokenizer=tokenizer)
    
X_std = model.encode(std_sentences)
X_std = normalize(X_std)
X_ugc = model.encode(ugc_sentences)
X_ugc = normalize(X_ugc)

X_cos = paired_cosine_distances(X_std, X_ugc)
X_cos_avg = X_cos.mean()

print('Average paired cosine distance', X_cos_avg)
```


An evaluation script is made available to compute the cosine distances of a file line by line (`evaluation/cos_dist.py`). Use the following command to call it:

```bash
cd RoLASER

python evaluation/cos_dist.py -m $MODEL_DIR \
        -t $TOKENIZER \
        --ugc-file $UGC_FILE \
        --std-file $STD_FILE
```

You can also run the demo script using this command:

```bash
bash ./demo.sh
```

It will output the pairwise cosine distances of the three example sentences used in the paper for LASER, RoLASER and c-RoLASER, and also append them to a single `outputs/outputs.log` file:

```
----------------------------------------
Pairwise cosine distances from LASER
----------------------------------------
c u 2moro
see you tomorrow
0.6577198

I love cheese!
I love cheese!
0.0

eye wud liek 2 aply 4 vilage idot
I would like to apply for village idiot.
0.56884325

Average across 3 sentences: 0.40885434

----------------------------------------
Pairwise cosine distances from RoLASER
----------------------------------------
c u 2moro
see you tomorrow
0.041529734

I love cheese!
I love cheese!
0.0

eye wud liek 2 aply 4 vilage idot
I would like to apply for village idiot.
0.21073495

Average across 3 sentences: 0.08408823

----------------------------------------
Pairwise cosine distances from c-RoLASER
----------------------------------------
c u 2moro
see you tomorrow
0.038794976

I love cheese!
I love cheese!
0.0

eye wud liek 2 aply 4 vilage idot
I would like to apply for village idiot.
0.26292953

Average across 3 sentences: 0.10057483

```


### Pre-trained models

link to download models + tokenizers?

## 3. Reproduction <a name="reproduction"></a>

link to download preprocessed data

Training
- fetch OSCAR 
- augment OSCAR
- tokenize
- fairseq preprocess
- fairseq train

Validation
- fetch flores dev
- augment flores dev
- validate
- select best checkpoint

Evaluation
- fetch flores devtest
- augment flores devtest
- evaluate
    - flores
    - multilexnorm
    - rocsmt
    - mteb
- plot sentences

## 4. Citation <a name="citation"></a>

