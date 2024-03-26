# RoLASER: A Robust LASER Encoder for English User-Generated Content

From the paper [Making Sentence Embeddings Robust to User-Generated Content (Nishimwe et al., 2024)](https://hal.science/hal-04520909).

## Table of Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Reproducing results](#reproducing) 
4. [Citation](#citation)


## 1. Introduction <a name="introduction"></a>

<p align="center">
  <img alt="Teacher-Student approach" width="500" src="./img/robust-laser.png">
</p>
      
RoLASER is a sentence embedding model trained using a teacher-student approach (with LASER as the teacher) to be robust to English user-generated content (UGC). Examples of such content are social media posts, which are known to present a lot of lexical variations (spelling errors, internet slang, abbreviations, ...). RoLASER maps non-standard UGC sentences close to their standard versions in the LASER embedding space, just as the original LASER encoder maps paraphrases and translations close to each other.

## 2. Usage <a name="usage"></a>

### Installation

Environment:
- (Mini)conda
- Python >= 3.9 (tested with `3.9.18`)
- CUDA 10.2
- GCC 11.2
- Pythorch 1.10.1

Dependencies:
- Fairseq fork: https://github.com/lydianish/fairseq
- LASER fork: https://github.com/lydianish/LASER

Installation commands:
```bash
# Install RoLASER 
git clone https://github.com/lydianish/RoLASER.git
cd RoLASER 

# Create conda environment with the required Python and GCC versions
conda create -n "rolaser_env" python=3.9.18 gxx=11.2.0=h702ea55_10 -c conda-forge

# Activate the conda environment
conda activate rolaser_env

# Install dependencies
bash ./install.sh

# Download models
bash ./download_models.sh
```

### Examples

#### 1. Sentence Similarity between non-standard UGC sentences and their standard equivalents

##### a. Demo script

You can run the demo script using this command:

```bash
bash ./demo.sh
```

It will output the pairwise cosine distances of the three example sentences used in the paper for LASER, RoLASER and c-RoLASER. It will write index-oriented JSON files for each model in the output directory. 

For example, `outputs_RoLASER.json`:
```json
{
    "0": {
        "ugc": "c u 2moro",
        "std": "see you tomorrow",
        "cos": 0.0415297337
    },
    "1": {
        "ugc": "I love cheese!",
        "std": "I love cheese!",
        "cos": 0.0
    },
    "2": {
        "ugc": "eye wud liek 2 aply 4 vilage idot",
        "std": "I would like to apply for village idiot.",
        "cos": 0.2107349485
    }
}
```

It will also append the outputs in a single `outputs/outputs.log` file:

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

##### b. Computing pairwise cosine distances between sentence embeddings with Python:

An evaluation script is made available to compute the cosine distances of a file line by line (`evaluation/cos_dist.py`). Use the following command to call it:

```bash
python ./evaluation/cos_dist.py -m $MODEL_DIR \
    -t $TOKENIZER \
    --ugc-file $UGC_FILE \
    --std-file $STD_FILE \
    -o $OUTPUT_DIR \
    --verbose
```

You can also modify your own Python script:

```python
import sys, os

from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize

# Set the path to the fairseq and LASER local repos:
# Normally, the environment variables are defined during installation
# If not, you can set them here:
# os.environ['FAIRSEQ'] = '/path/to/fairseq'
# os.environ['LASER'] = '/path/to/LASER'

sys.path.append(os.environ['FAIRSEQ'])
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

### Pre-trained models

The models, tokenizers and vocabulary files for LASER, RoLASER and c-RoLASER are available [here](https://zenodo.org/records/10864557).

## 3. Reproducing results <a name="reproducing"></a>

Coming soon

<!--
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
-->

## 4. Citation <a name="citation"></a>

```bibtex
@unpublished{nishimwe:hal-04520909,
  TITLE = {{Making Sentence Embeddings Robust to User-Generated Content}},
  AUTHOR = {Nishimwe, Lydia and Sagot, Beno{\^i}t and Bawden, Rachel},
  URL = {https://hal.science/hal-04520909},
  NOTE = {Accepted at LREC-COLING 2024},
  YEAR = {2024},
  MONTH = Mar,
  KEYWORDS = {sentence embeddings ; robustness ; user-generated content (UGC)},
  PDF = {https://hal.science/hal-04520909/file/Lydia_Nishimwe_LREC_COLING_2024.pdf},
  HAL_ID = {hal-04520909},
  HAL_VERSION = {v1},
}
```