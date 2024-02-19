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
- Python 
- Pythorch 1.10.1+cu102
- GCC

Dependencies:

- Fairseq: `git clone https://github.com/lydianish/fairseq.git`


- LASER: `git clone https://github.com/lydianish/LASER.git`

### Examples

The following demo script (`demo.sh`) is made available to test the models on a few sentences:

```bash
#!/bin/bash

set -e

UGC_FILE="data/demo_ugc.txt"
STD_FILE="data/demo_std.txt"
OUTPUT_DIR="outputs/"

MODELS="LASER RoLASER c-RoLASER"

for MODEL in $MODELS
do
    MODEL_DIR=models/${MODEL}

    python evaluation/rolaser.py -m $MODEL_DIR --ugc-file $UGC_FILE --std-file $STD_FILE -o $OUTPUT_DIR

    cat outputs/outputs_${MODEL}.txt | tee -a outputs/outputs.log
done
```

Run the demo using this command:

```bash
cd RoLASER

bash ./demo.sh
```

It will output the pairwise cosine distances of the three example sentences used in the paper for all models (`outputs_LASER.txt`, `outputs_RoLASER.txt`, `outputs_c-RoLASER.txt`) and also append them to a single `outputs.log` file:

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

eye wud liek 2 aply 4 vilage idotI would like to apply for village idiot.
0.56884325

Average across 3 lines: 0.40885433554649353

----------------------------------------
Pairwise cosine distances from RoLASER
----------------------------------------

c u 2moro
see you tomorrow
0.041529708

I love cheese!
I love cheese!
6.3265886e-12

eye wud liek 2 aply 4 vilage idotI would like to apply for village idiot.
0.21073465

Average across 3 lines: 0.08408811688423157

----------------------------------------
Pairwise cosine distances from c-RoLASER
----------------------------------------

c u 2moro
see you tomorrow
0.013441907

I love cheese!
I love cheese!
0.013131566

eye wud liek 2 aply 4 vilage idotI would like to apply for village idiot.
0.26292965

Average across 3 lines: 0.09650104492902756
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

