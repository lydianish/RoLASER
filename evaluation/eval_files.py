import sys, os, argparse

from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize

assert os.environ.get('FAIRSEQ') and os.environ.get('LASER'), 'Please set the FAIRSEQ and LASER environment variables'

sys.path.append(os.environ['FAIRSEQ'])
sys.path.append(f"{os.environ['LASER']}/source")

from embed import EmbedLoad, embed_sentences

def read_embeddings(input_file, dim=1024, normalized=True, verbose=False, fp16=False):
    X = EmbedLoad(input_file, dim, verbose, fp16)
    if normalized:
        X = normalize(X)
    return X

DASHES = '-' * 40

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ugc-file', help='name of UGC data file', type=str, default='./data/demo_ugc.txt')
    parser.add_argument('--std-file', help='name of standard data file', type=str, default='.data/demo_std.txt')
    parser.add_argument('-m', '--model-dir', help='path to model directory', type=str)
    parser.add_argument('-o', '--output-dir', help='path to directory to save embeddings and results', type=str)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_dir)
    model = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.pt')][0]
    tokenizer = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('-tokenizer.py')][0]
    vocab = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.cvocab')][0]
    
    embed_dir = os.path.join(args.output_dir, 'embeddings', model_name)
    os.makedirs(embed_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'outputs_{model_name}.txt')

    ugc_filename = os.path.basename(args.ugc_file)
    std_filename = os.path.basename(args.std_file)
    ugc_embed_file = os.path.join(embed_dir, f'{ugc_filename}.bin')
    std_embed_file = os.path.join(embed_dir, f'{std_filename}.bin')

    print('Embedding UGC file', args.ugc_file)
    if not os.path.exists(ugc_embed_file):
        embed_sentences(
            ifname=args.ugc_file,
            encoder_path=model,
            custom_tokenizer=tokenizer,
            custom_vocab_file=vocab,
            verbose=True,
            output=ugc_embed_file
        )
    print('Embedding standard file', args.std_file)
    if not os.path.exists(std_embed_file):
        embed_sentences(
            ifname=args.std_file,
            encoder_path=model,
            custom_tokenizer=tokenizer,
            custom_vocab_file=vocab,
            verbose=True,
            output=std_embed_file
        )

    X_std = read_embeddings(std_embed_file, normalized=True)
    X_ugc = read_embeddings(ugc_embed_file, normalized=True)

    X_cos = paired_cosine_distances(X_std, X_ugc)
    X_cos_avg = X_cos.mean()

    print('Computing pairwise cosine distances from', model_name)
    with open(args.ugc_file) as f_ugc, open(args.std_file) as f_std, open(output_file, 'w') as f_out:
        n = 0
        f_out.write(DASHES + '\n')
        f_out.write(f'Pairwise cosine distances from {model_name}\n')
        f_out.write(DASHES + '\n')
        for ugc_line, std_line, cos in zip(f_ugc, f_std, X_cos):
            f_out.write('\n')
            f_out.write(ugc_line)
            f_out.write(std_line)
            f_out.write(str(cos))
            f_out.write('\n')
            n += 1
        f_out.write(f'\nAverage across {n} lines: {X_cos_avg}\n\n')
    
    print("Outputs written in", output_file)





