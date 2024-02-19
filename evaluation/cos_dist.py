import sys, os, argparse

from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize

sys.path.append('/home/lnishimw/scratch/fairseq')
os.environ['LASER'] = '/home/lnishimw/scratch/LASER' # required
sys.path.append(f"{os.environ['LASER']}/source")

from rolaser import RoLaserEncoder

DASHES = '-' * 40

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ugc-file', help='name of UGC data file', type=str, default='./data/demo_ugc.txt')
    parser.add_argument('--std-file', help='name of standard data file', type=str, default='./data/demo_std.txt')
    parser.add_argument('-m', '--model-dir', help='path to model directory', type=str, required=True)
    parser.add_argument('-t', '--tokenizer', help='tokenizer type', type=str, choices=['spm', 'roberta', 'char'], required=True)
    args = parser.parse_args()

    ugc_sentences = [ line.strip() for line in open(args.ugc_file).readlines() ]
    std_sentences = [ line.strip() for line in open(args.std_file).readlines() ]
    
    model_name = os.path.basename(args.model_dir)
    model = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.pt')][0]
    vocab = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.cvocab')][0]

    model = RoLaserEncoder(model_path=model, vocab=vocab, tokenizer=args.tokenizer)
    
    X_std = model.encode(std_sentences)
    X_std = normalize(X_std)
    X_ugc = model.encode(ugc_sentences)
    X_ugc = normalize(X_ugc)

    X_cos = paired_cosine_distances(X_std, X_ugc)
    X_cos_avg = X_cos.mean()

    print(DASHES)
    print('Pairwise cosine distances from', model_name)
    print(DASHES)

    for ugc, std, cos in zip(ugc_sentences, std_sentences, X_cos):
        print(ugc)
        print(std)
        print(cos)
        print()
    
    print("Average across", str(X_cos.size), "sentences:", X_cos.mean())
    print()
    



