import sys, os, argparse
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize

assert os.environ.get('FAIRSEQ') and os.environ.get('LASER'), 'Please set the FAIRSEQ and LASER environment variables'

sys.path.append(os.environ['FAIRSEQ'])
sys.path.append(f"{os.environ['LASER']}/source")

from rolaser import RoLaserEncoder

DASHES = '-' * 40

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ugc-file', help='path to UGC data file', type=str, default='./data/demo_ugc.txt')
    parser.add_argument('--std-file', help='path to standard data file', type=str, default='./data/demo_std.txt')
    parser.add_argument('-m', '--model-name', help='model name', type=str)
    parser.add_argument('-d', '--model-dir', help='path to model directory', type=str, required=True)
    parser.add_argument('-t', '--tokenizer', help='tokenizer type', type=str, choices=['spm', 'roberta', 'char'], required=True)
    parser.add_argument('-o', '--output-dir', help='path to output directory', type=str, default='.')
    parser.add_argument('-v', '--verbose', help='print scores line by line', action='store_true')
    args = parser.parse_args()

    ugc_sentences = [ line.strip() for line in open(args.ugc_file).readlines() ]
    std_sentences = [ line.strip() for line in open(args.std_file).readlines() ]
    
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = os.path.basename(args.model_dir)
    model = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.pt')][0]
    vocab = [f.path for f in os.scandir(args.model_dir) if f.path.endswith('.cvocab')][0]

    model = RoLaserEncoder(model_path=model, vocab=vocab, tokenizer=args.tokenizer)
    
    X_std = model.encode(std_sentences)
    X_std = normalize(X_std)
    X_ugc = model.encode(ugc_sentences)
    X_ugc = normalize(X_ugc)

    X_cos = paired_cosine_distances(X_std, X_ugc)

    outputs = pd.DataFrame(columns=['ugc', 'std', 'cos'])
    outputs['ugc'] = ugc_sentences
    outputs['std'] = std_sentences
    outputs['cos'] = X_cos
    
    output_file = os.path.join(args.output_dir, f'outputs_{model_name}.json')
    outputs.to_json(output_file, orient='index')
    print('Outputs saved in', output_file)

    print(DASHES)
    print('Pairwise cosine distances from', model_name)
    print(DASHES)

    if args.verbose:
        for ugc, std, cos in zip(ugc_sentences, std_sentences, X_cos):
            print(ugc)
            print(std)
            print(cos)
            print()
        
    print("Average across", str(X_cos.size), "sentences:", X_cos.mean())
    print()
    



