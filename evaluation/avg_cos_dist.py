import os, argparse, pandas as pd
import json 
import matplotlib.pyplot as plt
import seaborn as sns

MODELS = {
    'laser2': 'LASER',
    'rolaser': 'RoLASER',
    'c-rolaser': 'c-RoLASER'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', help='path to output directory', type=str, default='.')
    args = parser.parse_args()

    score_files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.startswith('outputs_') and f.endswith('.json')]

    score_dataframes = []
    for score_file in score_files:
        model = MODELS[score_file.split('_')[1].split('.')[0]]
        scores = pd.DataFrame.from_dict(json.load(open(score_file)), orient='index')
        scores['model'] = model
        score_dataframes.append(scores)

    all_scores = pd.concat(score_dataframes)
    all_scores.to_csv(os.path.join(args.output_dir, 'all_scores.csv'))
    all_scores[['model', 'cos']].groupby('model').describe().to_csv(os.path.join(args.output_dir, 'scores_summary.csv'))

    sns.boxplot(data=all_scores, x='model', y='cos', hue='model', legend=False)
    plt.ylabel('Cosine distance')
    plt.xlabel('Model')
    plt.savefig(os.path.join(args.output_dir, 'cosine_distance.png'))

