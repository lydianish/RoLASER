import os, argparse
import numpy as np
from datasets import Dataset

assert os.environ.get('NLAUGMENTER'), 'Please set the NLAUGMENTER environment variable'

sys.path.append(os.environ['NLAUGMENTER'])

from nlaugmenter.transformations.abbreviation_transformation.transformation import Abbreviate
from nlaugmenter.transformations.insert_abbreviation.transformation import AbbreviationInsertionEN
from nlaugmenter.transformations.replace_abbreviation_and_acronyms.transformation import ReplaceAbbreviations
from nlaugmenter.transformations.butter_fingers_perturbation.transformation import ButterFingersPerturbation
from nlaugmenter.transformations.change_char_case.transformation import ChangeCharCase
from nlaugmenter.transformations.close_homophones_swap.transformation import CloseHomophonesSwap
from nlaugmenter.transformations.contraction_expansions.transformation import ContractionExpansions
from nlaugmenter.transformations.dyslexia_words_swap.transformation import DyslexiaWordsSwap
from nlaugmenter.transformations.leet_letters.transformation import LeetLetters
from nlaugmenter.transformations.replace_spelling.transformation import SpellingTransformation
from nlaugmenter.transformations.slangificator.transformation import Slangificator
from nlaugmenter.transformations.weekday_month_abbreviation.transformation import WeekdayMonthAbbreviation
from nlaugmenter.transformations.whitespace_perturbation.transformation import WhitespacePerturbation

SEED = 0
TOTAL_NOISE_PROBA = 0.1
TRANSFORMATIONS = ['abr1', 'abr2', 'abr3', 'fing', 'homo', 'cont', 'dysl', 'leet', 'spel', 'slng', 'week', 'spac'] # removed 'case' because LASER preprocessing includes lowercasing

def sample_prob(default_prob):
    return np.random.choice([default_prob/2, default_prob, 3*default_prob/2], p=[1/4, 1/2, 1/4])

def init_transformation(name, seed=0, max_outputs=1):
    if name not in TRANSFORMATIONS:
        raise AttributeError(name + " was not found in the list of known transformations...")
    if name == "abr1":
        prob = sample_prob(0.1)
        return [ Abbreviate(prob=prob, seed=seed+1, max_outputs=max_outputs), name, prob ]
    if name == "abr2":
        return [ AbbreviationInsertionEN(seed=seed+2, max_outputs=max_outputs), name ]
    if name == "abr3":
        return [ ReplaceAbbreviations(seed=seed+3, max_outputs=max_outputs, case_sensitive=False), name ]
    if name == "fing":
        prob = sample_prob(0.05)
        return [ ButterFingersPerturbation(prob=prob, seed=seed+4, max_outputs=max_outputs), name, prob ]
    if name == "case":
        prob = sample_prob(0.1)
        return [ ChangeCharCase(prob=prob, seed=seed+5, max_outputs=max_outputs), name, prob ]
    if name == "homo":
        corrupt_prob = sample_prob(0.5)
        return [ CloseHomophonesSwap(corrupt_prob=corrupt_prob, seed=seed+6, max_outputs=max_outputs), name, corrupt_prob ]
    if name == "cont":
        return [ ContractionExpansions(), name ]
    if name == "dysl":
        return [ DyslexiaWordsSwap(seed=seed+7, max_outputs=max_outputs), name ]
    if name == "leet":
        max_leet = sample_prob(0.05)
        return [ LeetLetters(max_leet=max_leet, seed=seed+8, max_outputs=max_outputs), name, max_leet ]
    if name == "spel":
        prob = sample_prob(0.2)
        return [ SpellingTransformation(prob=prob, seed=seed+9, max_outputs=max_outputs), name, prob ]
    if name == "slng":
        return [ Slangificator(seed=seed+10, max_outputs=max_outputs), name ]
    if name == "week":
        return [ WeekdayMonthAbbreviation(), name ]
    if name == "spac":
        remove_prob = sample_prob(0.1)
        add_prob = sample_prob(0.05)
        return [ WhitespacePerturbation(remove_prob=remove_prob, add_prob=add_prob, seed=seed+11, max_outputs=max_outputs), name, remove_prob, add_prob ]
    
def corrupt_sentence(sentence, prob=TOTAL_NOISE_PROBA, seed=SEED, trans=TRANSFORMATIONS):
    trans = np.array(trans)
    transformations_to_apply = [ init_transformation(name, seed=seed) for name in trans[np.random.random(trans.size) < prob ] ]
    np.random.shuffle(transformations_to_apply)
    transformations = []
    for t in transformations_to_apply:
        sentence = t[0].generate(sentence)[0]
        transformations.append(','.join([str(param) for param in t[1:]]))
    new_sentence = sentence.rstrip() + ' \n'
    applied_transformations = ';'.join(transformations) + '\n'
    return new_sentence, applied_transformations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", dest="input_file", help="path to raw input file", type=str)
    parser.add_argument("-s", "--seed", help="random seed", type=int, default=SEED)
    parser.add_argument("-p", "--prob", help="probalility of adding UGC phenomena", type=float, default=TOTAL_NOISE_PROBA)
    args = parser.parse_args()

    def corrupt_example(example):
        return {'output': corrupt_sentence(example['sentence'], args.prob, args.seed)}

    dirname, basename = os.path.split(args.input_file)
    ugc_dir =  os.path.join(dirname, "ugc", str(args.seed))
    trans_dir =  os.path.join(dirname, "trans", str(args.seed))
    
    os.makedirs(ugc_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    filename, file_extension = os.path.splitext(basename)
    output_file = os.path.join(ugc_dir, filename + "_mix_all" + file_extension)
    transformation_file = os.path.join(trans_dir, filename + "_mix_all_trans" + file_extension)

    np.random.seed(args.seed)
    
    with open(args.input_file, "r") as f1:
        sentences = f1.readlines()
    
    ds = Dataset.from_dict({"sentence": sentences})
    ds = ds.map(corrupt_example)
    ds = ds.add_column("new_sentence", np.array(ds["output"])[:,0])
    ds = ds.add_column("transformations", np.array(ds["output"])[:,1])
    
    print("Writing new sentences in", output_file)
    with open(output_file, "w") as f2:
        f2.writelines(ds["new_sentence"])
    
    print("Writing transformations in", transformation_file)
    with open(transformation_file, "w") as f3:
        f3.writelines(ds["transformations"])
