import pickle

import json
import numpy as np

from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM


def create_perturbation_set_from_attacker(output_file_path, max_candidate):
    perturb = {}

    transformation = WordSwapEmbedding(max_candidates=max_candidate)

    word_list_file = '/home/lizongyi/.cache/textattack/word_embeddings/paragramcf/wordlist.pickle'
    word2index = np.load(word_list_file, allow_pickle=True)

    for word in word2index.keys():
        perturb[word] = transformation._get_replacement_words(word)

    with open(f'{output_file_path}/textfooler_{max_candidate}.json', 'w', encoding='utf-8') as fout:
        json.dump(perturb, fout)


if __name__ == '__main__':
    create_perturbation_set_from_attacker('/disks/sdb/lzy/adversarialBenchmark/dne_external_data', 50)
