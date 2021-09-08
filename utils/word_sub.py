import os
import string
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk import word_tokenize
import networkx as nx
import argparse

import sys
sys.path.append("..")
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM


def filename(embd_path, dataset, filename):
    return os.path.join(embd_path, dataset, filename)


def get_wordembd(embd_path):
    word_embd = {}
    embd_file = os.path.join(embd_path, 'counter-fitted-vectors.txt')
    with open(embd_file, "r") as f:
        tem = f.readlines()
        for line in tem:
            line = line.strip()
            line = line.split(' ')
            word = line[0]
            vec = line[1:len(line)]
            vec = [float(i) for i in vec]
            vec = np.asarray(vec)
            word_embd[word] = vec

    Name = embd_path + '/word_embd.pkl'
    output = open(Name, 'wb')
    pickle.dump(word_embd, output)
    output.close()


def get_vocabluary(dataset, data_path, embd_path):
    print('Generate vocabulary')
    pkl_file = open(embd_path + 'word_embd.pkl', 'rb')
    word_embd = pickle.load(pkl_file)
    pkl_file.close()
    Name = filename(embd_path, dataset, 'vocab.pkl')

    if dataset != 'snli':
        vocab = {}
        folder_lists = [data_path]
        for folder_list in folder_lists:
            for input_file in os.listdir(folder_list):
                # if os.path.isdir(os.path.join(folder_list,folder)):
                #     for input_file in os.listdir(os.path.join(folder_list,folder)):
                if input_file.endswith(".txt"):
                    with open(os.path.join(folder_list, input_file), "r") as f:
                        while True:
                            tem_text = f.readline().strip()
                            if tem_text[-2:] == '\t1' or tem_text[-2:] == '\t1':
                                tem_text = tem_text[:-2]
                            if tem_text:
                                tem_text = tem_text.translate(str.maketrans('', '', string.punctuation))
                                tem_text = tem_text.split(' ')
                                for word in tem_text:
                                    if word in vocab.keys():
                                        vocab[word]['freq'] = vocab[word]['freq'] + 1
                                    else:
                                        if word in word_embd.keys():
                                            vocab[word] = {'vec': word_embd[word], 'freq': 1}
                            else:
                                break

        output = open(Name, 'wb')
        pickle.dump(vocab, output)
        output.close()
        print('Finish Generate vocabulary')

    elif dataset == 'snli':
        vocab = {}
        folder_lists = [data_path]
        for folder_list in folder_lists:
            for input_file in os.listdir(folder_list):
                # if os.path.isdir(os.path.join(folder_list,folder)):
                #     for input_file in os.listdir(os.path.join(folder_list,folder)):
                if input_file.endswith(".txt"):
                    with open(os.path.join(folder_list, input_file), "r") as f:
                        while True:
                            tem_text = f.readline().strip()
                            if tem_text:
                                tem_text = tem_text.translate(str.maketrans('', '', string.punctuation))
                                pos = tem_text.find('\t')
                                tem_text = tem_text[pos+1:]
                                tem_text = tem_text.split(' ')
                                for word in tem_text:
                                    if word in vocab.keys():
                                        vocab[word]['freq'] = vocab[word]['freq'] + 1
                                    else:
                                        if word in word_embd.keys():
                                            vocab[word] = {'vec': word_embd[word], 'freq': 1}
                            else:
                                break

        output = open(Name, 'wb')
        pickle.dump(vocab, output)
        output.close()
        print('Finish Generate Amazon vocabulary')


def process_with_all_but_not_top(dataset, embd_path):
    # code for processing word embd using all-but-not-top
    print('Process word embd using all-but-not-top')
    pkl_file = open(filename(embd_path, dataset, 'vocab.pkl'), 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    num_word = len(vocab)
    dim_vec = len(vocab['high']['vec'])
    embd_matrix = np.zeros([num_word, dim_vec])
    embd_matrix0 = np.zeros([num_word, dim_vec])

    count = 0
    tem_list = []
    for key in vocab.keys():
        tem_vec = vocab[key]['vec']
        tem_vec = tem_vec / np.sqrt((tem_vec ** 2).sum())
        embd_matrix[count, :] = tem_vec
        tem_list.append(key)
        count += 1

    mean_embd_matrix = np.mean(embd_matrix, axis=0)
    for i in range(embd_matrix.shape[0]):
        embd_matrix0[i, :] = embd_matrix[i, :] - mean_embd_matrix
    covMat = np.cov(embd_matrix0, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(-eigVals)
    eigValIndice = eigValIndice[0:8]
    n_eigVect = eigVects[:, eigValIndice]
    embd_matrix = embd_matrix0 - np.dot(np.dot(embd_matrix, n_eigVect), n_eigVect.T)

    Name = filename(embd_path, dataset, 'embd_pca.pkl')
    output = open(Name, 'wb')
    pickle.dump(embd_matrix, output)
    output.close()

    # update vocabulary
    count = 0
    for key in tem_list:
        vocab[key]['vec'] = embd_matrix[count, :].flatten()
        count += 1

    Name = filename(embd_path, dataset, 'vocab_pca.pkl')

    output = open(Name, 'wb')
    pickle.dump(vocab, output)
    output.close()

    print('Finish Process word embd using all-but-not-top')


def get_word_substitution_table(dataset, embd_path, similarity_threshold=0.8):
    print('Generate word substitude table')

    pkl_file = open(filename(embd_path, dataset, 'vocab_pca.pkl'), 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    counterfitted_neighbor = {}
    key_list = list(vocab.keys())
    similarity_num_threshold = 100000
    freq_threshold = 1
    neighbor_network_node_list = []
    neighbor_network_link_list = []

    num_word = len(key_list)
    dim_vec = vocab[key_list[0]]['vec'].shape[1]

    embd_matrix = np.zeros([num_word, dim_vec])
    for _ in range(len(key_list)):
        embd_matrix[_, :] = vocab[key_list[_]]['vec']

    for _ in tqdm(range(len(key_list))):
        word = key_list[_]

        if vocab[word]['freq'] > freq_threshold:

            counterfitted_neighbor[word] = []
            neighbor_network_node_list.append(word)

            dist_vec = np.dot(embd_matrix[_, :], embd_matrix.T)
            dist_vec = np.array(dist_vec).flatten()

            idxes = np.argsort(-dist_vec)
            idxes = np.where(dist_vec > similarity_threshold)
            idxes = idxes[0].tolist()

            tem_num_count = 0
            for ids in idxes:
                if key_list[ids] != word and vocab[key_list[ids]]['freq'] > freq_threshold:
                    counterfitted_neighbor[word].append(key_list[ids])
                    neighbor_network_link_list.append((word, key_list[ids]))
                    tem_num_count += 1
                    if tem_num_count >= similarity_num_threshold:
                        break

        if _ % 2000 == 0:
            neighbor = {'neighbor': counterfitted_neighbor, 'link': neighbor_network_link_list,
                        'node': neighbor_network_node_list}
            Name = filename(embd_path, dataset, 'neighbor_constraint_pca' + str(similarity_threshold) + '.pkl')
            output = open(Name, 'wb')
            pickle.dump(neighbor, output)
            output.close()

    neighbor = {'neighbor': counterfitted_neighbor, 'link': neighbor_network_link_list,
                'node': neighbor_network_node_list}
    Name = filename(embd_path, dataset, 'neighbor_constraint_pca' + str(similarity_threshold) + '.pkl')

    output = open(Name, 'wb')
    pickle.dump(neighbor, output)
    output.close()
    print('Finish Generate word substitude table')


def get_perturbation_set(dataset, embd_path, similarity_threshold=0.8, perturbation_constraint=100):
    # code for generate perturbation set
    print('Generate perturbation set')
    freq_threshold = 1

    pkl_file = open(filename(embd_path, dataset, 'neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'), 'rb')
    neighbor = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(filename(embd_path, dataset, 'vocab.pkl'), 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    counterfitted_neighbor = neighbor['neighbor']
    neighbor_network_node_list = neighbor['node']
    neighbor_network_link_list = neighbor['link']
    perturb = {}

    size_threshold = perturbation_constraint

    key_list = list(vocab.keys())
    num_word = len(key_list)
    dim_vec = vocab[key_list[0]]['vec'].shape[0]
    embd_matrix = np.zeros([num_word, dim_vec])
    for _ in range(len(key_list)):
        embd_matrix[_, :] = vocab[key_list[_]]['vec']

    # find independent components in the network
    G = nx.Graph()
    for node in neighbor_network_node_list:
        G.add_node(node)
    for link in neighbor_network_link_list:
        G.add_edge(link[0], link[1])

    for c in nx.connected_components(G):
        nodeSet = G.subgraph(c).nodes()
        if len(nodeSet) > 1:
            if len(nodeSet) <= perturbation_constraint:
                tem_key_list = nodeSet
                tem_key_list = list(tem_key_list)
                tem_num_word = len(tem_key_list)
                tem_embd_matrix = np.zeros([tem_num_word, dim_vec])
                # for _ in range(len(tem_key_list)):
                #     tem_embd_matrix[_, :] = vocab[tem_key_list[_]]['vec']
                for index, word in enumerate(tem_key_list):
                    tem_embd_matrix[index, :] = vocab[word]['vec']
                for node in nodeSet:
                    perturb[node] = {'set': G.subgraph(c).neighbors(node), 'isdivide': 0}
                    dist_vec = np.dot(vocab[node]['vec'], tem_embd_matrix.T)
                    dist_vec = np.array(dist_vec).flatten()
                    idxes = np.argsort(-dist_vec)
                    tem_list = []
                    for ids in idxes:
                        if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                            tem_list.append(tem_key_list[ids])
                    perturb[node]['set'] = tem_list

            else:
                tem_key_list = nodeSet
                tem_num_word = len(tem_key_list)
                tem_key_list = list(tem_key_list)
                tem_embd_matrix = np.zeros([tem_num_word, dim_vec])
                for _ in range(len(tem_key_list)):
                    tem_embd_matrix[_, :] = vocab[tem_key_list[_]]['vec']

                for node in tqdm(nodeSet):
                    perturb[node] = {'set': G.subgraph(c).neighbors(node), 'isdivide': 1}
                    if len(list(perturb[node]['set'])) > size_threshold:
                        raise ValueError('size_threshold is too small')

                    dist_vec = np.dot(vocab[node]['vec'], tem_embd_matrix.T)
                    dist_vec = np.array(dist_vec).flatten()
                    idxes = np.argsort(-dist_vec)
                    tem_list = []
                    tem_count = 0
                    for ids in idxes:
                        if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                            tem_list.append(tem_key_list[ids])
                            tem_count += 1
                        if tem_count == size_threshold:
                            break
                    perturb[node]['set'] = tem_list

    Name = filename(embd_path, dataset,
                    'perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(size_threshold) + '.pkl')
    output = open(Name, 'wb')
    pickle.dump(perturb, output)
    output.close()
    print('generate perturbation set finishes')
    print('-' * 89)


def get_perturbation_from_attacker(dataset, embd_path, attacker, max_candidate):
    print('Generate perturbation set from attacker')
    pkl_file = open(filename(embd_path, dataset, 'vocab.pkl'), 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    if attacker not in ['textfooler', 'pwws', 'bae']:
        exit(199)
    if attacker in ['textfooler', 'pwws']:
        transformation = WordSwapEmbedding(max_candidates=max_candidate)
    else:
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=max_candidate)

    perturb = {}
    for word in vocab:
        perturb[word] = {'set': transformation._get_replacement_words(word), 'isdivide': 0}

    Name = filename(embd_path, dataset,
                    f'perturbation_{attacker}_{max_candidate}.pkl')
    output = open(Name, 'wb')
    pickle.dump(perturb, output)
    output.close()
    print('generate perturbation set finishes')
    print('-' * 89)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset", default='agnews', type=str,
                        help="The name of data set: imdb or amazon")
    parser.add_argument("--data_path", default='/disks/sdb/lzy/adversarialBenchmark/dataset/', type=str,
                        help="The input data dir.")
    parser.add_argument("--embd_path", default='/disks/sdb/lzy/adversarialBenchmark/cache/embed/', type=str,
                        help="The data dir of embedding table.")
    parser.add_argument("--similarity_threshold", default=0.8, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--attacker", default='textfooler', type=str,
                        help="same perturbation set with attacker")
    parser.add_argument("--max_candidate", default=50, type=int,
                        help="max candidates of corresponding attacker")

    args = parser.parse_args()

    data_path = args.data_path
    embd_path = args.embd_path  # cache
    dataset = args.dataset
    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint
    attacker = args.attacker
    max_candidate = args.max_candidate

    if dataset not in ['imdb', 'sst2', 'agnews', 'snli']:
        raise ValueError('dataset not valid')

    embd_file = embd_path + '/word_embd.pkl'

    if not os.path.exists(embd_file):
        get_wordembd(embd_path)

    if not os.path.exists(os.path.join(embd_path, dataset)):
        os.makedirs(os.path.join(embd_path, dataset))

    if not os.path.exists(embd_path + '/' + dataset + '/vocab.pkl'):
        get_vocabluary(dataset, data_path, embd_path)

    if not os.path.exists(embd_path + '/' + dataset + '/' + 'embd_pca.pkl') or not os.path.exists(
            data_path + '/' + dataset + '/' + 'vocab_pca.pkl'):
        process_with_all_but_not_top(dataset, embd_path)

    if not os.path.exists(
            embd_path + '/' + dataset + '/' + 'neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'):
        get_word_substitution_table(dataset, embd_path, similarity_threshold=similarity_threshold)

    if not os.path.exists(
            embd_path + '/' + dataset + '/' + 'perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(
                    perturbation_constraint) + '.pkl'):
        get_perturbation_set(dataset, embd_path, similarity_threshold=similarity_threshold,
                             perturbation_constraint=perturbation_constraint)

    if not os.path.exists(
        f'{embd_path}/{dataset}/perturbation_{attacker}_{max_candidate}.pkl'):
        get_perturbation_from_attacker(dataset, embd_path, attacker, max_candidate)


if __name__ == "__main__":
    main()
