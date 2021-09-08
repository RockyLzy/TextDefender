from typing import NamedTuple
import re


def load_sentences(file_path, sep=r"\s+"):
    ret = []
    sentence = []
    for line in open(file_path, encoding='utf8'):
        line = line.strip("\n")
        if line == "":
            if not sentence == []:
                ret.append(sentence)
                sentence = []
        else:
            sentence.append(re.split(sep, line))
    return ret


def permute_cols(*files, pidx, out_file):
    contents = []
    for file in files:
        sentences = load_sentences(file)
        contents.append(sentences)

    if out_file is None:
        out = None
    else:
        out = open(out_file, "w", encoding='utf8')
    # out = None
    for s_id in range(len(contents[0])):
        for r_id in range(len(contents[0][s_id])):
            line = []
            for f_id, c_id in pidx:
                line.append(contents[f_id][s_id][r_id][c_id])
            print(" ".join(line), file=out)
        print("", file=out)


# permute_cols("../dataset/ontonotes4/train.mix.bmes",
#              pidx=((0, 0), (0, 2)),
#              out_file="../dataset/ontonotes4/train.pos.bmes")


# name = 'test'
# pred = 'nlpir'
# permute_cols("../dataset/ontonotes4/{}.mix.bmes".format(name),
#              "../dataset/ontonotes4/{}.seg.bmes.{}".format(name, pred),
#              "../dataset/ontonotes4/{}.pos.bmes.{}".format(name, pred),
#              pidx=((0, 0), (1, 1), (2, 1), (0, 3)),
#              out_file="../dataset/ontonotes4/{}.mix.bmes.{}".format(name, pred))


# name = 'train'
# permute_cols("../dataset/resume/{}.ner.bmes".format(name),
#              "../dataset/resume/{}.seg.bmes.thu".format(name),
#              "../dataset/resume/{}.pos.bmes.thu".format(name),
#              pidx=((0, 0), (1, 1), (2, 1), (0, 1)),
#              out_file="../dataset/resume/{}.mix.bmes.thu".format(name))


def check_seg(file1, file2):
    content_1 = load_sentences(file1)
    content_2 = load_sentences(file2)
    corr_num = 0
    total_num = 0
    ner_wrong_num = 0
    ner_total_num = 0
    for s_id in range(len(content_1)):
        for r_id in range(len(content_1[s_id])):
            total_num += 1
            if content_1[s_id][r_id][3] != 'O':
                ner_total_num += 1

            if content_1[s_id][r_id][1] == content_2[s_id][r_id][1]:
                corr_num += 1
            else:
                print(content_1[s_id][r_id][3])
                if content_1[s_id][r_id][3] != 'O':
                    ner_wrong_num += 1
    print(corr_num / total_num)
    print(ner_wrong_num / (total_num - corr_num))
    print(ner_wrong_num / ner_total_num)

# check_seg(
#     "../dataset/ontonotes4/dev.mix.bmes",
#     "../dataset/ontonotes4/dev.seg.bmes.pred",)
