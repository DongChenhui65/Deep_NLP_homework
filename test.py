import os
import math
import jieba
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk import FreqDist


def file_conduct():
    corpus = []
    datapath = "./jyxstxtqj_downcc.com"
    filelist = os.listdir(datapath)

    # 去除无效信息
    for filename in filelist:
        filepath = datapath + '/' + filename
        with open(filepath, "r", encoding="gb18030") as file:
            filecontext = file.read()
            filecontext = filecontext.replace(
                "本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
            filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站", '')
            corpus.append(filecontext)
            file.close()

    fw = open('all_sentence.txt', 'w', encoding='utf-8')

    for filecontext in corpus:
        filecontext = filecontext.replace('\n', '')  # 去除换行符
        filecontext = filecontext.replace(' ', '')  # 去除空格
        filecontext = filecontext.replace('\u3000', '')  # 去除全角空白符
        fw.write(filecontext.strip(' ') + '\n')  # 去除字符串前后控空格

    fw.close()


def word_num_sum():  # 统计字数以及去除段前后的无效字符
    with open('all_sentence.txt', 'r', encoding='utf-8') as f:
        corpus = []
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
    return corpus


# 使用jieba库进行分词实验结果
def jieba_split(res):
    split_words = []
    words_num = 0
    count_all = 0
    for line in res:
        for x in jieba.cut(line):
            split_words.append(x)
            # words_num += 1
    # 词频统计
    word_fc = FreqDist(split_words)

    # 去除停用词
    stopword_file = open('cn_stopwords.txt', "r", encoding='utf-8')
    # stopword_file = open('cn_punctuation.txt', "r", encoding='utf-8')
    stopwordlist = stopword_file.read().split('\n')  # 分解字符串
    stopword_file.close()
    for stopword in stopwordlist:
        del word_fc[stopword]

    for key, value in word_fc.items():
        count_all += len(key) * value
        words_num += value

    word_fc_sort = sorted(word_fc.items(), key=lambda x: x[1], reverse=True)
    print('jieba分词：')
    print("语料库字数:", count_all)
    print("分词个数:", words_num)
    print("平均词长:", round(count_all / words_num, 5))

    entropy = []
    entropy = [-(uni_word[1] / words_num) * math.log(uni_word[1] / words_num, 2) for uni_word in word_fc_sort]
    print("基于jieba分割的中文平均信息熵为:", round(sum(entropy), 5), "比特/词")
    # print("基于jieba分割的中文平均信息熵为:", round(sum(entropy) / len(entropy), 5), "比特/词", "\n")

    ranks = []
    freqs = []
    for rank, value in enumerate(word_fc_sort):  # 0 ('的', 87343)
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1

    plt.loglog(ranks, freqs)
    plt.title('jieba分词', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.xlabel('词语频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('词语名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.grid(True)
    plt.show()
    ############################################


# 按字分词实验结果（与jieba_split类似)
def normal_split(res):
    split_words = []
    words_num = 0
    count_all = 0
    for line in res:
        for x in line:
            split_words.append(x)
            # words_num += 1
    # 词频统计
    word_fc = FreqDist(split_words)

    # 去除停用词
    stopword_file = open('cn_stopwords.txt', "r", encoding='utf-8')
    # stopword_file = open('cn_punctuation.txt', "r", encoding='utf-8')
    stopwordlist = stopword_file.read().split('\n')  # 分解字符串
    stopword_file.close()
    for stopword in stopwordlist:
        del word_fc[stopword]

    for key, value in word_fc.items():
        count_all += len(key) * value
        words_num += value

    word_fc_sort = sorted(word_fc.items(), key=lambda x: x[1], reverse=True)
    print('按字分词：')
    print("语料库字数:", count_all)
    print("分词个数:", words_num)
    print("平均词长:", round(count_all / words_num, 5))

    entropy = []
    entropy = [-(uni_word[1] / words_num) * math.log(uni_word[1] / words_num, 2) for uni_word in word_fc_sort]
    print("基于按字分割的中文平均信息熵为:", round(sum(entropy), 5), "比特/词")
    # print("基于jieba分割的中文平均信息熵为:", round(sum(entropy) / len(entropy), 5), "比特/词", "\n")

    ranks = []
    freqs = []
    for rank, value in enumerate(word_fc_sort):  # 0 ('的', 87343)
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1

    plt.loglog(ranks, freqs)
    plt.title('按字分词', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.xlabel('词语频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('词语名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.grid(True)
    plt.show()
    ############################################


if __name__ == "__main__":
    file_conduct()
    corpus = word_num_sum()
    jieba_split(corpus)
    normal_split(corpus)
