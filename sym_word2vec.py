# encoding=utf-8

import jieba
import json
import re
import string
import gensim

import numpy as np
np.random.seed(1337)  # for reproducibility

from gensim.models import Word2Vec
from gensim.models.wrappers.fasttext import FastText as FT_wrapper
import logging
import os
import sys

from collections import defaultdict

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sym2id_dict = dict()
dis2id_dict = dict()



def punc_filter(text):
    string=re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）:：．～]+".decode("utf8"), "".decode("utf8"), text)
    string=re.sub("[\u4e00-\u9fa5a-zA-Z0-9]".decode("utf8"), "".decode("utf8"), string)
    return string


#模型的训练

jieba.load_userdict('./symptom_dict_jieba.dict')
corpus_sentences = []
with open("./medical_baike_corpus.txt", 'r') as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue

        # if len(line.decode('utf-8')) < 4:
        #     continue
        # line = punc_filter(line.decode('utf8'))
        # line = line.encode('utf8')

        # line_seg = jieba.cut(line)
        # line_list = " ".join(line_seg)        
        # line_list = line_list.encode('utf8').split(' ')

        line_list = line.strip().split(" ")
        # print line_list
        corpus_sentences.append(line_list)
        if len(corpus_sentences) % 10000 == 0:
            print "%s\%s" % (len(corpus_sentences), "175509")



# # for fasttext
# # Set FastText home to the path to the FastText executable
# ft_home = '/Users/baidu/fastText/fasttext'
# lee_train_file = './fasttext_corpus.txt'
# # train the model
# model_wrapper = FT_wrapper.train(ft_home, lee_train_file)
# print(model_wrapper)
# model_wrapper.save('fasttext_word2vector.model')
# # for fasttext


# for word2vec
model = gensim.models.Word2Vec(corpus_sentences, size=200, workers=8)
model.save('word2vector_baike.model')
print(model)
# for word2vec

'''

# with open('../new_symptom_mapping_id.txt', 'r') as f:
#     for line in f:
#         sym_name, sym_id= line.strip().split(' ')
#         sym2id_dict[sym_name] = sym_id

with open('./disease_list.txt', 'r') as f:
    for line in f:
        dis_name = line.strip()
        dis2id_dict[dis_name] = 1

# 模型的加载
if __name__ == '__main__':
    model = Word2Vec.load('./word2vector_baike.model')
    # ft_model = FT_wrapper.load('fasttext_word2vector.model')
    # query_word = sys.argv[1].split(' ')
    # ft_query_word = [u.decode('utf-8') for u in query_word]
    # try:
    #     result = model.most_similar(positive=query_word, topn=20)
    #     ft_result = ft_model.most_similar(positive=ft_query_word, topn=20)
    #     # print model.similarity('症状','持续性')
    #     print "word2vec"
    #     for item in result:
    #         if sym2id_dict.has_key(item[0]):
    #         # if dis2id_dict.has_key(item[0]):
    #             print('   "'+item[0]+'"  下一轮询问概率:'+str(item[1]))
    #     print "fasttext"
    #     for item in ft_result:
    #         if sym2id_dict.has_key(item[0].encode('utf-8')):
    #     # if dis2id_dict.has_key(item[0]):
    #             print('   "'+item[0].encode('utf-8')+'"  下一轮询问概率:'+str(item[1]))
    
    # for dis in dis2id_dict.keys():
    #     try:
    #         result = model.most_similar(positive=dis, topn=100)
    #         if len(result) != 0:
    #             print dis
    #             for item in result:
    #                 # if sym2id_dict.has_key(item[0]):
    #                 print('   "'+item[0]+'"  转移概率:'+str(item[1]))

    #     except Exception as e:
    #         pass
 
    doc_content = defaultdict(list)
    doc_content_score = defaultdict(dict)
    dis = "胃镜"
    doc_rank = defaultdict(float)

    with open('medical_doc_xiaohua.txt', 'r') as f:
        for line in f:
            doc, intro = line.strip().split('\t')
            intro_list = intro.strip().split(' ')
            doc_content[doc] = intro_list

    for doc, content in doc_content.items():
        # print doc
        for entity in content:
            try:
                score = model.similarity(dis, entity)
                doc_content_score[doc][entity] = score
                # print "similartiy %s\t%s\t%s" % (dis, entity, score)
            except Exception as e:
                pass

    for doc, entity_score_list in doc_content_score.items():
        # print doc
        entity_scores = sorted(entity_score_list.items(), key=lambda d:d[1], reverse=True)
        for e in entity_scores[0:1]:
            doc_rank[doc] = e[1]
            print doc, e[0],e[1]

    # for k,v in doc_rank.items():
    #     print k,v
    print dis
    doc_rank_list = sorted(doc_rank.items(), key=lambda d:d[1], reverse=True)
    for doc, score in doc_rank_list:
        print doc, score
'''
