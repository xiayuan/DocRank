#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
This module use word2vector for doctor ranking
Authors: xiayuan(xiayuan@baidu.com)
Date:    2018/07/25 14:23:02

"""

import jieba
import json
import re
import string
import gensim

import numpy as np
np.random.seed(1337)  # for reproducibility

from gensim.models import Word2Vec
# from gensim.models.wrappers.fasttext import FastText as FT_wrapper

from collections import defaultdict
import logging
import os
import sys


sym2id_dict = dict()
dis2id_dict = dict()


# with open('/Users/baidu/Desktop/baidu_work/melody/new_symptom_mapping_id.txt', 'r') as f:
#     for line in f:
#         sym_name, sym_id = line.strip().split(' ')
#         sym2id_dict[sym_name] = sym_id

# with open('/Users/baidu/Desktop/baidu_work/melody/mapping_disease_name_id.txt', 'r') as f:
#     for line in f:
#         dis_name, dis_id = line.strip().split('\t')
#         dis2id_dict[dis_name] = dis_id


class DoctorRanking(object): 
    """
    Class for DoctorRanking
    """
    def __init__(self, *args, **kwargs):
        """
        """
        self.model =  Word2Vec.load('./word2vector_baike.model')
        self.doc_class_dict = dict()
        self.doc_info_dict = dict()
        self.class_doc = defaultdict(list)
        self.doc_content = defaultdict(list)
        self.doc_content_score = defaultdict(dict)
        self.doc_rank = defaultdict(float)

        self.doc_name = ""
        self.doc_info = ""
        self.doc_class = ""

        self.TOPK = 1
        self.rankDocList = []

        with open('./doc_dep_info_all.txt', 'r') as f:
            for line in f:
                doc, dep, info, intro = line.strip().split('\t')
                self.doc_class_dict[doc] = dep
                self.doc_info_dict[doc] = info
                self.class_doc[dep].append(doc)
                intro_list = intro.strip().split(' ')
                self.doc_content[doc] = intro_list

    def rank(self, dis, department, topk=1):
        """
        Use word2vector to calculate similarity
        """
        # calculate similarity
        for doc, content in self.doc_content.items():
            # print doc
            if doc not in self.class_doc[department]:
                continue
            for entity in content:
                try:
                    score = self.model.similarity(dis, entity)
                    self.doc_content_score[doc][entity] = score
                    # print "similartiy %s\t%s\t%s" % (dis, entity, score)
                except Exception as e:
                    pass

        # rank the similarity, pick the max similarity as doctor score
        for doc, entity_score_list in self.doc_content_score.items():
            # print doc
            entity_scores = sorted(entity_score_list.items(), key=lambda d:d[1], reverse=True)
            for e in entity_scores[0:1]:
                self.doc_rank[doc] = e[1]
                # print doc, e[0], e[1]

        doc_rank_list = sorted(self.doc_rank.items(), key=lambda d:d[1], reverse=True)

        # return the doctor info
        try:
            self.doc_name = doc_rank_list[topk][0]
            self.doc_info = self.doc_info_dict[self.doc_name]
            self.doc_class = department
        except Exception as e:
            pass

        return self.doc_name, self.doc_class, self.doc_info


    def rankDoc(self, dis, department, n):
        for i in range(n):
            self.rankDocList.append(self.rank(dis, department, i+1))
        return self.rankDocList

    def randomDoc(self, dis, department, n):
        for doc in self.class_doc[department][0:n]:
            self.rankDocList.append((doc, department, self.doc_info_dict[doc]))
        return self.rankDocList

    def rankList(self, dis, department, n):
        docList = self.rankDoc(dis, department, n)
        # print docList
        if docList[0][0] != "":
            return docList
        else:
            self.rankDocList = []
            return self.randomDoc(dis, department, n)

if __name__ == '__main__':
    dis = sys.argv[1]
    department = sys.argv[2]
    # dis = "心肌炎"
    # department = "心血管内科"
    ranker = DoctorRanking()
    # doc_name, doc_class, doc_info = ranker.rank(dis, department, 3)
    docList = ranker.rankList(dis, department, 3)
    for doc in docList:
        doc_name, doc_class, doc_info = doc[0],doc[1],doc[2]
        print "患者输入的疾病: %s" % dis
        print "系统推荐医生信息:"
        print "%s\t%s\t%s" % (doc_name, doc_class, doc_info)
