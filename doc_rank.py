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
        self.doc_title = defaultdict(int)

        self.TOPK = 1

        with open('./doc_dep_info_all_new.txt', 'r') as f:
            for line in f:
                try:
                    doc, title_no, dep, info, intro = line.strip().split('\t')
                    self.doc_class_dict[doc] = dep
                    self.doc_info_dict[doc] = info
                    self.doc_title[doc] = title_no
                    self.class_doc[dep].append(doc)
                    intro_list = intro.strip().split(' ')
                    self.doc_content[doc] = intro_list
                except:
                    print line

    def rank(self, dis, department, topk=1):
        """
        Use word2vector to calculate similarity
        """
        doc_content_score = defaultdict(dict)
        doc_rank = defaultdict(tuple)
        # calculate similarity
        for doc, content in self.doc_content.items():
            # print doc
            if doc not in self.class_doc[department]:
                continue
            for entity in content:
                try:
                    score = self.model.similarity(dis, entity)
                    doc_content_score[doc][entity] = score
                    # print "similartiy %s\t%s\t%s" % (dis, entity, score)
                except Exception as e:
                    pass

        # rank the similarity, pick the max similarity as doctor score
        for doc, entity_score_list in doc_content_score.items():
            # print doc
            entity_scores = sorted(entity_score_list.items(), key=lambda d:d[1], reverse=True)
            for e in entity_scores[0:1]:
                try:
                    doc_rank[doc] = (e[1], int(self.doc_title[doc]))
                    # print doc, e[0], e[1]
                except Exception as e:
                    pass
                

        doc_rank_list = sorted(doc_rank.items(), key=lambda d: (d[1][0], -d[1][1]), reverse=True)

        # return the doctor info
        doc_class = department
        try:
            doc_score = doc_rank_list[topk-1][1][0]
            if doc_score > 0.7:
                doc_name = doc_rank_list[topk-1][0]
                doc_info = self.doc_info_dict[doc_name]
                doc_title_no = self.doc_title[doc_name]
            else:
                doc_name = ""
                doc_info = ""
                doc_title_no = ""

        except Exception as e:
            doc_name = ""
            doc_info = ""
            doc_title_no = ""
            pass

        return doc_name, doc_class, doc_info, doc_title_no


    def rankDoc(self, dis, department, n):
        rankDocList = []
        for i in range(n):
            rankDocList.append(self.rank(dis, department, i+1))
        return rankDocList

    def randomDoc(self, dis, department, n):
        rankDocList = []
        for doc in self.class_doc[department][0:n]:
            rankDocList.append((doc, department, self.doc_info_dict[doc]))
        return rankDocList

    def rankList(self, dis, department, n):
        docList = self.rankDoc(dis, department, n)
        # print docList
        if docList[0][0] != "":
            return "rank", docList
        else:
            return "default", self.randomDoc(dis, department, n)

if __name__ == '__main__':
    # dis = sys.argv[1]
    # department = sys.argv[2]
    ranker = DoctorRanking()
    # dis = "外阴瘙痒"
    # department = "儿科"
    # doc_name, doc_class, doc_info = ranker.rank(dis, department, 3)
    with open('doc_rank_test.txt', 'r') as f:
        for line in f:
            _, department, dis = line.strip().split('\t')
            t, docList = ranker.rankList(dis, department, 1)
            # print t,docList
            # print len(docList)
            try:
                for doc in docList:
                    doc_name, doc_class, doc_info = doc[0], doc[1], doc[2]
                    # print "患者输入的疾病: %s" % dis
                    # print "系统推荐医生信息:"
                    # print "%s\t%s\t%s\t%s" % (doc_name, doc_class, doc_info, doc_title_no)
                    print "%s\t%s\t%s\t%s\t%s" % (t, dis, doc_name, doc_class, doc_info)
            except:
                pass
                # print line
                # print docList

    # dis = "牙痛"
    # department = "口腔科"
    # docList = ranker.rankList(dis, department, 3)
    # print len(docList)
    # for doc in docList:
    #     doc_name, doc_class, doc_info, doc_title_no = doc[0], doc[1], doc[2], doc[3]
    #     print "患者输入的疾病: %s" % dis
    #     print "系统推荐医生信息:"
    #     print "%s\t%s\t%s\t%s" % (doc_name, doc_class, doc_info, doc_title_no)

    # dis = "肺炎"
    # department = "呼吸内科"
    # docList = ranker.rankList(dis, department, 3)
    # print len(docList)
    # for doc in docList:
    #     doc_name, doc_class, doc_info, doc_title_no = doc[0], doc[1], doc[2], doc[3]
    #     print "患者输入的疾病: %s" % dis
    #     print "系统推荐医生信息:"
    #     print "%s\t%s\t%s\t%s" % (doc_name, doc_class, doc_info, doc_title_no)
