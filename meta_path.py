#coding=utf-8
import random
from copy import deepcopy
import numpy as np
import io
import os
# from sklearn import metrics
import tensorflow as tf
import copy

# add an element to 2-array dictionary
def addtodict2(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a: {key_b: val}})


data_query = {}
data_title = {}
threshold = 3.5
query_nei = {}
title_nei = {}
input_data='./eval.tsv'
output_data='./eval_nei.tsv'
#input_data='../../../human_data/222data.tsv'
#output_data='./eval_nei.tsv'
with io.open(input_data, 'r', encoding='utf-8') as f:
    for i, line1 in enumerate(f):
        line = line1.strip().split('\t')
        query = line[0]
        title = line[1]
        score = float(line[2]) 
        if len(title) == 0:
            print("title none is: ", i)
        if score >= threshold:
            addtodict2(data_query, query, title, score)
            addtodict2(data_title, title, query, score)
print("Finished: reading file as a dict.")

null_q = 0
for query in data_query:    ##for key in a和 for key in a.keys():完全等价,但是前者效率更高
    neighbor = sorted(data_query[query].items(), key=lambda item:item[1], reverse=True) ##对data_query进行排序
    nei_num = len(neighbor)
    if nei_num == 0:
        null_q += 1
        query_nei.setdefault(query, [])
    elif nei_num == 1:
        query_nei.setdefault(query, []).extend(neighbor[0][0])
    else:
        query_nei.setdefault(query, []).extend([neighbor[0][0], neighbor[1][0]])
print("total query 's num is: ", len(data_query))
print("null query 's num is: ", null_q)

null_t = 0
for title in data_title:
    neighbor = sorted(data_title[title].items(), key=lambda item:item[1], reverse=True)
    nei_num = len(neighbor)
    if nei_num == 0:
        null_t += 1
        title_nei.setdefault(title, [])
    elif nei_num == 1:
        title_nei.setdefault(title, []).extend([neighbor[0][0]])
    else:
        title_nei.setdefault(title, []).extend([neighbor[0][0], neighbor[1][0]])
print("total title 's num is: ", len(data_title))
print("null title 's num is: ", null_t)
print("Finished: find (two) neighbors.")

with io.open(input_data, 'r', encoding='utf-8') as fa, io.open(output_data, 'w', encoding='utf-8') as fb:
    for i, line1 in enumerate(fa):
        line = line1.strip().split('\t')
        query = line[0]
        title = line[1]
        score = line[2]
        if score in {'4', '5'}:
            score = '1'
        else:
            score = '0'
        qiq1_t, qiq1_q, qiq2_t, qiq2_q, iqi1_q, iqi1_t, iqi2_q, iqi2_t = title, query, title, query, query, title, query, title
        if query in query_nei:
            qiq1_t = query_nei[query][0]
            if qiq1_t in title_nei:
                qiq1_q = title_nei[qiq1_t][0]
                if len(title_nei[qiq1_t]) > 1:
                    qiq2_t = qiq1_t
                    qiq2_q = title_nei[qiq1_t][1]
        if title in title_nei:
            iqi1_q = title_nei[title][0]
            if iqi1_q in query_nei:
                iqi1_t = query_nei[iqi1_q][0]
                if len(query_nei[iqi1_q]) > 1:
                    iqi2_q = iqi1_q
                    iqi2_t = query_nei[iqi1_q][1]
        fb.write(query + '\t' + title + '\t' + score + '\t' + qiq1_t + '\t' + qiq1_q + '\t' + qiq2_t + '\t' + qiq2_q + '\t' + iqi1_q + '\t' + iqi1_t + '\t' + iqi2_q + '\t' + iqi2_t + '\n') 
print("Finished!")




