# -*- coding: utf-8 -*-

"""
    @Time    : 2024/6/15 14:12
    @Author  : Liu Zhaoyang 2023233158
    @File    : utils.py
    @Project : wikipediaMining
"""
from transformers import BertTokenizer, BertModel
import torch
import spacy
import networkx as nx
import warnings
from nltk import pos_tag,word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
warnings.filterwarnings('ignore')



nlp = spacy.load('en_core_web_sm')
bert_path = "/root/data/jupyter/utils/nlp/model_params/bert-base-cased/"
tokenizer = BertTokenizer.from_pretrained(bert_path)



def get_number_of_verbs(sentence:str):
    text = word_tokenize(sentence) #分词
    # print([(word,tag) for word,tag in pos_tag(text) if word.isalpha()]) # 查询标注结果
    return len([(word,tag) for word,tag in pos_tag(text) if 'VB' in tag]) #提取名词和代词

def get_number_of_noun_and_prep(sentence:str):
    text = word_tokenize(sentence) #分词
    # print([(word,tag) for word,tag in pos_tag(text) if word.isalpha()]) # 查询标注结果
    return len([(word,tag) for word,tag in pos_tag(text) if 'NN' in tag or 'PRP' in tag]) #提取名词和代词


def get_graph_of_sentence(doc, style='undirect'):
    """
    输入的参数一定要是spacy返回的doc对象
    doc = nlp(sentence)
    :param doc:
    :return:
    """
    if style == 'undirect':
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    for token in doc:
        if token.dep_ == 'punct' and len(token) == 1:
            continue
        else:
            G.add_node(token.text, tag=token.tag_)

    for token in doc:
        if token.dep_ == 'punct' and len(token) == 1:
            continue
        else:
            node1 = token.text
            node2 = token.head.text
            G.add_edge(node1, node2, tag=token.dep_)
    return G



def get_the_nearest_verb(G, doc, element):
    verbs = []
    for token in doc:
        if token.dep_ == 'punct':
            pass
        if 'V' in token.tag_:
            verbs.append(token)
    verb_dist = []
    for verb in verbs:
        verb_dist.append((verb, len(nx.shortest_path(G, source=str(element), target=str(verb)))))
    verb_dist = sorted(verb_dist, key=lambda x: x[1])
    return verb_dist[0]




def get_model_input_of_sentence(sentence):
    sentence = ' '.join(sentence.split())[:-1].replace('.', ',').replace('\"', '\'') + '.'
    sentence = sentence.replace('\n', '')
    doc = nlp(sentence)
    G = nx.Graph()

    tokens = tokenizer.encode_plus(sentence, padding='max_length', max_length=86, truncation=True)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    sub_dict = {}
    for token in doc:
        if token.dep_ == 'punct' and len(token) == 1:
            continue
        else:
            # print(token.text, token.tag_, token.pos_)
            subwords = tokenizer.tokenize(token.text)
            if len(subwords) == 0:
                continue
            # print(subwords)
            sub_dict[token.text] = subwords[0]
            for subword in subwords:
                token_ids = tokenizer.convert_tokens_to_ids(subword)
                idx = input_ids.index(token_ids)
                G.add_node(subword, id=token_ids, idx=idx)
            for subword in subwords[1:]:
                G.add_edge(subwords[0], subword, tag=token.dep_)

    for token in doc:
        if token.dep_ == 'punct' and len(token) == 1:
            continue
        else:
            node1 = sub_dict[token.text]
            node2 = sub_dict[token.head.text]
            G.add_edge(node1, node2, tag=token.dep_)

    source = []
    target = []
    for u, v, attr in G.edges(data=True):
        source.append(G.nodes[u]['idx'])
        target.append(G.nodes[v]['idx'])
        # print(G.nodes[u]['idx'], G.nodes[v]['idx'])
    edge_index = [source, target]

    node_tokens = []
    for node, attr in G.nodes(data=True):
        node_tokens.append((G.nodes[node]['id'], G.nodes[node]['idx'] - 1))
    node_tokens = [a[0] for a in sorted(node_tokens, key=lambda x: x[1])]

    # return torch.tensor(node_tokens) torch.tensor(input_ids), torch.tensor(attention_mask),
    # return node_tokens

    return torch.tensor(node_tokens), torch.tensor(edge_index)

def get_all_verb(doc):
    verbs = []
    for token in doc:
        if token.dep_ == 'punct':
            pass
        if 'V' in token.tag_:
            verbs.append(token)
    return verbs


def get_attention_mask(sentence, triplet):
    sentence = str(sentence).replace('\n', ' ').replace('\xa0', ' ')
    sentence = ' '.join(sentence.split())[:-1].replace('.', ',').replace('\"', '\'') + '.'
    if 'Heavyweight' in sentence:
        sentence = sentence.replace('Heavyweight', 'Heavy Weight')
    if '[cs]' in sentence:
        sentence = sentence.replace('[cs]', '')
    sentence = sentence.replace('.', '')

    # sentence = sentence.replace('(', '').replace(')', '')
    sentence = re.sub(r'\s+', ' ', sentence)
    if sentence[-1] == ' ':
        sentence = sentence[:-1] + '.'
    else:
        sentence = sentence + '.'

    doc = nlp(sentence)
    G = get_graph_of_sentence(doc)

    trips = []

    for trp in triplet:
        tmp = str(trp).split(' ')
        s = tmp[0].replace('.', '').replace(',', '').replace('\"', '\'').replace('(', '').replace(')', '')
        trips.append(s)

    verbs = get_all_verb(doc)
    sub_nodes = []
    for verb in verbs:
        for tr in trips:
            # try:
            sub_nodes.extend(list(nx.shortest_path(G, source=str(tr), target=str(verb))))
            sub_nodes.extend(list(nx.shortest_path(G, source=str(tr), target=str(verb))))
            sub_nodes.extend(list(nx.shortest_path(G, source=str(tr), target=str(verb))))
    sub_nodes = list(set(sub_nodes))
    atten_words = []
    for word in sentence.split(' '):
        if word in sub_nodes:
            atten_words.append(word)
    for trp in triplet:
        atten_words.append(trp)
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=86)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    atten_words_BPE = []
    for word in atten_words:
        atten_words_BPE.extend(tokenizer.tokenize(word))
    word_ids = [tokenizer.vocab[word] for word in atten_words_BPE]

    word_indices = [torch.where(input_ids == word_id)[1] for word_id in word_ids]
    custom_attention_mask = torch.zeros(attention_mask.size())
    for idx in torch.cat(word_indices):
        custom_attention_mask[:, idx] = 1
        # custom_attention_mask[:, idx+1:] = 0
    custom_attention_mask = custom_attention_mask.type(torch.int)
    return input_ids, custom_attention_mask, attention_mask


def seed_everything(seed=42):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(temp, embedding, label):
    """
    calculate the contrastive loss
    这里的embedding指的是一个batch
    """
    # 首先计算WMD矩阵用来代替余弦相似度矩阵
    # cosine_sim = get_wmd_dist_in_batch(embedding)
    cosine_sim = cosine_similarity(embedding, embedding)
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))

    contrastive_loss = 0

    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss / cosine_sim.shape[0]


def custom_index(custom_mask, edge_index):
    size = edge_index.size()[1]

    mask = custom_mask[0, :size].bool()
    #
    source_list = edge_index[0, :]
    target_list = edge_index[1, :]

    result1 = source_list[mask].unsqueeze(0)
    result2 = target_list[mask].unsqueeze(0)
    # res1 = torch.mul(mask_list, source_list)
    # res2 = torch.mul(mask_list, target_list)

    new_edge_index = torch.cat([result1, result2], dim=0)
    return new_edge_index

