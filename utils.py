import json
import torch as T
import csv
from itertools import groupby
import random
import pandas
from TweetNormalizer import *

def configure_input_format(data, labels, tokenizer, train = False):
    new_data, new_labels, random_entities = [], [], []
    for i, sentence in enumerate(data):
        span_label = ''
        types = []
        new_data.append('What are the entities? \n ' + " ".join(sentence) + tokenizer.eos_token)

        #sentence_map[i] = [len(new_data)-1]

        for j, label in enumerate(labels[i]):



            if label != 'O':

                tag, entity_type = label.split('-', 1)
                if tag == 'B':
                    types.append(entity_type)
                    if len(span_label)>0:
                        if span_label[-1] == ' ':
                            span_label = span_label[:-1]
                            span_label += ';'
                if len(span_label)>0:
                    if span_label[-1] == ';':
                        span_label += ' '
                if sentence[j] == '&amp;':
                    span_label += '&amp '
                elif sentence[j] == '&quot;':
                    span_label += '&quot '
                elif sentence[j] == ';':
                    span_label += 'apos'
                else:
                    span_label += sentence[j] + ' '

            else:
                if span_label != '' and span_label[-1] != ';':
                    span_label = span_label[:-1]
                    span_label += ';'
        if not span_label:
            span_label += '[No answer]'
        else:
            if span_label[-1] == ';' or span_label[-1]==' ':
                span_label = span_label[:-1]

        if span_label != '[No answer]':
            entities = list(span_label.split('; '))

        else:
            entities = []

        '''
        print(sentence)
        print(labels[i])
        print(entities)
        print(types)
        print(len(entities))
        print(len(types))
        '''
        assert len(entities) == len(types)
        span_label += tokenizer.eos_token

        new_labels.append(span_label)


        if len(entities)>0 and train:
            for k, entity in enumerate(entities):
                new_data.append('What type is ' + entity +'? \n '+ ' '.join(sentence) + tokenizer.eos_token)
                #sentence_map[i].append(sentence_map[i][-1]+1)
                random_entities.append(entity)
                if 'per' == types[k][:3].lower():
                    new_labels.append('person')
                elif 'loc' == types[k][:3].lower():
                    new_labels.append(('location'))
                elif 'mis' == types[k][:3].lower():
                    new_labels.append('miscellaneous')
                elif 'org' == types[k][:3].lower():
                    new_labels.append('organization')
                else:
                    new_labels.append(types[k].lower())
                new_labels[-1] += tokenizer.eos_token


    if train:
        random.seed(1)
        idx = [i for i in range(len(random_entities))]
        random.shuffle(idx)
        random.seed(2)
        data_idx = [i for i in range(len(data))]
        random.shuffle(data_idx)
        random.shuffle(data_idx)
        idx = idx[:1500]
        data_idx = data_idx[:1500]
        for j, id in enumerate(idx):
            new_data.append('What type is '+ random_entities[id]+'? \n '+ ' '.join(data[data_idx[j]]) + tokenizer.eos_token)
            new_labels.append(tokenizer.eos_token)
        print('Number of sentences in train set: '+str(len(new_data)))
    assert len(new_data) == len(new_labels)

    return new_data, new_labels


def configure_T5_input_format(data, labels, queries, entity_types, tokenizer):
    new_data, new_labels = [], []
    for i, sentence in enumerate(data):
        for type in entity_types:
            span_label = ''
            new_data.append(queries[type]+' \n '+" ".join(sentence) + tokenizer.eos_token)
            for j, label in enumerate(labels[i]):
                if label != 'O':
                    tag, entity_type = label.split('-', 1)
                    if entity_type == type:
                        if len(span_label)>0:
                            if span_label[-1] == ';':
                                span_label+= ' '
                        span_label += sentence[j] + ' '
                else:
                    if span_label != '' and span_label[-1] != ';':
                        span_label = span_label[:-1]
                        span_label+= ';'
            if not span_label:
                span_label += '[No answer]'
            else:
                if span_label[-1] == ';' or span_label[-1] == ' ':
                    span_label = span_label[:-1]
            span_label += tokenizer.eos_token
            new_labels.append(span_label)
    return new_data, new_labels


def make_map(data, labels, queries, entity_types, tokenizer):
    new_data, new_labels = configure_T5_input_format(data, labels, queries, entity_types, tokenizer)
    incr = len(entity_types)
    QA_label_map = []
    for i in range(0, len(new_data), incr):
        QA_labels = new_labels[i:i+incr]
        QA_map = {}
        for j, type in enumerate(entity_types):
            QA_map[type] = QA_labels[j][:-4]
        QA_label_map.append(QA_map)
    return QA_label_map

def make_confusion_matrix(label_maps, pred_maps, query_labels):
    confusion = [[0 for i in range(len(query_labels)+3)] for j in range(len(query_labels)+3)]

    for i in range(len(label_maps)):

        predictions = pred_maps[i]
        labels = label_maps[i]

        fl =0

        for type in query_labels:
            predictions[type] = predictions[type].split('; ')
            labels[type] = labels[type].split('; ')
            if predictions[type] != ['[No answer]'] or labels[type] != ['[No answer]']:
                fl=1

        if fl == 0:
            confusion[-3][-3] +=1
            continue
        for j, type in enumerate(query_labels):

            for z, entity in  enumerate(predictions[type]):
                if entity == '[No answer]':
                    break
                flag=0
                if entity in labels[type]:
                    confusion[j][j] += 1
                    flag=1
                    labels[type].remove(entity)
                    continue

                for k, l_type in enumerate(query_labels):
                    if l_type == type:
                        continue
                    if entity in labels[l_type]:
                        labels[l_type].remove(entity)
                        confusion[k][j] +=1
                        flag=1
                        break

                if flag ==0:
                    confusion[len(query_labels)][j] += 1

                    for k, l_type in enumerate(query_labels):
                        for phrase in labels[l_type]:
                            if entity in phrase and entity != phrase:
                                confusion[len(query_labels)+1][j] +=1

                                flag=1
                                break
                        if flag == 1:
                            break

                    if flag ==0:
                        confusion[len(query_labels)+2][j] += 1

        for j, type in enumerate(query_labels):
            if labels[type] != ['[No answer]']:
                confusion[j][len(query_labels)] += len(labels[type])

                for l, entity in enumerate(labels[type]):
                    flag=0
                    for l_type in query_labels:
                        for phrase in predictions[l_type]:
                            if entity in phrase and entity != phrase:
                                confusion[j][len(query_labels)+1] +=1
                                flag=1

                                break
                        if flag ==1:
                            break
                    if flag ==0:
                        confusion[j][len(query_labels)+2] +=1

    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pandas.DataFrame(confusion, query_labels+['Not found', 'partial', 'at all'], query_labels+['Not found', 'partial', 'at all']))
        print()



def generate_queries(dataset):
    dic, d = {}, {}
    if dataset == 'conll03':
        dic['ORG'] = "Which corporate, governmental or organizational entities are present in the text?"
        dic['PER'] = "Which named person or family entities are present in the text?"
        dic["LOC"] = "Which location or geographical entities are present in the text?"
        dic["MISC"] = "Which miscellaneous entities including events, nationalities, products and works of art are present in the text?"
        labels = ["ORG", "PER", "LOC", "MISC"]
        d["queries"] = dic
        d["entity_types"] = labels

    elif dataset == 'wnut17':

        dic['corporation'] = "Which corporate entities are present in the text?"
        dic['person'] = "Which named person or family entities are present in the text?"
        dic['location'] = "Which location or geographical entities are present in the text?"
        dic['group'] = "Which group entities are present in the text?"
        dic['creative-work'] = "Which creative work entities are present in the text?"
        dic['product'] = "Which product entities are present in the text?"
        labels = ['location', 'product', 'corporation', 'creative-work', 'group', 'person']
        d['queries'] = dic
        d['entity_types'] = labels

    elif dataset == 'wnut16':
        d['entity_types'] = ['company', 'facility', 'geo-loc', 'movie', 'musicartist', 'other',
                             'person', 'product', 'sportsteam', 'tvshow']
        d["queries"] ={}
    with open("Data/queries/"+dataset+"_queries.json", "w") as f:
        json.dump(d, f)

def count_iterations(data, batch_size):
    length = len(data)
    iters = length//batch_size
    if length % batch_size > 0:
        iters += 1
    return iters


def convert_to_bio(tags):
    for i, tag in enumerate(tags):
        if len(tag)>1:
            t, l = tag.split('-')
            if i==0:
                tags[i] = 'B-'+ l
            elif t=='B':
                tags[i] = 'B-' + l
            elif t=='I' and tags[i-1][0]=='O':
                tags[i] = 'B-' + l
            elif t == 'I':
                t1, l1 = tags[i-1].split('-')
                if l1!=l:
                    tags[i] = 'B-'+ l
                else:
                    tags[i] = 'I-' + l

    return tags

def read_dataset(file_location, dataset, delimiter ='\t'):
    if dataset == 'wnut17' or dataset == 'wnut16':
        with open(file_location) as stream:
            reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
            labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
            tokens, labels = zip(*labeled_tokens)
            sentences, labels = [list(t) for t in tokens], [list(l)  for l in labels]
            for i, sentence in enumerate(sentences):
                for j, token in enumerate(sentence):
                    sentences[i][j] = normalizeToken(token)
                    '''
                    if 'https://' in token or 'http://' in token or 'www.' in token:
                        sentences[i][j] = '<URL>'
                    elif '#' in token and len(token) > 1:
                        sentences[i][j] = '<HASH>'
                    elif token[0] == '@' and len(token) > 1:
                        sentences[i][j] = '<USER>'
                    '''
            return sentences, labels
    elif dataset =='conll03':
        sentences, labels = [], []
        with open(file_location, 'r') as f:
            words, tags = [], []
            for line in f:
                if line != '\n':
                    tmp = line.split(' ')
                    word, tag = tmp[0], tmp[-1][:-1]
                    if len(word) > 0 and len(tag) > 0:
                        words.append(normalizeToken(word))
                        tags.append(tag)
                else:
                    if len(words) > 0:
                        assert len(words) == len(tags)

                        sentences.append(words)
                        labels.append(tags)

                        words, tags = [], []

        return sentences, labels



def preprocess(pathname, dataset):
    if pathname[-1]!='/':
        pathname += '/'

    if dataset == 'conll03' or dataset == 'wnut16':
        train_file = 'train.txt'
        test_file  = 'test.txt'
        val_file = 'dev.txt'
    elif dataset == 'wnut17':
        train_file = 'wnut17train.conll'
        test_file = 'emerging.test.annotated'
        val_file = 'emerging.dev.conll'
    else:
        train_file, test_file, val_file = '', '', ''
        assert 'Wrong dataset parameter. Select from wnut17, conll03'


    sentences, labels = read_dataset(pathname+train_file, dataset)
    d = {}
    d['sentences'] = sentences
    d['labels'] = labels
    with open('Data/'+dataset+'_train_data.json', 'w') as f:
        json.dump(d, f)
    sentences, labels = read_dataset(pathname+val_file, dataset)
    d = {}
    d['sentences'] = sentences
    d['labels'] = labels
    with open('Data/'+dataset+'_val_data.json', 'w') as f:
        json.dump(d, f)
    sentences, labels = read_dataset(pathname+test_file, dataset)
    d = {}
    d['sentences'] = sentences
    d['labels'] = labels
    with open('Data/'+dataset+'_test_data.json', 'w') as f:
        json.dump(d, f)


def compute_F1(tp, pred_len, gold_len):
    prec = tp / pred_len if pred_len > 0 else 0
    rec = tp / gold_len if gold_len > 0 else 0
    F1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return prec, rec, F1

def find_index(entity_split, text):
    j, flag, ind=0, -1, -1

    for i, token in enumerate(text):

        if token == entity_split[j]:
            if flag == -1:
                ind = i
                flag = 1
            j+=1
            if j == len(entity_split):
                return ind
        else:
            if flag != -1:
                flag = -1
                ind = -1
                j =0
    return ind



def sequence_labeller(text, pred_map):
    pred_labels = ['O' for i in range(len(text))]
    tmp = text
    for entity_type in pred_map:
        if pred_map[entity_type] != '[No answer]':
            entities = list(pred_map[entity_type].split('; '))
            j=0
            for entity in entities:
                if entity in ' '.join(text):
                    entity_split = [word for word in entity.split(' ')]

                    ind = find_index(entity_split, text)
                    if ind == -1:
                        continue


                    if pred_labels[ind] == 'O':
                        pred_labels[ind] = 'B-' + entity_type
                        if len(entity_split)>1:
                            ind+=1
                            for _ in range(len(entity_split)-1):
                                pred_labels[ind] = 'I-'+entity_type
                                ind+=1


    return pred_labels


def generate_eval_file(out_file_name):


    with open(out_file_name+'.json', 'r') as f:
        data = json.load(f)
    test_text = data['test_text']
    test_labels = data['test_labels']
    test_pred_labels = data['test_pred_labels']

    output_list = []
    for i, text in enumerate(test_text):

        for j, token in enumerate(text):

            output_list.append(token+'\t_\t'+test_labels[i][j]+'\t'+test_pred_labels[i][j])
        output_list.append('\n')

    with open(out_file_name+'.txt', 'w') as f:
        for item in output_list:
            f.write('%s\n' % item)

def generate_misclassifications_file(orig_text, test_pred_labels, orig_labels,out_file_name):

    with open(out_file_name+'_misclassified.txt', 'w') as f:
        for i in range(len(orig_text)):
            if test_pred_labels[i] != orig_labels[i]:
                f.write(str(' '.join(orig_text[i]))+'\n'+str(' '.join(test_pred_labels[i]))+'\n'+str(' '.join(orig_labels[i]))+'\n\n')


def write_file(filename, dataset, delimiter='\t'):
    """dataset is a list of tweets where each token can be a tuple of n elements"""
    with open(filename, '+w') as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')
        for tweet in dataset:
            writer.writerow(list(tweet))


def generate_wnut17_eval_file(model_size, dataset, val = False, location = 'out/'):
    if val:
        out_file_name = location+model_size+'_'+dataset+'_val'
    else:
        out_file_name = location+model_size+'_'+dataset+'_test'

    with open(out_file_name+'.json', 'r') as f:
        data = json.load(f)

    test_pred_labels = data['test_pred_labels']
    test_text = data['test_text']

    dataset = []
    for i, tweet in enumerate(test_text):
        tweet_data = list(zip(tweet, test_pred_labels[i]))
        dataset += tweet_data + [()]
    write_file(out_file_name + '_prediction.tsv', dataset)


    '''
    len_pred= 0
    for i in test_pred_labels:
        len_pred += len(i)
    len_text = 0
    for i in test_text:
        len_text += len(i)
    print(len_text, len_pred)
    
    output_list = []
    for i, text in enumerate(test_text):
        for j, token in enumerate(text):
            output_list.append(token+'\t'+test_pred_labels[i][j])
        output_list.append('\n')
    with open(out_file_name+'_prediction.tsv', 'w') as f:
        for item in output_list:
            f.write('%s\n' % item)
    '''





'''

def evaulate_predictions(predictions, targets):

    assert len(predictions) == len(targets)

    tp, gold_len, pred_len = 0,0,0
    for i, labels in enumerate(targets):

        tmp_labels = labels.split(', ')
        tmp_preds = predictions[i].split(', ')

        for label in tmp_labels:
            if label != '[No answer]':
                gold_len += 1

        for pred in tmp_preds:
            if pred != '[No answer]':
                pred_len += 1

        for j, label in enumerate(tmp_labels):
            try:
                if label.lower() == tmp_preds[j].lower() and label != '[No answer]':
                    tp+=1
                    #print('match!! = '+label)
            except IndexError:
                break
    return tp, pred_len, gold_len
    
    
        prev_type = ''
        phrase = ''
        for a, token in enumerate(text):

            if test_labels[j][a] != 'O':
                tag, type = test_labels[j][a].split('-')

                if tag == 'B':
                    if phrase != '':
                        pred_label_type = search_phrase(phrase, QA_pred_map)
                        if pred_label_type == prev_type:
                            tp+=1
                            print('match found! = '+phrase)
                            pred_labels.append('B-'+pred_label_type)
                        else:
                            pred_labels.append('O')

                    phrase = token
                    prev_type = type
                else:
                    phrase += ' '+ token

            else:

                if phrase != '':
                    pred_label_type = search_phrase(phrase, QA_pred_map)
                    if pred_label_type == prev_type:
                        tp += 1
                        print('match found! = ' + phrase)
                        pred_labels.append('B-' + pred_label_type)
                        for _ in range(len(list(phrase.split(' ')))-1):
                            pred_labels.append('I-'+pred_label_type)
                    else:
                        for _ in range(len(list(phrase.split(' ')))):
                            pred_labels.append('O')
                    phrase = ''
                pred_labels.append('O')
                prev_type = ''
        print(pred_labels, len(pred_labels))
        print(test_labels[j], len(test_labels[j]))

        assert len(test_labels) != len(pred_labels)
        test_pred_labels.append(pred_labels)


    print(tp, gold_len, pred_len, compute_F1(tp, pred_len, gold_len))



def search_phrase(phrase, label_map):
    for label_type in label_map:
        if phrase in label_map[label_type]:
            return label_type
    return False


'''
