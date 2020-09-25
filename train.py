from utils import *

import torch as T
#import transformers
from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.tokenization_t5 import T5Tokenizer
#from transformers import T5Tokenizer

import argparse
import random
from SM3 import SM3
import math
from tqdm import tqdm
from batcher import batcher
import numpy as np
from conlleval import *
import logging

logging.basicConfig(level=logging.ERROR)


checkpoint_path = 'checkpoint/'
out_location = 'out/'


parser = argparse.ArgumentParser()
parser.add_argument(
        "--learning_rate",
        default=0.1,
        type=float,
        required=False,
    )
parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        required=False,
    )
parser.add_argument(
        "--total_batch_size",
        default=128,
        type=int,
        required=False,
    )
parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        required=False,
    )
parser.add_argument(
        "--dim",
        default=768,
        type=int,
        required=False,
    )
parser.add_argument(
        "--dataset",
        default='conll03',
        type=str,
        required=False,
    )
parser.add_argument(
        "--seed",
        default=5,
        type=int,
        required=False,
    )
parser.add_argument(
        "--model_size",
        default='t5-base',
        type=str,
        required=False,
    )
parser.add_argument(
        "--weight_location",
        type=str,
        required=False,
    )
parser.add_argument(
        "--load_model_path",
        default='',
        type=str,
        required=False,
    )
parser.add_argument(
        "--train",
        default=False,
        action="store_true"
    )
parser.add_argument(
        "--val",
        default=False,
        action="store_true"
    )
parser.add_argument(
        "--test",
        default=False,
        action="store_true"
    )
parser.add_argument(
    '--preprocess',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--generate_queries',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--orig_t5',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--format2',
    default=False,
    action='store_true'
)


args = parser.parse_args()
preprocess_data = args.preprocess
generate_queries_json = args.generate_queries
TRAIN = args.train
VAL = args.val
TEST = args.test
load_model_path = args.load_model_path
model_size = args.model_size
weight_location = args.weight_location
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
total_batch_size = args.total_batch_size
num_epochs = args.num_epochs
transformer_dim = args.dim
dataset = args.dataset
SEED = args.seed
orig_t5 = args.orig_t5
warmup_steps = 1000
format2 = args.format2
name, size_ = model_size.split('-')



if orig_t5:

    if weight_location:
        model = T5ForConditionalGeneration.from_pretrained(weight_location)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_size)
    model_name = name + '-' + size_
    tokenizer = T5Tokenizer.from_pretrained(model_size)
else:
    model = T5ForConditionalGeneration.from_pretrained(weight_location)
    model_name = 'unifiedQA-' + size_
    tokenizer = T5Tokenizer.from_pretrained(model_size)
#config = T5Config.from_pretrained(model_size)




if preprocess_data:
    preprocess('Data/'+dataset, dataset)
if generate_queries_json:
    generate_queries(dataset)


with open('Data/'+dataset+'_train_data.json', 'r') as f:
    train_data = json.load(f)
with open('Data/'+dataset+'_val_data.json', 'r') as f:
    val_data = json.load(f)
with open('Data/'+dataset+'_test_data.json', 'r') as f:
    test_data = json.load(f)

with open('Data/queries/'+dataset+'_queries.json', 'r') as f:
    query_data = json.load(f)

train_text = train_data['sentences']
train_labels = train_data['labels']
val_text = val_data['sentences']
val_labels = val_data['labels']
test_text = test_data['sentences']
test_labels = test_data['labels']

query_labels = query_data['entity_types']
queries = query_data['queries']


if format2:
    train_text_format, train_labels_format = configure_input_format(train_text, train_labels, tokenizer, train=True)
    val_text_format, val_labels_format = configure_input_format(val_text, val_labels, tokenizer, train=False)
    test_text_format, test_labels_format = configure_input_format(test_text, test_labels, tokenizer, train=False)
else:
    train_text_format, train_labels_format = configure_T5_input_format(train_text, train_labels, queries, query_labels,
                                                                       tokenizer)
    val_text_format, val_labels_format = configure_T5_input_format(val_text, val_labels, queries, query_labels,
                                                                   tokenizer)
    test_text_format, test_labels_format = configure_T5_input_format(test_text, test_labels, queries, query_labels,
                                                                     tokenizer)

random.seed(SEED)
idx = [i for i in range(len(train_text_format))]
random.shuffle(idx)
train_text_format = [train_text_format[i] for i in idx]
train_labels_format = [train_labels_format[i] for i in idx]




#val_labels_format_map = make_map(val_text, val_labels, queries, query_labels, tokenizer)
#test_labels_format_map = make_map(test_text, test_labels, queries, query_labels, tokenizer)


'''

train_text= train_text[:50]
train_labels = train_labels[:50]
train_text_format = train_text_format[:50]
train_labels_format = train_labels_format[:50]

val_text = val_text[:50]
val_labels = val_labels[:50]
val_text_format = val_text_format[:50]
val_labels_format = val_labels_format[:50]

test_text = test_text[:100]
test_labels = test_labels[:100]
test_text_format = test_text_format[:100]
test_labels_format = test_labels_format[:100]

print(len(train_text))
print(len(train_labels))
print(len(train_text_format))
print(len(train_labels_format))

print(len(val_text))
print(len(val_labels))
print(len(val_text_format))
print(len(val_labels_format))
'''


assert len(train_text_format) == len(train_labels_format)
assert len(val_text_format) == len(val_labels_format)


train_iterations = count_iterations(train_text_format, train_batch_size)
val_iterations = count_iterations(val_text_format, train_batch_size)
test_iterations = count_iterations(test_text_format, train_batch_size)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

model.to(device)


def load_model(load_model_path, model, optimizer, scheduler):
    checkpoint = T.load(load_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['past epoch'] 
    train_losses = checkpoint['train_losses']
    val_F1s = checkpoint['val_F1s']

    print('\nRESTORATION COMPLETE\n')
    return model, optimizer, scheduler,epoch, train_losses, val_F1s


def run_epochs(model, checkpoint_path ):
    global num_epochs, TRAIN, TEST, VAL

    optimizer = SM3(model.parameters(), lr=learning_rate)
    lr_lambda = lambda step: min(learning_rate, (step / warmup_steps))
    scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    val_F1s = []


    best_val_F1 = -math.inf
    impatience =0

    epoch =0
    load_epoch =0
    load_epoch =0
    optimizer.zero_grad()

    if load_model_path:
        model, optimizer, scheduler, load_epoch, train_losses, val_F1s= load_model(load_model_path,
                                                                                       model, optimizer, scheduler)
        load_epoch += 1
    for epoch in range(load_epoch, num_epochs):

        if TRAIN:
            train_loss = run_batches([train_text_format, train_labels_format],
                                                epoch, model, optimizer, scheduler, train_iterations,
                                                train=True, val = False, test = False, desc='Train batch')
            train_losses.append(train_loss)
            print("\nEpoch:{:3d}, ".format(epoch+1) + "Mean train loss: {:3f}\n".format(train_loss))


            T.save({
                'past epoch': epoch,
                'train_losses': train_losses,
                'val_F1s': val_F1s,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path +model_name +'_finetune_' + dataset+'_'+str(epoch) + ".pt")

            print("Checkpoint Created!\n\n")

        if VAL:

            val_F1 = run_batches([val_text_format, val_labels_format],
                                    epoch, model, optimizer, scheduler, val_iterations,
                                    train=False, val = True, test = False, desc='Validation batch')
            val_F1s.append(val_F1)
            impatience+=1
            print("\nValidation F1: {:3f}\n".format(val_F1))



            if not TRAIN :
                exit()
            else:
                if val_F1> best_val_F1:
                    impatience=0
                    best_val_F1 = val_F1
        if not TRAIN and not VAL:
            break
        print('\nImpatience Level:{:3d}\n'.format(impatience))
        if impatience>3:
            break

    if TEST:

        test_F1 = run_batches([test_text_format, test_labels_format],
                                epoch, model, optimizer, scheduler, test_iterations,
                                train=False, val = False, test = True,desc='Test batch')

        print("\nTest F1: {:3f}\n".format(test_F1))
        if not TRAIN :
            exit()



    if TRAIN:
        if format2:
            tmp = 'final_format2'
        else:
            tmp = 'final_format1'
        T.save({
            'past epoch': epoch,
            'train_losses': train_losses,
            'val_F1s': val_F1s,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path +model_name +'_finetune_' + dataset+'_'+str(epoch)+'_'+tmp+ ".pt")
        print("Final checkpoint created!")


    return

def run_batches(text, epoch, model, optimizer, scheduler, iterations, train=True,
                val = False, test = False, desc='Batch'):
    losses = []
    display_step = 100
    grad_step = total_batch_size // train_batch_size

    predictions = []
    pred_probabilities = []
    pred_token_ids = []
    softmax = T.nn.Softmax(dim=1)


    with tqdm(total = iterations, desc=desc, position=0) as bar:

        i=0
        for tokenized_batch, tokenized_batch_labels\
                in batcher(text, tokenizer=tokenizer,batch_size=train_batch_size, sort=False, shuffle=train):


            batch = T.tensor(tokenized_batch['input_ids']).long().to(device)
            batch_mask = T.tensor(tokenized_batch['attention_mask']).long().to(device)

            batch_labels = T.tensor(tokenized_batch_labels['input_ids']).long().to(device)
            batch_labels_mask = T.tensor(tokenized_batch_labels['attention_mask']).long().to(device)

            if train:

                model.train()
                outputs = model(input_ids = batch,
                                attention_mask = batch_mask,
                                decoder_attention_mask = batch_labels_mask,
                                lm_labels = batch_labels)
                loss, prediction_scores = outputs[:2]

                losses.append(loss.item())
            else:

                with T.no_grad():
                    model.eval()

                    output_ids, logit_probs, logit_token_ids = model.generate(input_ids=batch, attention_mask=batch_mask, max_length=500)

                    predictions += [tokenizer.decode(ids) for ids in output_ids]

                    pred_probabilities.extend((np.transpose(logit_probs)))
                    pred_token_ids.extend((np.transpose(logit_token_ids)))



            if train:
                loss /= grad_step
                loss.backward()
                if (i+1) % grad_step ==0:

                    T.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if i % display_step ==0:
                    bar.write(
                        "Epoch: {:3d}, Iteration: {:5d},".format(epoch+1, i) + " Loss: {:3f}".format(losses[-1]))


            i+=1
            bar.update(1)

    if val:
        if format2:
            F1 = evaluation_format2(model, val_text, val_labels,predictions , pred_probabilities, pred_token_ids, val)
        else:
            F1 = evaluation_format1(val_text, val_labels, predictions, val)
        return F1
    elif test:

        if format2:
            F1 = evaluation_format2(model, test_text, test_labels, predictions, pred_probabilities, pred_token_ids,  val)
        else:
            F1 = evaluation_format1(test_text, test_labels, predictions, val)
        return F1
    else:
        return np.mean(losses)



def evaluation_format2(model, orig_text, orig_labels, predictions, pred_probabilities, pred_token_ids, val = False):

    global tokenizer, query_labels, model_size, dataset, train_batch_size, device, out_location

    print("\nEntity type prediction ongoing.....\n")

    label_map = {'person':'PER', 'location':'LOC', 'miscellaneous':'MISC', 'organization':'ORG'}
    label_map_inv= {label_map[k]:k for k in label_map}

    #incr = len(query_labels)
    i=0
    entity_predictions = []
    #print(sentence_map)
    test_pred_labels = []


    entity_probs = []
    entity_type_prob_map = defaultdict(list)

    if val:
        desc = 'Val entity typing '
    else:
        desc = 'Test entity typing'
    with tqdm(total=len(orig_text), desc=desc, position=0) as bar:

        for j, text in enumerate(orig_text):

            #QA_sentences = test_text_format[i:i+incr]
            #QA_labels = test_labels_format[i:i + incr]

            QA_pred_map = {}
            if dataset=='conll03':
                for label in query_labels:
                    QA_pred_map[label] = ''
            else:
                for label in query_labels:
                    QA_pred_map[label] = ''


            tmp_probs = []

            probability = 1
            for k, token_id in enumerate(pred_token_ids[j]):
                if token_id == 117:
                    tmp_probs.append(probability)
                elif token_id ==1:
                    tmp_probs.append(probability)
                    break
                else:
                    probability *= pred_probabilities[j][k]

            preds = predictions[j]
            entity_preds = list(preds.split('; '))

            entity_probs.append(tmp_probs)
            '''
            print(' '.join(text))
            print(preds)
            print(pred_token_ids[j])
            print(pred_probabilities[j])
            print(tmp_probs)
            print()
            '''


            QA_questions = []
            entity_type_pred = []

            if entity_preds[0]!= '[No answer]':
                for entity in entity_preds:
                    QA_questions.append('What type is ' + entity +'? \n '+ ' '.join(text) + tokenizer.eos_token)

                for tokenized_batch, tokenized_batch_labels in batcher(
                        [QA_questions, ['' for _ in range(len(QA_questions))]],
                        tokenizer=tokenizer, batch_size=train_batch_size, sort=False, shuffle=False):
                    batch = T.tensor(tokenized_batch['input_ids']).long().to(device)
                    batch_mask = T.tensor(tokenized_batch['attention_mask']).long().to(device)

                    with T.no_grad():
                        output_ids, logit_probs, logit_token_ids = model.generate(input_ids=batch, attention_mask=batch_mask, max_length=150)

                        entity_type_pred += [tokenizer.decode(ids) for ids in output_ids]

                        #pred_probabilities.extend((np.transpose(logit_probs)))
                        #pred_token_ids.extend((np.transpose(logit_token_ids)))
                        logit_probs = np.transpose(logit_probs)
                        for l, token_ids in enumerate(np.transpose(logit_token_ids)):
                            probability =1
                            for m, token_id in enumerate(token_ids):
                                if token_id ==117:
                                    assert token_id == 118 #throw error if semi-colon is present in entity type prediction
                                elif token_id ==1:
                                    entity_type_prob_map[j].append([tokenizer.decode(output_ids[l]), probability])
                                    break
                                else:
                                    probability *= logit_probs[l][m]
                #print(predictions[j])
                #print(entity_type_prob_map[j])
                #print()
                for k, entity in enumerate(entity_preds):
                    try:
                        if dataset == 'conll03':
                            entity_type = label_map[entity_type_pred[k]]
                        else:
                            entity_type = entity_type_pred[k]
                        if QA_pred_map[entity_type] == '':
                            QA_pred_map[entity_type] += entity
                        else:
                            QA_pred_map[entity_type] += '; ' + entity
                    except:
                        continue

            else:
                for label in query_labels:
                    QA_pred_map[label] = '[No answer]'

                entity_type_pred += ['[No answer]']

            assert len(entity_preds) == len(entity_type_pred)

            for key in list(QA_pred_map.keys()):
                if QA_pred_map[key] == '':
                    QA_pred_map[key] += '[No answer]'

            test_pred_labels.append(sequence_labeller(text, QA_pred_map))
            entity_predictions.append(QA_pred_map)
            bar.update(1)
            #print(' '.join(text))
            #print(repr(test_text_format[j]))
            #print(test_labels_format[j])
            #print(test_labels_format_map[j])
            #print(entity_preds)
            #print(entity_type_pred)
            #print(QA_pred_map)
            #print(test_pred_labels[-1])
            #print(orig_labels[j])
            #print()


    d={}
    d['test_pred_labels'] = test_pred_labels
    d['test_text'] = orig_text
    d['test_labels'] = orig_labels
    d['entity_probs'] = entity_probs
    d['entity_type_probs'] = entity_type_prob_map

    file_rows=[]
    for i in range(len(orig_text)):
        if test_pred_labels[i] != orig_labels[i]:
            file_rows += [orig_text[i], test_pred_labels[i], orig_labels[i]]

            entity_preds = predictions[i].split('; ')

            #tmp += [' ' for _ in range(len(orig_text[i]) - len(entity_preds))]
            file_rows += [entity_preds]
            file_rows += [[str(round(prob, 4)) for prob in entity_probs[i]]]
            #print(file_rows[-1])
            if i in entity_type_prob_map:
                tmp = [[], []]
                for j in range(len(entity_type_prob_map[i])):
                    tmp[0].append(entity_type_prob_map[i][j][0])
                    tmp[1].append(str(round(entity_type_prob_map[i][j][1], 4)))

                file_rows += tmp

            file_rows += [' ']



    if val:
        out_file_name = out_location+model_size+'_'+dataset+'_format2_val'
    else:
        out_file_name = out_location+model_size+'_'+dataset+'_format2_test'
        with open('out/t5-base_conll03_probs_misclassified.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(file_rows)



    generate_misclassifications_file(orig_text, test_pred_labels, orig_labels, out_file_name)

    #if TEST:
        #make_confusion_matrix(test_labels_format_map, entity_predictions, query_labels)

    with open(out_file_name+'.json', 'w') as f:
        json.dump(d, f)

    generate_eval_file(out_file_name)

    with open(out_file_name+'.txt', 'r') as f:
        file_data = f.readlines()

    res = evaluate_conll_file(file_data)
    return res[2]


def evaluation_format1(orig_text, orig_labels, predictions, val=False):
    global tokenizer, query_labels, model_size, dataset, train_batch_size, device, out_location

    incr = len(query_labels)
    i = 0
    entity_predictions = []
    test_pred_labels = []
    for j, text in enumerate(orig_text):
        QA_sentences = test_text_format[i:i + incr]
        # QA_labels = test_labels_format[i:i + incr]
        QA_pred = predictions[i:i + incr]

        QA_pred_map = {}
        for k, ids in enumerate(QA_pred):
            QA_pred_map[query_labels[k]] = QA_pred[k]

        test_pred_labels.append(sequence_labeller(text, QA_pred_map))
        i += incr
        entity_predictions.append(QA_pred_map)
        '''
        print(' '.join(text))
        print(QA_pred_map)

        print(test_pred_labels[-1])
        print(orig_labels[j])
        print()
        '''
    d = {}
    d['test_pred_labels'] = test_pred_labels
    d['test_text'] = orig_text
    d['test_labels'] = orig_labels

    if val:
        out_file_name = out_location + model_size + '_' + dataset + '_format1_val'
    else:
        out_file_name = out_location + model_size + '_' + dataset + '_format1_test'

    generate_misclassifications_file(orig_text, test_pred_labels, orig_labels, out_file_name)
    #if TEST:
        #make_confusion_matrix(test_labels_format_map, entity_predictions, query_labels)

    with open(out_file_name + '.json', 'w') as f:
        json.dump(d, f)

    generate_eval_file(out_file_name)

    with open(out_file_name + '.txt', 'r') as f:
        file_data = f.readlines()

    res = evaluate_conll_file(file_data)
    return res[2]





run_epochs(model, checkpoint_path)




















