import numpy as np
import random
import re
import copy


def batcher(sample_tuples,
            tokenizer,
            batch_size=32, bucket_size_factor=1,
            sort=True, shuffle=True,
            SEED=None):

    if SEED is not None:
        random.seed(SEED)

    data1 = sample_tuples[0]
    data_len = len(data1)



    #for i in range(data_len):
    '''
    print(sample_tuples[0][i])
    print(len(sample_tuples[0][i]))
    print(len(sample_tuples[1][i]))
    print(sample_tuples[0][i], sample_tuples[1][i])
    '''
    #assert len(sample_tuples[0][i]) == len(sample_tuples[-1][i])

    def reorder(samples, idx):
        return [samples[i] for i in idx]

    def reorder_all(sample_tuples, idx):
        return [reorder(samples, idx) for samples in sample_tuples]

    if shuffle:
        random_idx = [i for i in range(data_len)]
        random.shuffle(random_idx)
        shuffled_sample_tuples = reorder_all(sample_tuples, random_idx)
    else:
        shuffled_sample_tuples = sample_tuples
    if sort:
        #data1 = shuffled_sample_tuples[sort_by_idx]
        true_seq_lens = [len(sample) for sample in shuffled_sample_tuples[0]]
        sorted_idx = np.flip(np.argsort(true_seq_lens), 0)
        sorted_sample_tuples = reorder_all(shuffled_sample_tuples, sorted_idx)
        '''
        print(len(sorted_sample_tuples[0]))
        print(len(sorted_sample_tuples[1]))
        print(sorted_idx)
        print(sorted_sample_tuples[0][10001])
        print(sorted_sample_tuples[1][10001])
        print(sorted_sample_tuples[0][10000])
        print(sorted_sample_tuples[1][10000])
        print()
        '''
    else:
        sorted_sample_tuples = shuffled_sample_tuples


    bucket_size = bucket_size_factor*batch_size

    c = 0
    buckets = []
    while c < data_len:

        start = c
        end = c+bucket_size

        if end > data_len:
            end = data_len

        bucket = [samples[start:end] for samples in sorted_sample_tuples]

        buckets.append(bucket)

        c = end

    if shuffle:
        random.shuffle(buckets)

    def max_len_in_span(samples):
        if isinstance(samples[0], str):
            return max([len(sample) for sample in samples])
        else:
            return -1


    for bucket in buckets:


        if shuffle:
            random_idx = [i for i in range(len(bucket[0]))]
            random.shuffle(random_idx)
            bucket = reorder_all(bucket, random_idx)
        batch_max_len = max_len_in_span(bucket[0])
        tokenized_inputs = tokenizer.batch_encode_plus(bucket[0], max_length=batch_max_len,
                                                       pad_to_max_length=True)

        tokenized_labels = tokenizer.batch_encode_plus(bucket[1], max_length=max_len_in_span(bucket[1]),
                                                       pad_to_max_length=True)

        yield tokenized_inputs, tokenized_labels


