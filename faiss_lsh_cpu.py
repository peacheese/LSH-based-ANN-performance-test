import sys
import time
import math
import faiss
import numpy as np

def cos_dist(query, embedding):
    inner_product, q_mod, e_mod = 0, 0, 0
    for i in range(len(query)):
        inner_product += query[i] * embedding[i]
        q_mod += query[i] * query[i]
        e_mod += embedding[i] * embedding[i]
    return inner_product / (math.sqrt(q_mod) * math.sqrt(e_mod))

if __name__ == '__main__':
    
    dim = 128
    x = int(sys.argv[1])
    k = int(sys.argv[2])
    print('{}M items, Top-K = {}'.format(x, k))
    total_num = x * 1024000

    # np.random.seed(1234)
    corpus = np.random.randint(0, 256, (total_num, dim)).astype('float32')
    query  = np.random.randint(0, 256, (1, dim)).astype('float32')

    lsh_index = faiss.IndexLSH(dim, 4 * dim)
    lsh_index.add(corpus)
    distance, lsh_idx = lsh_index.search(query, k)
    first_idx = lsh_idx[0][0]
    print(cos_dist(corpus[first_idx], query[0]))
    # print(len(query[0]))

    # print(distance)
    acc_index = faiss.IndexFlatIP(dim)
    acc_index.add(corpus)
    distance, acc_idx = acc_index.search(query, k)
    first_idx = acc_idx[0][0]
    print(cos_dist(corpus[first_idx], query[0]))
    
    count = 0
    for item in lsh_idx[0]:
        if item in acc_idx[0]:
            count += 1
    print(count)