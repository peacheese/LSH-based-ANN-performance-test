#include <algorithm>
#include <iostream>
#include <string.h>
#include <cstring>
#include <stdio.h>
#include <cstdlib>
#include <random>
#include <vector>
#include <queue>
#include <time.h>
#include <set>

// System Args
#define BANK_WIDTH              32                  // 32B width Bank
#define ITEM_NUM                128                 // 128dim Embedding
#define SIZE_EACH_ITEM          1                   // 1B each (int_8)
#define EMBEDDINGS_EACH_BANK    256000               // Vector num in 1 Bank
#define MAX_K                   1024                // Top-1024 at most
// LSH Args
#define K                       4                   // K-bit Flags
#define L                       1                   // Hash Table num
#define T                       1                   // Hamming distance 

typedef unsigned char int_8;

struct hashTable {
    std::vector<int> bucket[(1 << K)];
};

struct Embedding {
    int_8 item[128];
};

// This function generates simple hash code
int hash_code_gen(int_8 element, int func_id) {
    return func_id == 0 ? (element * 10000019 + 99988753) :
           func_id == 1 ? (element * 99998111 + 10010069) :
           func_id == 2 ? (element * 10019813 + 99989083) :
                          (element * 100000801 + 10003121);
}

// This function recieve an embedded vector, returns its index in table L
int hash_locate(int_8* embedding, int table_idx) {
    int sign[K] = {0};
    for (int i = 0; i < ITEM_NUM; i++) {
        int hash_code = hash_code_gen(embedding[i], table_idx) % (1 << K);
        for (int j = 0; j < K; j++) {
            sign[j] += ((hash_code & 1) == 1 ? 1 : -1);
            hash_code = (hash_code >> 1);
        }
    }
    for (int i = 0; i < K; i++) {
        sign[i] = sign[i] > 0 ? 1 : 0;
    }
    int index = 0;
    for (int k = 0; k < K; k++) {
        index += sign[k] * (1 << k);
    }
    return index;
}

Embedding Corpus[EMBEDDINGS_EACH_BANK * 4];

struct EmbeddingInfo {
    double score;
    int index;
    EmbeddingInfo(double _s, int _i) {
        score = _s;
        index = _i;
    }
};

struct cmpEmbeddingInfo{
    bool operator() (EmbeddingInfo& a, EmbeddingInfo& b) {
        return a.score > b.score;
    }
};

double sim_calc(int_8* query, Embedding element) {
    double inner_product = 0, q_mod = 0, e_mod = 0;
    for (int i = 0; i < ITEM_NUM; i++) {
        inner_product += (double)(query[i]) * (double)(element.item[i]);
        q_mod += (double)(query[i]) * (double)(query[i]);
        e_mod += (double)(element.item[i]) * (double)(element.item[i]);
    }
    return inner_product / (sqrt(q_mod) * sqrt(e_mod));
}

int main() {

    srand((unsigned)time(NULL));
    // L hash tables
    hashTable table[L];

    for (int embedding_channel = 0; embedding_channel < 4; embedding_channel ++) {
        // Generate random embeddings one by one
        for (int i = 0; i < EMBEDDINGS_EACH_BANK; i++) {
            int_8 embedding[ITEM_NUM]; 
            for (int j = 0; j < ITEM_NUM; j++) {
                embedding[j] = (int_8) (rand() % 256);
            }   
            // Fill Corpus
            int embedding_idx = i + embedding_channel * EMBEDDINGS_EACH_BANK; 
            for (int j = 0; j < ITEM_NUM; j++) {
                Corpus[embedding_idx].item[j] = embedding[j];
            }
            // Once generated one embedding, locate its index in L tables
            for (int j = 0; j < L; j++) {
                int index = hash_locate(embedding, j);
                table[j].bucket[index].push_back(embedding_idx);
            }
        }
    }

    std::cout << "Table information" << std::endl;
    for (int j = 0; j < L; j++) {
        std::cout << "Table " << j << " : ";
        for (int k = 0; k < (1 << K); k++) {
            std::cout << "(index=" << k << ")" << table[j].bucket[k].size() << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Start Query Simulation" << std::endl << std::endl;
    // Random Query
    for (int i = 0; i < 1; i++) {

        int_8 query[ITEM_NUM];
        for (int j = 0; j < ITEM_NUM; j++) {
            query[j] = (int_8) (rand() % 256);
        }   

        // Union
        // std::vector<int> indexs;
        // for (int j = 0; j < L; j++) {
        //     int index = hash_locate(query, j);
        //     std::cout << "Hit table " << j << "(index=" << index << ") " << table[j].bucket[index].size() << std::endl;
        //     indexs.insert(indexs.end(), table[j].bucket[index].begin(), table[j].bucket[index].end());
        //     for (int k = 0; k < K; k++) {
        //         // Change k-th bit of index
        //         int neighbor = index ^ (1 << k);
        //         indexs.insert(indexs.end(), table[j].bucket[neighbor].begin(), table[j].bucket[neighbor].end());
        //     }
        // }
        // std::set<int> indexs_set(indexs.begin(), indexs.end());
        // std::cout << "Hit total embeddings " << s.size() << std::endl;

        // Intersection
        std::set<int> indexs_set;
        for (int j = 0; j < L; j++) {
            int index = hash_locate(query, j);
            std::cout << "Hit table " << j << "(index=" << index << ") " << table[j].bucket[index].size() << std::endl;
            // Find all buckets in hamming distance L(L = 1)
            std::vector<int> indexs;
            indexs.insert(indexs.end(), table[j].bucket[index].begin(), table[j].bucket[index].end());
            for (int k = 0; k < K; k++) {
                // Change k-th bit of index
                int neighbor = index ^ (1 << k);
                indexs.insert(indexs.end(), table[j].bucket[neighbor].begin(), table[j].bucket[neighbor].end());
            }
            std::cout << "Hit table " << j << "(index=" << index << "'s Neighbors) " << indexs.size() << std::endl;
            std::set<int> tmp(indexs.begin(), indexs.end());
            if (j == 0)
                indexs_set = tmp;
            else {
                std::set<int> intersection;
                std::set_intersection(indexs_set.begin(), indexs_set.end(), tmp.begin(), tmp.end(), inserter(intersection, intersection.begin())); 
                indexs_set = intersection;
            }
        }

        std::cout << std::endl;
        std::cout << "Hit Total embeddings " << indexs_set.size() << std::endl << std::endl;
        // Now Choose Top 1024 of Corpus
        std::priority_queue <EmbeddingInfo, std::vector<EmbeddingInfo>, cmpEmbeddingInfo> corpus_top;
        for (int j = 0; j < EMBEDDINGS_EACH_BANK * 4; j++) {
            double score = sim_calc(query, Corpus[j]);
            if (corpus_top.size() < MAX_K){ 
                corpus_top.push(EmbeddingInfo(score, j));
            } else {
                if(score < corpus_top.top().score) {
                    continue;
                } else {
                    corpus_top.pop(); 
                    corpus_top.push(EmbeddingInfo(score, j));
                }
            }
        }
        // Now Choose Top 1024 of IndexsSet
        std::priority_queue <EmbeddingInfo, std::vector<EmbeddingInfo>, cmpEmbeddingInfo> indexs_top;
        for (auto it = indexs_set.cbegin(); it != indexs_set.cend(); it++) {
            double score = sim_calc(query, Corpus[*it]);
            if (indexs_top.size() < MAX_K){ 
                indexs_top.push(EmbeddingInfo(score, *it));
            } else {
                if(score < indexs_top.top().score) {
                    continue;
                } else {
                    indexs_top.pop();
                    indexs_top.push(EmbeddingInfo(score, *it));
                }
            }
        }

        // for (int j = 0; j < ITEM_NUM; j++) 
        //     std::cout << (int)query[j] << " ";
        // std::cout << std::endl;

        // for (int j = 0; j < ITEM_NUM; j++)
        //     std::cout << (int)Corpus[corpus_top.top().index].item[j] << " ";
        
        std::cout << std::endl;
        std::cout << sim_calc(query, Corpus[corpus_top.top().index]) << std::endl;

        std::cout << std::endl;
        std::cout << sim_calc(query, Corpus[indexs_top.top().index]) << std::endl;
        
        std::set<int> corpus_, indexs_;
        int count = 0;

        while(!corpus_top.empty()) {
            corpus_.insert(corpus_top.top().index);
            // std::cout << corpus_top.top().score << " ";
            corpus_top.pop();
        }
        std::cout << std::endl;
        while(!indexs_top.empty()) {
            // std::cout << indexs_top.top().score << " ";
            if (corpus_.find(indexs_top.top().index) != corpus_.end()) {
                count += 1;
            }
            indexs_top.pop();
        }

        std::cout << "Count = " << count << std::endl;
        
    }

    // std::cout << "Start similar queries simulation" << std::endl << std::endl;
    // for (int i = 0; i < 20; i++) {

    //     int_8 basic_query[ITEM_NUM];
    //     for (int j = 0; j < ITEM_NUM; j++) {
    //         basic_query[j] = (int_8) (rand() % 256);
    //     } 
  
    //     for (int j = 0; j < L; j++) {
    //         std::cout << "Table" << j << " result:" << std::endl;
    //         int basic_index = hash_locate(basic_query, j);
    //         std::cout << "Basic index = " << basic_index << std::endl;
    //         for (int k = 0; k < 5; k++) {
    //             int_8 similar_query[ITEM_NUM];
    //             int change_pos = (rand() % ITEM_NUM);
    //             for (int j = 0; j < ITEM_NUM; j++) 
    //                 similar_query[j] = basic_query[j];
    //             if (((int) (similar_query[change_pos])) != 0) 
    //                 similar_query[change_pos] -= 1;
    //             else 
    //                 similar_query[change_pos] += 1;
    //             int similar_index = hash_locate(similar_query, j);
    //             std::cout << "Similar index " << k << " = " << similar_index << std::endl;
    //         }
    //     }

    // }

    return 0;
}
