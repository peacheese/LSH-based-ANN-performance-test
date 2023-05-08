#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <time.h>

using namespace std;

#define BANK_WIDTH              32                  // 32B width Bank
#define ITEM_NUM                128                 // 128dim Embedding
#define SIZE_EACH_ITEM          1                   // 1B each (int_8)
#define EMBEDDINGS_EACH_BANK    25600               // Vector num in 1 Bank
#define MAX_K                   1024                // Top-1024 at most

#define K 5
#define L 8

typedef unsigned char int_8;

struct hashTable {
    vector<int> bucket[(1 << K)];
};

int main() {

    // xrt::bo in_buf[16];
    size_t buffer_size = BANK_WIDTH * EMBEDDINGS_EACH_BANK;
    size_t hash_table_size = (1 << K) * sizeof(int) + (4 * EMBEDDINGS_EACH_BANK) * sizeof(int);
    size_t srp_size = ITEM_NUM * K * L * sizeof(int);

    default_random_engine generator((unsigned)time(NULL));
    srand((unsigned)time(NULL));

    // L hash tables
    hashTable table[L];

    // Generate random SRP matrix (ITEM_NUM x KL)
    normal_distribution<double> distribution(0.0, 1.0);
    int SRP[K * L][ITEM_NUM];
    for (int i = 0; i < K * L; i++) 
        for (int j = 0; j < ITEM_NUM; j++)
            SRP[i][j] = distribution(generator) > 0 ? 1 : -1;

    for (int embedding_channel = 0; embedding_channel < 4; embedding_channel ++) {
        // in_buf[bank] = xrt::bo(device, buffer_size, bank);
        int_8 buf_map_0[buffer_size / SIZE_EACH_ITEM];
        int_8 buf_map_1[buffer_size / SIZE_EACH_ITEM];
        int_8 buf_map_2[buffer_size / SIZE_EACH_ITEM];
        int_8 buf_map_3[buffer_size / SIZE_EACH_ITEM];
        // Generate random embeddings one by one
        for (int i = 0; i < EMBEDDINGS_EACH_BANK; i++) {
            int_8 embedding[ITEM_NUM]; 
            for (int j = 0; j < ITEM_NUM; j++) {
                embedding[j] = (int_8) (rand() % 256);
            }   
            // Once generated one embedding, locate its index in L tables
            int embedding_idx = i + embedding_channel * EMBEDDINGS_EACH_BANK; 
            // First generate fingerprint
            int sign[K * L];
            for (int j = 0; j < K * L; j++) {
                int res = 0;
                for (int k = 0; k < ITEM_NUM; k++) 
                    res += SRP[j][k] * (int) (embedding[k]);
                sign[j] = res > 0 ? 1 : 0;
            }
            // Then allocate buckets of L tables
            for (int j = 0; j < L; j++) {
                int index = 0;
                for (int k = 0; k < K; k++) {
                    index += sign[j * K + k] * (1 << k);
                }
                table[j].bucket[index].push_back(embedding_idx);
            }
            // Copy the embedding to 4 banks
            memcpy(buf_map_0 + i * BANK_WIDTH, embedding, BANK_WIDTH);
            memcpy(buf_map_1 + i * BANK_WIDTH, embedding + (ITEM_NUM / 4), BANK_WIDTH);
            memcpy(buf_map_2 + i * BANK_WIDTH, embedding + (ITEM_NUM / 4 * 2), BANK_WIDTH);
            memcpy(buf_map_3 + i * BANK_WIDTH, embedding + (ITEM_NUM / 4 * 3), BANK_WIDTH);
        }
        // in_buf[bank].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // for (int i = 0; i < L; i++) 
    //     for (int j = 0; j < (1 << K); j++) 
    //         cout << "Table" << i << " Bucket" << j << " size : " << table[i].bucket[j].size() << endl;

    for (int j = 0; j < (1 << K); j++) {
        int size = 0;
        for (int i = 0; i < L; i++) 
            size += table[i].bucket[j].size();
        cout << "Bucket" << j << " Total size : " << size << endl;
    }

    
    
}