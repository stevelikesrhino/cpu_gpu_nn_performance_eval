#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <byteswap.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <random>

#define BYTE unsigned char

// read MNIST label header from fp
void read_label_headers(FILE *fp, int *magic, int *num_labels) {
    fread(magic, sizeof(int), 1, fp);
    fread(num_labels, sizeof(int), 1, fp);
    *magic = __bswap_32(*magic);
    *num_labels = __bswap_32(*num_labels);
}

// read MNIST label data from fp
void read_label_data(FILE *fp, int num_labels, BYTE *labels) {
    fread(labels, sizeof(BYTE), num_labels, fp);
}

// read MNIST headers from fp
void read_mnist_headers(FILE *fp, int *magic, int *num_images, int *num_rows,
int *num_cols)
{
    fread(magic, sizeof(int), 1, fp);
    fread(num_images, sizeof(int), 1, fp);
    fread(num_rows, sizeof(int), 1, fp);
    fread(num_cols, sizeof(int), 1, fp);
    *magic = bswap_32(*magic);
    *num_images = bswap_32(*num_images);
    *num_rows = bswap_32(*num_rows);
    *num_cols = bswap_32(*num_cols);
}

// read MNIST images from fp
void read_mnist_images(FILE *fp, int num_images, int num_rows, int num_cols,
BYTE *images)
{
    int total_pixel_count = num_images * num_rows * num_cols;
    if(fread(images, sizeof(BYTE), total_pixel_count, fp) != total_pixel_count)
    {
        printf("Error reading images from file");
        exit(1);
    }
}



// parallel block matrix multiplication using std::threads
// input: BYTE matrix a, b, output: BYTE matrix c
void matrix_multiply(BYTE *a, BYTE *b, BYTE *c, int a_rows, int a_cols,
                    int b_rows, int b_cols, int c_rows, int c_cols)
{
    if(a_cols != b_rows) {
        printf("Error: matrix dimensions do not match for multiplication");
        exit(1);
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int block_size = a_rows / num_threads;
    int remainder = a_rows % num_threads;

    int start = 0;
    int end = 0;

    for(int i = 0; i < num_threads; i++) {
        start = end;
        end = start + block_size;
        if(i == num_threads - 1) {
            end += remainder;
        }
        threads[i] = std::thread([=] {
            for(int j = start; j < end; j++) {
                for(int k = 0; k < b_cols; k++) {
                    int sum = 0;
                    for(int l = 0; l < a_cols; l++) {
                        sum += a[j * a_cols + l] * b[l * b_cols + k];
                    }
                    c[j * c_cols + k] = sum;
                }
            }
        });
    }

    for(int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

// parallel block matrix addition using std::threads
// input: BYTE matrix a, b, output: BYTE matrix c
void matrix_add(BYTE *a, BYTE *b, BYTE *c, int a_rows, int a_cols,
                int b_rows, int b_cols, int c_rows, int c_cols)
{
    if(a_rows != b_rows || a_cols != b_cols) {
        printf("Error: matrix dimensions do not match for addition");
        exit(1);
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int block_size = a_rows / num_threads;
    int remainder = a_rows % num_threads;

    int start = 0;
    int end = 0;

    for(int i = 0; i < num_threads; i++) {
        start = end;
        end = start + block_size;
        if(i == num_threads - 1) {
            end += remainder;
        }
        threads[i] = std::thread([=] {
            for(int j = start; j < end; j++) {
                for(int k = 0; k < a_cols; k++) {
                    c[j * a_cols + k] = a[j * a_cols + k] + b[j * a_cols + k];
                }
            }
        });
    }

    for(int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

// parallel block matrix subtraction using std::threads
// input: BYTE matrix a, b, output: BYTE matrix c
void matrix_subtract(BYTE *a, BYTE *b, BYTE *c, int a_rows, int a_cols,
                     int b_rows, int b_cols, int c_rows, int c_cols)
{
    if(a_rows != b_rows || a_cols != b_cols) {
        printf("Error: matrix dimensions do not match for subtraction");
        exit(1);
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int block_size = a_rows / num_threads;
    int remainder = a_rows % num_threads;

    int start = 0;
    int end = 0;

    for(int i = 0; i < num_threads; i++) {
        start = end;
        end = start + block_size;
        if(i == num_threads - 1) {
            end += remainder;
        }
        threads[i] = std::thread([=] {
            for(int j = start; j < end; j++) {
                for(int k = 0; k < a_cols; k++) {
                    c[j * a_cols + k] = a[j * a_cols + k] - b[j * a_cols + k];
                }
            }
        });
    }

    for(int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

// parallel block matrix transpose using std::threads
// input: BYTE matrix a, output: BYTE matrix b
void matrix_transpose(BYTE *a, BYTE *b, int a_rows, int a_cols, int b_rows,
                      int b_cols)
{
    if(a_rows != b_cols || a_cols != b_rows) {
        printf("Error: matrix dimensions do not match for transpose");
        exit(1);
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int block_size = a_rows / num_threads;
    int remainder = a_rows % num_threads;

    int start = 0;
    int end = 0;

    for(int i = 0; i < num_threads; i++) {
        start = end;
        end = start + block_size;
        if(i == num_threads - 1) {
            end += remainder;
        }
        threads[i] = std::thread([=] {
            for(int j = start; j < end; j++) {
                for(int k = 0; k < a_cols; k++) {
                    b[k * b_cols + j] = a[j * a_cols + k];
                }
            }
        });
    }

    for(int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}


// parallel block matrix constant multiplication using std::threads
// input: BYTE matrix a, BYTE constant c, output: BYTE matrix b
void matrix_constant_multiply(BYTE *a, BYTE *b, BYTE c, int a_rows, int a_cols,
                              int b_rows, int b_cols)
{
    if(a_rows != b_rows || a_cols != b_cols) {
        printf("Error: matrix dimensions do not match for constant multiplication");
        exit(1);
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int block_size = a_rows / num_threads;
    int remainder = a_rows % num_threads;

    int start = 0;
    int end = 0;

    for(int i = 0; i < num_threads; i++) {
        start = end;
        end = start + block_size;
        if(i == num_threads - 1) {
            end += remainder;
        }
        threads[i] = std::thread([=] {
            for(int j = start; j < end; j++) {
                for(int k = 0; k < a_cols; k++) {
                    b[j * a_cols + k] = a[j * a_cols + k] * c;
                }
            }
        });
    }

    for(int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

// RELU (since middle layers are small, no need to parallelize)
// input: BYTE vector a, output: BYTE vector b
void relu(BYTE *a, BYTE *b, int a_size, int b_size)
{
    if(a_size != b_size) {
        printf("Error: vector dimensions do not match for relu");
        exit(1);
    }

    for(int i = 0; i < a_size; i++) {
        if(a[i] < 0) {
            b[i] = 0;
        } else {
            b[i] = a[i];
        }
    }
}

// softmax (since final layer is small, no need to parallelize)
// input: BYTE vector a, output: BYTE vector b
void softmax(BYTE *a, BYTE *b, int a_size, int b_size)
{
    if(a_size != b_size) {
        printf("Error: vector dimensions do not match for softmax");
        exit(1);
    }

    BYTE max = 0;
    for(int i = 0; i < a_size; i++) {
        if(a[i] > max) {
            max = a[i];
        }
    }

    BYTE sum = 0;
    for(int i = 0; i < a_size; i++) {
        b[i] = exp(a[i] - max);
        sum += b[i];
    }

    for(int i = 0; i < a_size; i++) {
        b[i] = b[i] / sum;
    }
}

void label_one_hot(int input, BYTE *output)
{
    for(int i = 0; i < 10; i++) {
        if(i == input) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}



int main(){
    FILE *fp;
    if((fp = fopen("./data/train-images.idx3-ubyte", "rb")) == NULL){
        printf("File open error!");
        exit(1);
    }

    int magic, num_images, num_rows, num_cols;
    read_mnist_headers(fp, &magic, &num_images, &num_rows, &num_cols);
    printf("magic: %d, num_images: %d, num_rows: %d, num_cols: %d\n"
            , magic, num_images, num_rows, num_cols);

    BYTE *train_images = (BYTE *)malloc(num_images * num_rows * num_cols * sizeof(BYTE));
    read_mnist_images(fp, num_images, num_rows, num_cols, train_images);

    for(int i = 0; i<num_rows; i++){
        for(int j = 0; j<num_cols; j++){
            if(train_images[i*num_cols + j] > 0)
                printf("*");
            else
                printf(" ");
        }
        printf("\n");
    }
    
    fclose(fp);
    
    if((fp = fopen("./data/train-labels.idx1-ubyte", "rb")) == NULL){
        printf("File open error!");
        exit(1);
    }

    read_label_headers(fp, &magic, &num_images);
    printf("\nmagic: %d, num_images: %d\n", magic, num_images);
    BYTE *train_labels = (BYTE *)malloc(num_images * sizeof(BYTE));
    read_label_data(fp, num_images, train_labels);
    printf("label of first image: %d\n", train_labels[0]);
    fclose(fp);
    
    BYTE *W2 = (BYTE *)malloc(128 * 784 * sizeof(BYTE));
    BYTE *W3 = (BYTE *)malloc(64 * 128 * sizeof(BYTE));
    BYTE *W4 = (BYTE *)malloc(32 * 64 * sizeof(BYTE));
    BYTE *W5 = (BYTE *)malloc(10 * 32 * sizeof(BYTE));

    BYTE *b2 = (BYTE *)malloc(128 * sizeof(BYTE));
    BYTE *b3 = (BYTE *)malloc(64 * sizeof(BYTE));
    BYTE *b4 = (BYTE *)malloc(32 * sizeof(BYTE));
    BYTE *b5 = (BYTE *)malloc(10 * sizeof(BYTE));

    BYTE *Z2 = (BYTE *)malloc(128 * sizeof(BYTE));
    BYTE *A2 = (BYTE *)malloc(128 * sizeof(BYTE));
    BYTE *Z3 = (BYTE *)malloc(64 * sizeof(BYTE));
    BYTE *A3 = (BYTE *)malloc(64 * sizeof(BYTE));
    BYTE *Z4 = (BYTE *)malloc(32 * sizeof(BYTE));
    BYTE *A4 = (BYTE *)malloc(32 * sizeof(BYTE));
    BYTE *Z5 = (BYTE *)malloc(10 * sizeof(BYTE));
    BYTE *A5 = (BYTE *)malloc(10 * sizeof(BYTE));

    BYTE *result = (BYTE *)malloc(10 * sizeof(BYTE));

    // init weight and biases to random values
    for(int i = 0; i < 128 * 784; i++) {
        W2[i] = rand() % 256;
    }
    for(int i = 0; i < 64 * 128; i++) {
        W3[i] = rand() % 256;
    }
    for(int i = 0; i < 32 * 64; i++) {
        W4[i] = rand() % 256;
    }
    for(int i = 0; i < 10 * 32; i++) {
        W5[i] = rand() % 256;
    }

    for(int i = 0; i < 128; i++) {
        b2[i] = rand() % 256;
    }
    for(int i = 0; i < 64; i++) {
        b3[i] = rand() % 256;
    }
    for(int i = 0; i < 32; i++) {
        b4[i] = rand() % 256;
    }
    for(int i = 0; i < 10; i++) {
        b5[i] = rand() % 256;
    }

    

    free(W2);
    free(W3);
    free(W4);
    free(W5);
    free(b2);
    free(b3);
    free(b4);
    free(b5);
    free(Z2);
    free(A2);
    free(Z3);
    free(A3);
    free(Z4);
    free(A4);
    free(Z5);
    free(A5);
    free(result);


    free(train_images);
    free(train_labels);
    return 0;
}