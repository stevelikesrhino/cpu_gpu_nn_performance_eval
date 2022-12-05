#include <byteswap.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

#define BYTE unsigned char

using namespace Eigen;

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
                        int *num_cols) {
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
                       BYTE *images) {
    int total_pixel_count = num_images * num_rows * num_cols;
    if (fread(images, sizeof(BYTE), total_pixel_count, fp) !=
        total_pixel_count) {
        printf("Error reading images from file");
        exit(1);
    }
}

// label to one-hot encoding
void label_to_one_hot(BYTE *labels, float *one_hot_labels, int num_labels) {
    for (int i = 0; i < num_labels; i++) {
        for (int j = 0; j < 10; j++) {
            one_hot_labels[i * 10 + j] = (labels[i] == j) ? 1 : 0;
        }
    }
}

// ReLU
float relu(float x) { return (x > 0) ? x : 0; }

// relu prime
float relu_prime(float x) { return (x > 0) ? 1 : 0; }

// final layer activation
// column wise, normalize each column to sum to 1
// return MatrixXf
MatrixXf minemax(MatrixXf x, int num_rows, int num_cols) {
    MatrixXf result(num_rows, num_cols);
    for(int i=0; i<num_cols; i++) {
        float max = x.col(i).maxCoeff();
        result.col(i) = x.col(i)/max;
    }
    return result;
}


#define NUM_NODES 128

int main() {
    FILE *fp;
    if ((fp = fopen("./data/train-images.idx3-ubyte", "rb")) == NULL) {
        printf("File open error!");
        exit(1);
    }

    int magic, num_images, num_rows, num_cols;
    read_mnist_headers(fp, &magic, &num_images, &num_rows, &num_cols);
    printf("magic: %d, num_images: %d, num_rows: %d, num_cols: %d\n", magic,
           num_images, num_rows, num_cols);

    BYTE *train_images_raw =
        (BYTE *)malloc(num_images * num_rows * num_cols * sizeof(BYTE));
    read_mnist_images(fp, num_images, num_rows, num_cols, train_images_raw);

    float *train_images =
        (float *)malloc(num_images * num_rows * num_cols * sizeof(float));
    for (int i = 0; i < num_images * num_rows * num_cols; i++) {
        train_images[i] = (float)train_images_raw[i] / 255.0;
    }

    /*
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (train_images[i * num_cols + j] > 0) {
                printf("* ");
            } else {
                printf("  ");
            }
        }
        printf("\n");
    } 
    */

    fclose(fp);

    if ((fp = fopen("./data/train-labels.idx1-ubyte", "rb")) == NULL) {
        printf("File open error!");
        exit(1);
    }

    read_label_headers(fp, &magic, &num_images);
    printf("\nmagic: %d, num_images: %d\n", magic, num_images);
    BYTE *train_labels = (BYTE *)malloc(num_images * sizeof(BYTE));
    read_label_data(fp, num_images, train_labels);
    printf("label of first image: %d\n", train_labels[0]);
    fclose(fp);

    printf("\n");

    // ***************** 5 layer NN *****************

    // map images to matrix
    MatrixXf X(num_rows * num_cols, num_images);
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < num_rows * num_cols; j++) {
            X(j, i) = train_images[i * num_rows * num_cols + j];
        }
    }
    
    std::cout << "shape of X: " << X.rows() << " x "
              << X.cols() << std::endl;

    // init layers
    MatrixXf Z1(NUM_NODES, num_images);
    MatrixXf A1(NUM_NODES, num_images);
    MatrixXf Z2(NUM_NODES, num_images);
    MatrixXf A2(NUM_NODES, num_images);
    MatrixXf Z3(NUM_NODES, num_images);
    MatrixXf A3(NUM_NODES, num_images);
    MatrixXf Z4(NUM_NODES, num_images);
    MatrixXf A4(NUM_NODES, num_images);
    MatrixXf Z5(10, num_images);
    MatrixXf A5(10, num_images);

    // init weights
    MatrixXf W1 = MatrixXf::Random(NUM_NODES, 784);
    MatrixXf W2 = MatrixXf::Random(NUM_NODES, NUM_NODES);
    MatrixXf W3 = MatrixXf::Random(NUM_NODES, NUM_NODES);
    MatrixXf W4 = MatrixXf::Random(NUM_NODES, NUM_NODES);
    MatrixXf W5 = MatrixXf::Random(10, NUM_NODES);

    // init biases
    MatrixXf b1 = MatrixXf::Random(NUM_NODES, 1);
    MatrixXf b2 = MatrixXf::Random(NUM_NODES, 1);
    MatrixXf b3 = MatrixXf::Random(NUM_NODES, 1);
    MatrixXf b4 = MatrixXf::Random(NUM_NODES, 1);
    MatrixXf b5 = MatrixXf::Random(10, 1);

    // init one-hot labels
    float *one_hot_labels = (float *)malloc(num_images * 10 * sizeof(float));
    label_to_one_hot(train_labels, one_hot_labels, num_images);
    MatrixXf Y = MatrixXf::Map(one_hot_labels, 10, num_images);
    std::cout << "shape of Y = " << Y.rows() << " x " << Y.cols() << std::endl;

    // init learning rate
    float learning_rate = 0.001;

    // init one matrix for ease of use
    MatrixXf one = MatrixXf::Ones(NUM_NODES, num_images);
    MatrixXf one_T = MatrixXf::Ones(num_images, NUM_NODES);
    MatrixXf column_one = MatrixXf::Ones(NUM_NODES, 1);

    // prepare for parallel
    Eigen::initParallel();
    Eigen::setNbThreads(8);
    // main loop
    std::cout << "start propagating" << std::endl;
    //timer
    auto start = std::chrono::high_resolution_clock::now();

    // bulk process the whole dataset instead of one by one
    
    for(int i = 0; i<100; i++){
        // forward propagation
        Z1 = W1 * X;
        Z1.colwise() += b1.col(0);
        A1 = Z1.unaryExpr(&relu);
        Z2 = W2 * A1;
        Z2.colwise() += b2.col(0);
        A2 = Z2.unaryExpr(&relu);
        Z3 = W3 * A2;
        Z3.colwise() += b3.col(0);
        A3 = Z3.unaryExpr(&relu);
        Z4 = W4 * A3;
        Z4.colwise() += b4.col(0);
        A4 = Z4.unaryExpr(&relu);
        Z5 = W5 * A4;
        Z5.colwise() += b5.col(0);
        A5 = minemax(Z5, 10, num_images);
        
        // backward propagation

        MatrixXf dZ5 = A5 - Y;
        MatrixXf dW5 = dZ5 * A4.transpose() / num_images;
        MatrixXf db5 = dZ5.rowwise().sum() / num_images;
        MatrixXf dZ4 = W5.transpose() * dZ5;
        MatrixXf dW4 = dZ4 * A3.transpose() / num_images;
        MatrixXf db4 = dZ4.rowwise().sum() / num_images;
        MatrixXf dZ3 = W4.transpose() * dZ4;
        dZ3 = dZ3.cwiseProduct(A3.unaryExpr(&relu_prime));
        MatrixXf dW3 = dZ3 * A2.transpose() / num_images;
        MatrixXf db3 = dZ3.rowwise().sum() / num_images;
        MatrixXf dZ2 = W3.transpose() * dZ3;
        dZ2 = dZ2.cwiseProduct(A2.unaryExpr(&relu_prime));
        MatrixXf dW2 = dZ2 * A1.transpose() / num_images;
        MatrixXf db2 = dZ2.rowwise().sum() / num_images;
        MatrixXf dZ1 = W2.transpose() * dZ2;
        dZ1 = dZ1.cwiseProduct(A1.unaryExpr(&relu_prime));
        MatrixXf dW1 = dZ1 * X.transpose() / num_images;
        MatrixXf db1 = dZ1.rowwise().sum() / num_images;

        // update weights and biases
        
        W5 = W5 - learning_rate * dW5;
        b5 = b5 - learning_rate * db5;
        W4 = W4 - learning_rate * dW4;
        b4 = b4 - learning_rate * db4;
        W3 = W3 - learning_rate * dW3;
        b3 = b3 - learning_rate * db3;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "end propagation" <<std::endl;
    std::cout << "time elapsed: " << elapsed.count() << " s\n";


    free(train_images_raw);
    free(train_images);
    free(train_labels);
    free(one_hot_labels);

    // ***************** test *****************
    std::cout << "\n\n ***** start testing *****\n" << std::endl;
    if ((fp = fopen("./data/t10k-labels.idx1-ubyte", "rb")) == NULL) {
        printf("File open error!");
        exit(1);
    }
    
    BYTE *test_labels = (BYTE *)malloc(10000 * sizeof(BYTE));
    read_label_headers(fp, &magic, &num_images);
    read_label_data(fp, 10000, test_labels);
    fclose(fp);

    if ((fp = fopen("./data/t10k-images.idx3-ubyte", "rb")) == NULL) {
        printf("File open error!");
        exit(1);
    }
    
    read_mnist_headers(fp, &magic, &num_images, &num_rows, &num_cols);
    BYTE *test_images_raw = (BYTE *)malloc(num_images * num_rows * num_cols * sizeof(BYTE));
    read_mnist_images(fp, num_images, num_rows, num_cols, test_images_raw);
    fclose(fp);
    float *test_images =
        (float *)malloc(num_images * num_rows * num_cols * sizeof(float));
    for (int i = 0; i < num_images * num_rows * num_cols; i++) {
        test_images[i] = (float)test_images_raw[i] / 255.0;
    }

    std::cout<< "num_images: " << num_images << std::endl;

    int correct = 0;
    for (int i = 0; i < num_images; i++) {
        // forward propagation
        Z1 = W1 * Map<MatrixXf>(test_images + i * 784, 784, 1) + b1;
        A1 = Z1.unaryExpr(&relu);
        Z2 = W2 * A1 + b2;
        A2 = Z2.unaryExpr(&relu);
        Z3 = W3 * A2 + b3;
        A3 = Z3.unaryExpr(&relu);
        Z4 = W4 * A3 + b4;
        A4 = Z4.unaryExpr(&relu);
        Z5 = W5 * A4 + b5;
        A5 = minemax(Z5, 10, 1);

        int max_index = 0;
        float max_value = A4(0, 0);
        for (int j = 1; j < 10; j++) {
            if (A4(j, 0) > max_value) {
                max_index = j;
                max_value = A4(j, 0);
            }
        }
        if (max_index == test_labels[i]) {
            correct++;
        }
    }

    std::cout << "\n\naccuracy: " << (float)correct / num_images << std::endl;

    free(test_images_raw);

    return 0;
}