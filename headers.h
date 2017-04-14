#ifndef HEADERS_H
#define HEADERS_H
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstddef>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <armadillo>
#include <limits.h>
#include <map>
#include <queue>

using namespace std;
using namespace cv;
using namespace arma;
map<string, int> cl_unique;     // for identifying the number of unique classes.
class Facemash{
    int n;                      // Dimension of the origin image.
    int N;                      // no. of training images.
    mat P;                      // Pattern matrix.
    mat X;                      // mean normalised pattern matrix.
    colvec mu;                  // mean vector.
    mat Y,Y_test,W;             // coordinates and eigenvectors (TODO: for eigenfaces only as of now.)
    mat scatter;                // scatter matrix ( eigenfaces )
    vec closest_models;         // indices of training points closest to test points.
    vector<string> classes;     // store classes of train images.
    vector<string> testClasses; // classes of test data points.
    int read_called;            // no. of times readData has been called.
    int C;                      // no. of classes.
    vec numClass;               // no. of training images in a class.
    mat cl_means;               // mean image for each class.
 public:
     Facemash(){
         read_called = 0;
     }
    void readData(string f);
    void sub_mean();
    mat eigenfaces(int);
    void fisherfaces();
    void save_model();

    void test_eigenfaces();
    double accuracy();
    void class_means();
    void test_fisherfaces();
};


void Facemash::readData(string filename){
    read_called++;
    ifstream ti(filename,ios::in);
    string fname;
    cv::Mat m;
    vector<vector<double> >v;

    while(getline(ti,fname)){
        if(read_called%2){
            classes.push_back(fname.substr(fname.find('\t')+1));
        }
        else
            testClasses.push_back(fname.substr(fname.find('\t')+1));

        m = imread(fname.substr(0,fname.find('\t')),IMREAD_GRAYSCALE);
        m.convertTo(m,CV_64F);
        vector<double> array((double*)m.data, (double*)m.data + m.rows * m.cols);
        v.push_back(array);
    }

    ti.close();
    n = v[0].size();
    N = v.size();
    P.set_size(n,N);

    for(int i=0; i < v.size(); i++){
        P.col(i) = conv_to< colvec >::from(v[i]);
        cl_unique[classes[i]]++;
    }
    C = cl_unique.size();
    numClass.set_size(C);
    X = P;  // do not modify the original pattern images.
}

void Facemash::sub_mean(){
    mu = mean(X,1);
    for(int i=0; i < X.n_cols; i++){
        X.col(i) -= mu;
    }
}

// run PCA and return eigenvectors corresponding to first K eigenvalues.
mat Facemash::eigenfaces(int K = 30){
        // Find eigenvectors of X'X and then use then for calculating eigenvectors of XX'
        // X has dimensions n x N . => X'X has dimensions NxN
        // and XX' has dimensions nxn (>> NxN).
        scatter = X.t()*X;
        cx_mat evec;
        cx_vec eval;
        eig_gen(eval,evec, scatter);
        mat eigvec_pseudo = conv_to< mat >::from(evec);
        vec eigval_pseudo = conv_to< vec >::from(eval);

        // Select best K eigenvectors. on the basis of decreasing eigenvalues.
        mat eigvec(N,K);

        for(int i=0; i < K; i++){
            double max_val = -1;
            int index = -1;
            for(int j=0; j < eigval_pseudo.n_rows ; j++){
                if(eigval_pseudo[j] > max_val){
                    max_val = eigval_pseudo[j];
                    index = j;
                }
            }
            eigval_pseudo[index] = -1;
            eigvec.col(i) = eigvec_pseudo.col(i);
        }

        //Use these eigenvectors to get K eigenvalues of the covariance matrix.
        // We now have K eigenvectors which is less than n in all.
        // These are the most important ones.
        // W contains the weights to be multiplied to get points in a lower dimension.
        W = X * eigvec;
        // normalise eigenvectors
        for(int i = 0;i < W.n_cols; i++){
            W.col(i) /= norm(W.col(i),"inf");
        }

        // get coordinates for the training data:
        Y = W.t() * X;
        cout << "Weights calculated: " << Y.n_rows << " x " << Y.n_cols << endl;

        return W;
}

void Facemash::save_model(){
    ofstream fout("train_model_eigenfaces.txt",ios::out);
    for(int i=0;i<W.n_rows;i++){
        for(int j = 0;j<W.n_cols;j++){
            fout<<W(i,j)<<" ";
        }
        fout<<"\n";
    }
    fout.close();
}

void Facemash::test_eigenfaces(){
    // weights for test data.
    Y_test = W.t() * X;
    //For each test vector, find the closest training vector.
    // Store the index of the closest training data point in the following matrix.
    closest_models.set_size(Y_test.n_cols);

    for(int i=0; i < Y_test.n_cols; i++){
        double min_dist = INT_MAX;
        for(int j = 0; j < Y.n_cols; j++){
            double dist = norm(Y_test.col(i) - Y.col(j));
            if(dist < min_dist){
                //TODO: Put threshold here.
                min_dist = dist;
                closest_models(i) = j;
            }

        }
    }
}

double Facemash::accuracy(){
    //Check accuracy now.
    int count = 0;
    for(int i = 0; i < Y_test.n_cols; i++){
        if(classes[closest_models[i]] == testClasses[i]){
            count++;
        }
    }
    return (((double)count) / ((double)Y_test.n_cols)) * 100;
}

void Facemash::class_means(){
    cl_means = zeros< mat >(n,C);
    for(int i = 0; i < P.n_cols; i++){
        int c = atoi(classes[i].c_str()) - 1;
        cl_means.col(c) += P.col(i);
        numClass[c]++;
    }
    for(int i = 0; i < C; i++){
        cl_means.col(i) /= numClass[i];
    }
    mu = mean(P,1); // total mean.
}
void Facemash::fisherfaces(){
    mat W_pca = eigenfaces(N-C);


    mat S_b = zeros< mat > (n,n);
    mat S_w = zeros< mat > (n,n);
    mat temp = P;
    // subtract class mean from pattern images.
    for(int j = 0; j < N; j++){
        temp.col(j) = temp.col(j) - cl_means.col(atoi(classes[j].c_str()) - 1);
    }
    cout << "Scatter evaluation begins: " << endl;

    for(int i = 0; i < N; i++){
        S_w += temp.col(i)*(temp.col(i).t());
        if(i%5 == 0)
            cout << i << " stages of " << N + C << "done" << endl;
    }
    for(int i = 0; i < C; i++){
        S_b = S_b + numClass[i] * (cl_means.col(i) - mu) * ((cl_means.col(i) - mu).t());
        if(i%5 == 0)
            cout << N + i << " stages of " << N + C << "done" << endl;
    }
    cout << N + C << " stages of " << N + C << "done" << endl;
    cout << "Scatter matrices evaluated"<<endl;
    // reducing dimensions to N-C, since S_w is singular.
    S_b = W_pca.t() * S_b * W_pca;
    S_w = W_pca.t() * S_w * W_pca;

    mat W_fld;
    cx_mat evec;
    cx_vec eval;
    eig_gen(eval, evec, inv(S_w) * S_b );

    W_fld = conv_to< mat >::from(evec);

    W = (W_fld.t() * W_pca.t()).t();
    cout << "Saving model\n";
    save_model();
    cout << "Reducing dimensions\n";
    Y = W.t() * P;
}

void Facemash::test_fisherfaces(){
    test_eigenfaces();
}


#endif
