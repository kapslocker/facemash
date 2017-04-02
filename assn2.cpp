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

using namespace std;
using namespace cv;
// Using armadillo library
using namespace arma;

void save_model(mat& A){
    ofstream fout("train_model_eigenfaces.txt",ios::out);
    for(int i=0;i<A.n_rows;i++){
        for(int j = 0;j<A.n_cols;j++){
            fout<<A(i,j)<<" ";
        }
        fout<<"\n";
    }
    fout.close();
}

//read data corresponding to train/test data and store the resulting image in a vector.
void readData(string filename, vector<vector<double> >& v, vector<string>& v2 ){
    ifstream ti(filename,ios::in);
    string fname;
    cv::Mat m;
    int temp=0;
    // This is to read the training data.
    while(getline(ti,fname)){
        v2.push_back(fname.substr(fname.find('\t')+1));
        m = imread(fname.substr(0,fname.find('\t')),IMREAD_GRAYSCALE);
        m.convertTo(m,CV_64F);
        vector<double> array((double*)m.data, (double*)m.data + m.rows * m.cols);
        v.push_back(array);
    }
    ti.close();
}
int n,N;

void compute_mean(colvec& mu,vector<vector<double> >& v ){
    std::vector<double> mu_vec(v[0].size());
    for(int i=0;i<mu_vec.size();i++){
        mu_vec[i] = 0;
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<N;j++){
            mu_vec[i] = mu_vec[i] + v[j][i];
        }
        mu_vec[i]/=v.size();
    }
    mu = conv_to< colvec>::from(mu_vec);
}
// dimension of the smaller space.

int main(int argc, char const *argv[]) {

    vector<vector<double> > v;
    // save the assigned class of each training data.
    vector<string> classes;
    readData("train.txt",v,classes);

    n = v[0].size();
    N = v.size();
    cout<<"N, the dimension of the orignal space is: "<<n<<endl;
    cout<<"No. of images = "<<N<<"\n";


    //compute mean
    colvec mu;
    compute_mean(mu,v);

    // X contains the images stacked column wise. Each column is a vector with image flattened out.
    // The mean has been subtracted from the images.
    arma::mat X(v[0].size(),v.size());

    for(int i=0;i<v.size();i++){
        colvec y = conv_to< colvec >::from(v[i]);
        X.col(i) = y - mu;
    }

    cout<<"Computing Scatter Matrix: ";
    //Scatter Matrix
    arma::mat S_t = X.t()*X;
    cout<<S_t.n_rows<<" x "<<S_t.n_cols<<endl;

    // Find eigenvectors of X'X and then use then for calculating eigenvectors of XX'
    // X has dimensions n x N . => X'X has dimensions NxN
    // and XX' has dimensions nxn (>> NxN).

    // // Compute SVD of the Pattern matrix.
    // arma::mat U,V;
    // arma::vec s;
    // cout<<"Computing SVD of the Scatter Matrix."<<endl;
    // svd(U,s,V,X);

    cx_mat evec;
    cx_vec eval;
    eig_gen(eval,evec, S_t);
    mat eigvec_pseudo = conv_to< mat >::from(evec);
    vec eigval_pseudo = conv_to< vec >::from(eval);


    //Use these eigenvectors to get N eigenvalues of the covariance matrix.
    mat U = X * eigvec_pseudo; // U_i = X * V_i
    // We now have N eigenvectors which is less than n in all. These are the most important ones.


    //TODO: Select best K eigenvectors. on the basis of decreasing eigenvalues.
    int K = N;
    // get weights for the training data:
    mat W = U.t() * X;
    cout<<"Weights calculated: " << W.n_rows << " x " << W.n_cols << endl;
    save_model(W);
    ///////////////////////////// Model trained ////////////////////////////////

    /////// Testing  //////

    vector<vector<double> > test_data;
    vector<string> test_classes;
    readData("test.txt",test_data,test_classes);

    arma::mat T(test_data[0].size(),test_data.size());
    for(int i=0 ;i < test_data.size(); i++){
        colvec y = conv_to< colvec >::from(test_data[i]);
        T.col(i) = y - mu;
    }
    cout << "The Test matrix has dimensions: " << T.n_rows << " x " << T.n_cols << endl;

    // weights for test data.
    mat W_test = U.t() * T;

    //For each test vector, find the closest training vector. Dist is evaluated by L2_norm.
    // store the index of the closest training data point in the following matrix.
    vec closest_models(W_test.n_cols);

    for(int i=0; i<W_test.n_cols; i++){
        double min_dist = INT_MAX;
        for(int j=0; j<W.n_cols; j++){
            double dist = norm(W_test.col(i) - W.col(j));
            if(dist<min_dist){
                min_dist = dist;
                closest_models(i) = j;
            }
            //TODO: Put threshold here if you wish. This is to keep safe from stray images being classified
            // for a class whereas it should be given a class of NOT in this class.
        }
    }
    //Check accuracy now.
    int count = 0;
    for(int i=0; i < W_test.n_cols; i++){
        if(classes[closest_models[i]] == test_classes[i]){
            count++;
        }
    }

    cout << "The accuracy for eigenfaces is: " << ((double)count) / ((double)W_test.n_cols) << endl;

    return 0;
}
