#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "RBM.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
	// 画像の読み込み
	const int size= 50;
	Mat src_img = imread(argv[1],0);
	// 画像の縮小
	Mat bin_img;
	resize(src_img, bin_img, Size(size,size));
	
	// 画像の二値化
	threshold(bin_img, bin_img, 0, 255, THRESH_BINARY|THRESH_OTSU);
	
	// 元画像の保存
	imwrite("original.png", bin_img);
	
	// データフォーマットの変換
	vector<int> image(size*size);
	for(int y=0; y<bin_img.rows; y++){
		for(int x=0; x <bin_img.cols; x++){
			if ((int) bin_img.data[y*bin_img.cols + x] == 0) image[y*bin_img.cols + x] = 0;
			else image[y*bin_img.cols + x] = 1;
		}
	}
	
	// RBMの訓練
	RBM myRBM(size*size, size/2);
	// 学習率を変えて2段階でやるとうまくいった
	for(int i = 0; i<100; i++){
		myRBM.train(image, 0.1);
	}
	for(int i=0; i<1000; i++){
		myRBM.train(image, 0.001);
	}
	
	// 復元画像の生成
	auto sample = myRBM.get_reconstruction(image);
	for(int y=0; y<bin_img.rows; y++){
		for(int x=0; x <bin_img.cols; x++){
			if (sample[y*bin_img.cols + x] == 0) bin_img.data[y*bin_img.cols + x] = 0;
			else bin_img.data[y*bin_img.cols + x] = 255;
		}
	}
	
	// 復元画像の保存
	imwrite("reconstruct.png", bin_img);
}