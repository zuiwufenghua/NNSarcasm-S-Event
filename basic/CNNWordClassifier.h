/*
 * CNNWordClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_CNNWordClassifier_H_
#define SRC_CNNWordClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"

#include "NRMat.h"
#include "MyLib.h"
#include "tensor.h"

#include "UniHidderLayer.h"
#include "BiHidderLayer.h"
#include "Utiltensor.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class CNNWordClassifier {
public:
	CNNWordClassifier() {
		_b_wordEmb_finetune = true;
		_dropOut = 0.5;
	}
	~CNNWordClassifier() {

	}

public:
	Tensor<xpu, 2, double> _wordEmb;
	Tensor<xpu, 2, double> _grad_wordEmb;
	Tensor<xpu, 2, double> _eg2_wordEmb;
	Tensor<xpu, 2, double> _ft_wordEmb;
	hash_set<int> _word_indexers;

	int _wordcontext;
	int _wordSize;
	int _wordDim;
	bool _b_wordEmb_finetune;
	int _wordHiddenSize;
	int _word_cnn_iSize;
	int _token_representation_size;

	int _hiddenSize;

	//word
	UniHidderLayer<xpu> _wordCnnProject;
	UniHidderLayer<xpu> _wordGatedPooling;
	UniHidderLayer<xpu> _wordTanhProject;

	//soft max
	UniHidderLayer<xpu> _olayer_linear;

	int _labelSize;

	Metric _eval;

	double _dropOut;

	int _remove; // 1, avg, 2, max, 3 min, 4 gated

	int _poolmanners;

	int topK = 10;

public:

	inline void init(const NRMat<double>& wordEmb, int wordcontext,
			int labelSize, int wordHiddenSize, int hiddenSize) {
		_wordcontext = wordcontext;
		_wordSize = wordEmb.nrows();
		_wordDim = wordEmb.ncols();
		_poolmanners = 4;

		_labelSize = labelSize;
		_hiddenSize = hiddenSize;
		_wordHiddenSize = wordHiddenSize;
		_token_representation_size = _wordDim;

		_word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

		_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
		_grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
		_eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
		_ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
		assign(_wordEmb, wordEmb);
		for (int idx = 0; idx < _wordSize; idx++) {
			norm2one(_wordEmb, idx);
		}

		_wordCnnProject.initial(_wordHiddenSize, _word_cnn_iSize, true, 30, 0);
		_wordGatedPooling.initial(_wordHiddenSize, _wordHiddenSize, true, 40,
				3);
		_wordTanhProject.initial(_hiddenSize, _poolmanners * _wordHiddenSize,
				true, 50, 0);

		_olayer_linear.initial(_labelSize, _hiddenSize, false, 60, 2);

		_eval.reset();
		_word_indexers.clear();

		_remove = 0;

	}

	inline void release() {
		FreeSpace(&_wordEmb);
		FreeSpace(&_grad_wordEmb);
		FreeSpace(&_eg2_wordEmb);
		FreeSpace(&_ft_wordEmb);
		_wordCnnProject.release();
		_wordGatedPooling.release();
		_wordTanhProject.release();
		_olayer_linear.release();
	}

	inline double process(const vector<Example>& examples, int iter) {
		_eval.reset();
		_word_indexers.clear();

		int example_num = examples.size();
		double cost = 0.0;
		int offset = 0;
		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			int sentNum = example.m_features.size();

			//word related cache
			Tensor<xpu, 3, double> wordprime[sentNum], wordprimeLoss[sentNum],
					wordprimeMask[sentNum];
			Tensor<xpu, 3, double> wordrepresent[sentNum],
					wordrepresentLoss[sentNum];
			Tensor<xpu, 4, double> wordinputcontext[sentNum],
					wordinputcontextLoss[sentNum];
			Tensor<xpu, 3, double> wordInput[sentNum], wordInputLoss[sentNum];
			Tensor<xpu, 3, double> wordHidden[sentNum], wordHiddenLoss[sentNum],
					wordHiddenLossTmp[sentNum];
			vector<vector<Tensor<xpu, 2, double> > > wordPool(sentNum),
					wordPoolLoss(sentNum);
			vector<vector<Tensor<xpu, 3, double> > > wordPoolIndex(sentNum),
					wordPoolIndexLoss(sentNum);

			Tensor<xpu, 3, double> wordGateweight[sentNum],
					wordGateweightLoss[sentNum], wordGateweightIndex[sentNum];
			Tensor<xpu, 2, double> wordGateweightsum[sentNum],
					wordGateweightsumLoss[sentNum];

			Tensor<xpu, 2, double> wordPoolmerge[sentNum],
					wordPoolmergeLoss[sentNum];
			Tensor<xpu, 2, double> wordProject[sentNum],
					wordProjectLoss[sentNum];
			Tensor<xpu, 2, double> wordOutput[sentNum], wordOutputLoss[sentNum],
					wordScores[sentNum];

			//initialize
			for (int idx = 0; idx < sentNum; idx++) {
				int window = 2 * _wordcontext + 1;
				const Feature& feature = example.m_features[idx];

				int word_num = feature.words.size();
				int word_cnn_iSize = _word_cnn_iSize;
				int wordHiddenSize = _wordHiddenSize;

				wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim),
						0.0);
				wordprimeLoss[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, _wordDim), 0.0);
				wordprimeMask[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, _wordDim), 1.0);
				wordrepresent[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, _token_representation_size), 0.0);
				wordrepresentLoss[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, _token_representation_size), 0.0);
				wordinputcontext[idx] = NewTensor<xpu>(
						Shape4(word_num, window, 1, _token_representation_size),
						0.0);
				wordinputcontextLoss[idx] = NewTensor<xpu>(
						Shape4(word_num, window, 1, _token_representation_size),
						0.0);
				wordInput[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, word_cnn_iSize), 0.0);
				wordInputLoss[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, word_cnn_iSize), 0.0);
				wordHidden[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
				wordHiddenLoss[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
				wordHiddenLossTmp[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);

				vector<Tensor<xpu, 2, double> > tempWordPool(_poolmanners);
				vector<Tensor<xpu, 2, double> > tempWordPoolLoss(_poolmanners);
				vector<Tensor<xpu, 3, double> > tempWordPoolIndex(_poolmanners);
				vector<Tensor<xpu, 3, double> > tempWordPoolIndexLoss(
						_poolmanners);
				for (int idm = 0; idm < _poolmanners; idm++) {
					tempWordPool[idm] = NewTensor<xpu>(
							Shape2(1, wordHiddenSize), 0.0);
					tempWordPoolLoss[idm] = NewTensor<xpu>(
							Shape2(1, wordHiddenSize), 0.0);
					tempWordPoolIndex[idm] = NewTensor<xpu>(
							Shape3(word_num, 1, wordHiddenSize), 0.0);
					tempWordPoolIndexLoss[idm] = NewTensor<xpu>(
							Shape3(word_num, 1, wordHiddenSize), 0.0);
				}
				wordPool[idx] = tempWordPool;
				wordPoolLoss[idx] = tempWordPoolLoss;
				wordPoolIndex[idx] = tempWordPoolIndex;
				wordPoolIndexLoss[idx] = tempWordPoolIndexLoss;

				wordGateweight[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
				wordGateweightIndex[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
				wordGateweightLoss[idx] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);

				wordGateweightsum[idx] = NewTensor<xpu>(
						Shape2(1, wordHiddenSize), 0.0);
				wordGateweightsumLoss[idx] = NewTensor<xpu>(
						Shape2(1, wordHiddenSize), 0.0);

				wordPoolmerge[idx] = NewTensor<xpu>(
						Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
				wordPoolmergeLoss[idx] = NewTensor<xpu>(
						Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
				wordProject[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
				wordProjectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize),
						0.0);
				wordOutput[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
				wordOutputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize),
						0.0);
				wordScores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
			}

			for (int idx = 0; idx < sentNum; idx++) {

				//forward propagation
				//word CNN
				const Feature& feature = example.m_features[idx];
				int word_num = feature.words.size();
				int window = 2 * _wordcontext + 1;
				int curcontext = _wordcontext;

				const vector<int>& words = feature.words;

				srand(iter * example_num + count * sentNum + idx);

				for (int idy = 0; idy < word_num; idy++) {
					offset = words[idy];
					wordprime[idx][idy][0] = _wordEmb[offset]
							/ _ft_wordEmb[offset];
				}

				//word dropout
				for (int idy = 0; idy < word_num; idy++) {
					for (int j = 0; j < _wordDim; j++) {
						if (1.0 * rand() / RAND_MAX >= _dropOut) {
							wordprimeMask[idx][idy][0][j] = 1.0;
						} else {
							wordprimeMask[idx][idy][0][j] = 0.0;
						}
					}
					wordprime[idx][idy] = wordprime[idx][idy]
							* wordprimeMask[idx][idy];
				}

				//word representation
				for (int idy = 0; idy < word_num; idy++) {
					wordrepresent[idx][idy] += wordprime[idx][idy];
				}

				//word context
				for (int idy = 0; idy < word_num; idy++) {
					wordinputcontext[idx][idy][0] += wordrepresent[idx][idy];
					for (int idc = 1; idc <= curcontext; idc++) {
						if (idy - idc >= 0) {
							wordinputcontext[idx][idy][2 * idc - 1] +=
									wordrepresent[idx][idy - idc];
						}
						if (idy + idc < word_num) {
							wordinputcontext[idx][idy][2 * idc] +=
									wordrepresent[idx][idy + idc];
						}
					}
				}

				//word reshape
				for (int idy = 0; idy < word_num; idy++) {
					offset = 0;
					for (int i = 0; i < window; i++) {
						for (int j = 0; j < _token_representation_size; j++) {
							wordInput[idx][idy][0][offset] =
									wordinputcontext[idx][idy][i][0][j];
							offset++;
						}
					}
				}

				//word convolution
				for (int idy = 0; idy < word_num; idy++)
					_wordCnnProject.ComputeForwardScore(wordInput[idx][idy],
							wordHidden[idx][idy]);

				//word pooling
				if ((_remove > 0 && _remove != 1)
						|| (_remove < 0 && _remove == -1) || _remove == 0) {
					avgpool_forward(wordHidden[idx], wordPool[idx][0],
							wordPoolIndex[idx][0]);
				}
				if ((_remove > 0 && _remove != 2)
						|| (_remove < 0 && _remove == -2) || _remove == 0) {
					maxpool_forward(wordHidden[idx], wordPool[idx][1],
							wordPoolIndex[idx][1]);
				}
				if ((_remove > 0 && _remove != 3)
						|| (_remove < 0 && _remove == -3) || _remove == 0) {
					minpool_forward(wordHidden[idx], wordPool[idx][2],
							wordPoolIndex[idx][2]);
				}

				//gated pooling
				if ((_remove > 0 && _remove != 4)
						|| (_remove < 0 && _remove == -4) || _remove == 0) {
					for (int idy = 0; idy < word_num; idy++) {
						_wordGatedPooling.ComputeForwardScore(
								wordHidden[idx][idy], wordGateweight[idx][idy]);
					}
					sumpool_forward(wordGateweight[idx], wordGateweightsum[idx],
							wordGateweightIndex[idx]);
					for (int idy = 0; idy < word_num; idy++) {
						wordPoolIndex[idx][3][idy] = wordGateweight[idx][idy]
								/ wordGateweightsum[idx];
					}
					for (int idy = 0; idy < word_num; idy++) {
						wordPool[idx][3] += wordHidden[idx][idy]
								* wordPoolIndex[idx][3][idy];
					}
				}

				concat(wordPool[idx], wordPoolmerge[idx]);
				_wordTanhProject.ComputeForwardScore(wordPoolmerge[idx],
						wordProject[idx]);
				_olayer_linear.ComputeForwardScore(wordProject[idx],
						wordOutput[idx]);
			}

			int optLabel = -1;
			for (int i = 0; i < _labelSize; ++i) {
				if (optLabel < 0
						|| wordOutput[0][0][i] > wordOutput[0][0][optLabel])
					optLabel = i;
			}

			//sigmoid
//      double sum = 0.0;
//      for (int i = 0; i < _labelSize; ++i) {
//        wordScores[0][0][i] = 1.0 / (1.0 + exp(-wordOutput[0][0][i]));
//      }
//      if (example.m_labels[optLabel] == true)
//        _eval.correct_label_count++;
//      _eval.overall_label_count++;
//      for (int i = 0; i < _labelSize; i++) {
//        double realY = example.m_labels[i] ? 1.0:0.0;
//        wordOutputLoss[0][0][i] = wordScores[0][0][i] - realY;
//        sum += log_loss(realY, wordScores[0][0][i]);
//      }
//      cost += sum / (example_num * _labelSize);

			//soft max
			double sum1 = 0.0;
			double sum2 = 0.0;
			double maxScore = wordOutput[0][0][optLabel];

			for (int i = 0; i < _labelSize; ++i) {
				wordScores[0][0][i] = -1e10;
				if (example.m_labels[i] >= 0) {
					wordScores[0][0][i] = exp(wordOutput[0][0][i] - maxScore);
					if (example.m_labels[i] == 1)
						sum1 += wordScores[0][0][i];
					sum2 += wordScores[0][0][i];
				}
			}
			cost += (log(sum2) - log(sum1)) / example_num;

			//topK score
			int topKIndex[topK];
			double topKScores[topK];
			for (int i = 0; i < topK; i++) {
				topKScores[i] = -1e10;
				for (int j = 0; j < _labelSize; j++) {
					if (wordScores[0][0][j] > topKScores[i]) {
						topKScores[i] = wordScores[0][0][j];
						topKIndex[i] = j;
					}
				}
				wordScores[0][0][topKIndex[i]] = -1e20;   //remove temporarily
			}
			for(int i = 0; i < topK; i++) wordScores[0][0][topKIndex[i]] = topKScores[i];  //recovery

			//correctness judge
			bool correct = true;
			int hitPosNum = 0;
			for(int i = 0; _labelSize; i++){
				if(example.m_labels[i]){
					bool find = false;
					for(int j = 0; j < topK; j++){
						if(i == topKIndex[j]){
							find = true;
							hitPosNum++;
							break;
						}
					}
					if(!find){
						if(hitPosNum >= topK) cout << "hitPosNum exceed topK: " << topK << endl;
						correct = false;
						break;
					}
				}
			}
			if(correct) _eval.correct_label_count++;
			_eval.overall_label_count++;


//			if (example.m_labels[optLabel] == true)
//				_eval.correct_label_count++;
//			_eval.overall_label_count++;

			for (int i = 0; i < _labelSize; ++i) {
				wordOutputLoss[0][0][i] = 0.0;
				if (example.m_labels[i] >= 0) {
					wordOutputLoss[0][0][i] = (wordScores[0][0][i] / sum2
							- example.m_labels[i]) / example_num;
				}
			}

			// loss backward propagation

			for (int idx = 0; idx < sentNum; idx++) {
				_olayer_linear.ComputeBackwardLoss(wordProject[idx],
						wordOutput[idx], wordOutputLoss[idx],
						wordProjectLoss[idx]);
				_wordTanhProject.ComputeBackwardLoss(wordPoolmerge[idx],
						wordProject[idx], wordProjectLoss[idx],
						wordPoolmergeLoss[idx]);

				unconcat(wordPoolLoss[idx], wordPoolmergeLoss[idx]);

				const Feature& feature = example.m_features[idx];
				int window = 2 * _wordcontext + 1;
				int curcontext = _wordcontext;

				const vector<int>& words = feature.words;
				int word_num = words.size();

				//word pooling
				if ((_remove > 0 && _remove != 1)
						|| (_remove < 0 && _remove == -1) || _remove == 0) {
					wordHiddenLossTmp[idx] = 0.0;
					pool_backward(wordPoolLoss[idx][0], wordPoolIndex[idx][0],
							wordHiddenLossTmp[idx]);
					wordHiddenLoss[idx] = wordHiddenLoss[idx]
							+ wordHiddenLossTmp[idx];
				}
				if ((_remove > 0 && _remove != 2)
						|| (_remove < 0 && _remove == -2) || _remove == 0) {
					wordHiddenLossTmp[idx] = 0.0;
					pool_backward(wordPoolLoss[idx][1], wordPoolIndex[idx][1],
							wordHiddenLossTmp[idx]);
					wordHiddenLoss[idx] = wordHiddenLoss[idx]
							+ wordHiddenLossTmp[idx];
				}
				if ((_remove > 0 && _remove != 3)
						|| (_remove < 0 && _remove == -3) || _remove == 0) {
					wordHiddenLossTmp[idx] = 0.0;
					pool_backward(wordPoolLoss[idx][2], wordPoolIndex[idx][2],
							wordHiddenLossTmp[idx]);
					wordHiddenLoss[idx] = wordHiddenLoss[idx]
							+ wordHiddenLossTmp[idx];
				}

				//gated pooling
				if ((_remove > 0 && _remove != 4)
						|| (_remove < 0 && _remove == -4) || _remove == 0) {
					for (int idy = 0; idy < word_num; idy++) {
						wordPoolIndexLoss[idx][3][idy] = wordPoolLoss[idx][3]
								* wordHidden[idx][idy];
						wordHiddenLoss[idx][idy] += wordPoolLoss[idx][3]
								* wordPoolIndex[idx][3][idy];
					}

					for (int idy = 0; idy < word_num; idy++) {
						wordGateweightLoss[idx][idy] +=
								wordPoolIndexLoss[idx][3][idy]
										/ wordGateweightsum[idx];
						wordGateweightsumLoss[idx] -=
								wordPoolIndexLoss[idx][3][idy]
										* wordGateweight[idx][idy]
										/ wordGateweightsum[idx]
										/ wordGateweightsum[idx];
					}

					pool_backward(wordGateweightsumLoss[idx],
							wordGateweightIndex[idx], wordGateweightLoss[idx]);

					wordHiddenLossTmp[idx] = 0.0;
					for (int idy = 0; idy < word_num; idy++) {
						_wordGatedPooling.ComputeBackwardLoss(
								wordHidden[idx][idy], wordGateweight[idx][idy],
								wordGateweightLoss[idx][idy],
								wordHiddenLossTmp[idx][idy]);
						//hiddenLoss[idy] += hiddenLossTmp[idy];
					}
					wordHiddenLoss[idx] = wordHiddenLoss[idx]
							+ wordHiddenLossTmp[idx];
				}

				//word convolution
				for (int idy = 0; idy < word_num; idy++) {
					_wordCnnProject.ComputeBackwardLoss(wordInput[idx][idy],
							wordHidden[idx][idy], wordHiddenLoss[idx][idy],
							wordInputLoss[idx][idy]);
				}

				//word reshape
				for (int idy = 0; idy < word_num; idy++) {
					offset = 0;
					for (int i = 0; i < window; i++) {
						for (int j = 0; j < _token_representation_size; j++) {
							wordinputcontextLoss[idx][idy][i][0][j] =
									wordInputLoss[idx][idy][0][offset];
							offset++;
						}
					}
				}

				//word context
				for (int idy = 0; idy < word_num; idy++) {
					wordrepresentLoss[idx][idy] +=
							wordinputcontextLoss[idx][idy][0];
					for (int idc = 1; idc <= curcontext; idc++) {
						if (idy - idc >= 0) {
							wordrepresentLoss[idx][idy - idc] +=
									wordinputcontextLoss[idx][idy][2 * idc - 1];
						}
						if (idy + idc < word_num) {
							wordrepresentLoss[idx][idy + idc] +=
									wordinputcontextLoss[idx][idy][2 * idc];
						}
					}
				}

				//word representation
				for (int idy = 0; idy < word_num; idy++) {
					wordprimeLoss[idx][idy] += wordrepresentLoss[idx][idy];
				}

				//word dropout
				for (int idy = 0; idy < word_num; idy++) {
					wordprimeLoss[idx][idy] = wordprimeLoss[idx][idy]
							* wordprimeMask[idx][idy];
				}

				//word finetune
				if (_b_wordEmb_finetune) {
					for (int idy = 0; idy < word_num; idy++) {
						offset = words[idy];
						_grad_wordEmb[offset] += wordprimeLoss[idx][idy][0];
						_word_indexers.insert(offset);
					}
				}

			}

			//release
			//word
			for (int idx = 0; idx < sentNum; idx++) {
				FreeSpace(&(wordprime[idx]));
				FreeSpace(&(wordprimeLoss[idx]));
				FreeSpace(&(wordprimeMask[idx]));
				FreeSpace(&(wordrepresent[idx]));
				FreeSpace(&(wordrepresentLoss[idx]));
				FreeSpace(&(wordinputcontext[idx]));
				FreeSpace(&(wordinputcontextLoss[idx]));
				FreeSpace(&(wordInput[idx]));
				FreeSpace(&(wordInputLoss[idx]));
				FreeSpace(&(wordHidden[idx]));
				FreeSpace(&(wordHiddenLoss[idx]));
				FreeSpace(&(wordHiddenLossTmp[idx]));
				for (int idm = 0; idm < _poolmanners; idm++) {
					FreeSpace(&(wordPool[idx][idm]));
					FreeSpace(&(wordPoolLoss[idx][idm]));
					FreeSpace(&(wordPoolIndex[idx][idm]));
					FreeSpace(&(wordPoolIndexLoss[idx][idm]));
				}
				FreeSpace(&(wordGateweight[idx]));
				FreeSpace(&(wordGateweightLoss[idx]));
				FreeSpace(&(wordGateweightIndex[idx]));
				FreeSpace(&(wordGateweightsum[idx]));
				FreeSpace(&(wordGateweightsumLoss[idx]));
				FreeSpace(&(wordPoolmerge[idx]));
				FreeSpace(&(wordPoolmergeLoss[idx]));
				FreeSpace(&(wordProject[idx]));
				FreeSpace(&(wordProjectLoss[idx]));
				FreeSpace(&(wordOutput[idx]));
				FreeSpace(&(wordOutputLoss[idx]));
				FreeSpace(&(wordScores[idx]));
			}
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	double log_loss(double real_y, double predict_y) {
		if (real_y == predict_y)
			return 0.0;
		else {
			if (predict_y <= 0.0)
				predict_y = 1e-9;
			else if (predict_y >= 1.0)
				predict_y = 1.0 - 1e-9;
			return -(real_y * log(predict_y) + (1 - real_y) * log(1 - predict_y));
		}
	}

	int predict(const vector<Feature>& features, vector<double>& results) {
		int sentNum = features.size();
		int offset = 0;

		//word related cache
		Tensor<xpu, 3, double> wordprime[sentNum];
		Tensor<xpu, 3, double> wordrepresent[sentNum];
		Tensor<xpu, 4, double> wordinputcontext[sentNum];
		Tensor<xpu, 3, double> wordInput[sentNum];
		Tensor<xpu, 3, double> wordHidden[sentNum];

		vector<vector<Tensor<xpu, 2, double> > > wordPool(sentNum);
		vector<vector<Tensor<xpu, 3, double> > > wordPoolIndex(sentNum);
		Tensor<xpu, 3, double> wordGateweight[sentNum],
				wordGateweightIndex[sentNum];
		Tensor<xpu, 2, double> wordGateweightsum[sentNum];

		Tensor<xpu, 2, double> wordPoolmerge[sentNum];
		Tensor<xpu, 2, double> wordProject[sentNum];
		Tensor<xpu, 2, double> wordOutput[sentNum], wordScores[sentNum];

		//initialize
		for (int idx = 0; idx < sentNum; idx++) {
			int window = 2 * _wordcontext + 1;
			const Feature& feature = features[idx];

			int word_num = feature.words.size();
			int word_cnn_iSize = _word_cnn_iSize;
			int wordHiddenSize = _wordHiddenSize;

			wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
			wordrepresent[idx] = NewTensor<xpu>(
					Shape3(word_num, 1, _token_representation_size), 0.0);
			wordinputcontext[idx] = NewTensor<xpu>(
					Shape4(word_num, window, 1, _token_representation_size),
					0.0);
			wordInput[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize),
					0.0);
			wordHidden[idx] = NewTensor<xpu>(
					Shape3(word_num, 1, wordHiddenSize), 0.0);

			vector<Tensor<xpu, 2, double> > tempWordPool(_poolmanners);
			vector<Tensor<xpu, 3, double> > tempWordPoolIndex(_poolmanners);
			for (int idm = 0; idm < _poolmanners; idm++) {
				tempWordPool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize),
						0.0);
				tempWordPoolIndex[idm] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
			}
			wordPool[idx] = tempWordPool;
			wordPoolIndex[idx] = tempWordPoolIndex;

			wordGateweight[idx] = NewTensor<xpu>(
					Shape3(word_num, 1, wordHiddenSize), 0.0);
			wordGateweightIndex[idx] = NewTensor<xpu>(
					Shape3(word_num, 1, wordHiddenSize), 0.0);
			wordGateweightsum[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize),
					0.0);
			wordPoolmerge[idx] = NewTensor<xpu>(Shape2(1, _poolmanners * wordHiddenSize), 0.0);
			wordProject[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
			wordOutput[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
			wordScores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
		}

		for (int idx = 0; idx < sentNum; idx++) {

			//forward propagation
			//word CNN
			const Feature& feature = features[idx];
			int word_num = feature.words.size();
			int window = 2 * _wordcontext + 1;
			int curcontext = _wordcontext;

			const vector<int>& words = feature.words;

			for (int idy = 0; idy < word_num; idy++) {
				offset = words[idy];
				wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
			}

			//word representation
			for (int idy = 0; idy < word_num; idy++) {
				wordrepresent[idx][idy] += wordprime[idx][idy];
			}

			//word context
			for (int idy = 0; idy < word_num; idy++) {
				wordinputcontext[idx][idy][0] += wordrepresent[idx][idy];
				for (int idc = 1; idc <= curcontext; idc++) {
					if (idy - idc >= 0) {
						wordinputcontext[idx][idy][2 * idc - 1] +=
								wordrepresent[idx][idy - idc];
					}
					if (idy + idc < word_num) {
						wordinputcontext[idx][idy][2 * idc] +=
								wordrepresent[idx][idy + idc];
					}
				}
			}

			//word reshape
			for (int idy = 0; idy < word_num; idy++) {
				offset = 0;
				for (int i = 0; i < window; i++) {
					for (int j = 0; j < _token_representation_size; j++) {
						wordInput[idx][idy][0][offset] =
								wordinputcontext[idx][idy][i][0][j];
						offset++;
					}
				}
			}

			//word convolution
			for (int idy = 0; idy < word_num; idy++)
				_wordCnnProject.ComputeForwardScore(wordInput[idx][idy],
						wordHidden[idx][idy]);

			//word pooling
			if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1)
					|| _remove == 0) {
				avgpool_forward(wordHidden[idx], wordPool[idx][0],
						wordPoolIndex[idx][0]);
			}
			if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2)
					|| _remove == 0) {
				maxpool_forward(wordHidden[idx], wordPool[idx][1],
						wordPoolIndex[idx][1]);
			}
			if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3)
					|| _remove == 0) {
				minpool_forward(wordHidden[idx], wordPool[idx][2],
						wordPoolIndex[idx][2]);
			}

			//gated pooling
			if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4)
					|| _remove == 0) {
				for (int idy = 0; idy < word_num; idy++) {
					_wordGatedPooling.ComputeForwardScore(wordHidden[idx][idy],
							wordGateweight[idx][idy]);
				}
				sumpool_forward(wordGateweight[idx], wordGateweightsum[idx],
						wordGateweightIndex[idx]);
				for (int idy = 0; idy < word_num; idy++) {
					wordPoolIndex[idx][3][idy] = wordGateweight[idx][idy]
							/ wordGateweightsum[idx];
				}
				for (int idy = 0; idy < word_num; idy++) {
					wordPool[idx][3] += wordHidden[idx][idy]
							* wordPoolIndex[idx][3][idy];
				}
			}

			// sentence embedding
			concat(wordPool[idx], wordPoolmerge[idx]);
			_wordTanhProject.ComputeForwardScore(wordPoolmerge[idx],
					wordProject[idx]);
			_olayer_linear.ComputeForwardScore(wordProject[idx],
					wordOutput[idx]);
		}

		//sigmoid
//		double sum = 0.0;
//		for (int i = 0; i < _labelSize; ++i) {
//			wordScores[0][0][i] = 1.0 / (1.0 + exp(-wordOutput[0][0][i]));
//		}
//		results.resize(_labelSize);
//		for (int i = 0; i < _labelSize; ++i) {
//			results[i] = wordScores[0][0][i];
//		}
//		int optLabel = 0;
//		for (int i = 1; i < _labelSize; ++i) {
//			if (results[i] > results[optLabel])
//				optLabel = i;
//		}

		int optLabel = 0;
		for (int i = 1; i < _labelSize; ++i) {
			if (wordOutput[0][0][i] > wordOutput[0][0][optLabel])
				optLabel = i;
		}

		double sum = 0.0;
		double maxScore = wordOutput[0][0][optLabel];
		for (int i = 0; i < _labelSize; ++i) {
			wordScores[0][0][i] = exp(wordOutput[0][0][i] - maxScore);
			sum += wordScores[0][0][i];
		}

		results.resize(_labelSize);
		for (int i = 0; i < _labelSize; ++i) {
			results[i] = wordScores[0][0][i] / sum;
		}

		//release
		//word
		for (int idx = 0; idx < sentNum; idx++) {
			FreeSpace(&(wordprime[idx]));
			FreeSpace(&(wordrepresent[idx]));
			FreeSpace(&(wordinputcontext[idx]));
			FreeSpace(&(wordInput[idx]));
			FreeSpace(&(wordHidden[idx]));
			for (int idm = 0; idm < _poolmanners; idm++) {
				FreeSpace(&(wordPool[idx][idm]));
				FreeSpace(&(wordPoolIndex[idx][idm]));
			}
			FreeSpace(&(wordGateweight[idx]));
			FreeSpace(&(wordGateweightIndex[idx]));
			FreeSpace(&(wordGateweightsum[idx]));

			FreeSpace(&(wordPoolmerge[idx]));
			FreeSpace(&(wordProject[idx]));
			FreeSpace(&(wordOutput[idx]));
			FreeSpace(&(wordScores[idx]));
		}
		return optLabel;
	}

	double computeScore(const Example& example) {
		int seq_size = example.m_features.size();
		int offset = 0;
		if (seq_size > 2) {
			std::cout << "error" << std::endl;
		}

		Tensor<xpu, 3, double> wordprime;
		Tensor<xpu, 3, double> wordrepresent;
		Tensor<xpu, 4, double> wordinputcontext;
		Tensor<xpu, 3, double> input;
		Tensor<xpu, 3, double> hidden;
		vector<Tensor<xpu, 2, double> > pool(_poolmanners);
		vector<Tensor<xpu, 3, double> > poolIndex(_poolmanners);
		Tensor<xpu, 3, double> gateweight, gateweightIndex;
		Tensor<xpu, 2, double> gateweightsum;

		Tensor<xpu, 2, double> poolmerge;
		Tensor<xpu, 2, double> project;
		Tensor<xpu, 2, double> output;
		Tensor<xpu, 2, double> scores;

		//initialize
		int idx = seq_size - 1;

		{
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int word_cnn_iSize =
					(idx == seq_size - 1) ?
							_word_cnn_iSize : _token_representation_size;
			int wordHiddenSize =
					(idx == seq_size - 1) ?
							_wordHiddenSize : _token_representation_size;
			const Feature& feature = example.m_features[idx];
			int word_num = feature.words.size();

			wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
			wordrepresent = NewTensor<xpu>(
					Shape3(word_num, 1, _token_representation_size), 0.0);
			wordinputcontext = NewTensor<xpu>(
					Shape4(word_num, window, 1, _token_representation_size),
					0.0);
			input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
			hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

			for (int idm = 0; idm < _poolmanners; idm++) {
				pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
				poolIndex[idm] = NewTensor<xpu>(
						Shape3(word_num, 1, wordHiddenSize), 0.0);
			}
			gateweight = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize),
					0.0);
			gateweightIndex = NewTensor<xpu>(
					Shape3(word_num, 1, wordHiddenSize), 0.0);

			gateweightsum = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
		}

		poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize),
				0.0);
		project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
		output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
		scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

		//forward propagation
		//input setting, and linear setting
		{
			const Feature& feature = example.m_features[idx];
			int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
			int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

			const vector<int>& words = feature.words;
			int word_num = words.size();
			//linear features should not be dropped out

			for (int idy = 0; idy < word_num; idy++) {
				offset = words[idy];
				wordprime[idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
			}

			//word representation
			for (int idy = 0; idy < word_num; idy++) {
				wordrepresent[idy] += wordprime[idy];
			}

			//word context
			for (int idy = 0; idy < word_num; idy++) {
				wordinputcontext[idy][0] += wordrepresent[idy];
				for (int idc = 1; idc <= curcontext; idc++) {
					if (idy - idc >= 0) {
						wordinputcontext[idy][2 * idc - 1] += wordrepresent[idy
								- idc];
					}
					if (idy + idc < word_num) {
						wordinputcontext[idy][2 * idc] += wordrepresent[idy
								+ idc];
					}
				}
			}

			//word reshape
			for (int idy = 0; idy < word_num; idy++) {
				offset = 0;
				for (int i = 0; i < window; i++) {
					for (int j = 0; j < _token_representation_size; j++) {
						input[idy][0][offset] = wordinputcontext[idy][i][0][j];
						offset++;
					}
				}
			}

			//word convolution
			for (int idy = 0; idy < word_num; idy++) {
				_wordCnnProject.ComputeForwardScore(input[idy], hidden[idy]);
			}

			//word pooling
			if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1)
					|| _remove == 0) {
				avgpool_forward(hidden, pool[0], poolIndex[0]);
			}
			if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2)
					|| _remove == 0) {
				maxpool_forward(hidden, pool[1], poolIndex[1]);
			}
			if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3)
					|| _remove == 0) {
				minpool_forward(hidden, pool[2], poolIndex[2]);
			}

			//gated pooling
			if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4)
					|| _remove == 0) {
				for (int idy = 0; idy < word_num; idy++) {
					_wordGatedPooling.ComputeForwardScore(hidden[idy],
							gateweight[idy]);
				}
				sumpool_forward(gateweight, gateweightsum, gateweightIndex);
				for (int idy = 0; idy < word_num; idy++) {
					poolIndex[3][idy] = gateweight[idy] / gateweightsum;
				}
				for (int idy = 0; idy < word_num; idy++) {
					pool[3] += hidden[idy] * poolIndex[3][idy];
				}
			}
		}

		// sentence
		concat(pool, poolmerge);
		_wordTanhProject.ComputeForwardScore(poolmerge, project);
		_olayer_linear.ComputeForwardScore(project, output);

		int optLabel = -1;
		for (int i = 0; i < _labelSize; ++i) {
			if (example.m_labels[i] >= 0) {
				if (optLabel < 0 || output[0][i] > output[0][optLabel])
					optLabel = i;
			}
		}

		double sum1 = 0.0;
		double sum2 = 0.0;
		double maxScore = output[0][optLabel];
		for (int i = 0; i < _labelSize; ++i) {
			scores[0][i] = -1e10;
			if (example.m_labels[i] >= 0) {
				scores[0][i] = exp(output[0][i] - maxScore);
				if (example.m_labels[i] == 1)
					sum1 += scores[0][i];
				sum2 += scores[0][i];
			}
		}

		double cost = (log(sum2) - log(sum1));

		//release
		{
			FreeSpace(&wordprime);
			FreeSpace(&wordrepresent);
			FreeSpace(&wordinputcontext);
			FreeSpace(&input);
			FreeSpace(&hidden);
			for (int idm = 0; idm < _poolmanners; idm++) {
				FreeSpace(&(pool[idm]));
				FreeSpace(&(poolIndex[idm]));
			}
			FreeSpace(&gateweight);
			FreeSpace(&gateweightIndex);
			FreeSpace(&gateweightsum);
		}
		FreeSpace(&poolmerge);
		FreeSpace(&project);
		FreeSpace(&output);
		FreeSpace(&scores);

		return cost;
	}

	void updateParams(double nnRegular, double adaAlpha, double adaEps) {
		_wordCnnProject.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_wordGatedPooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_wordTanhProject.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		if (_b_wordEmb_finetune) {
			static hash_set<int>::iterator it;
			Tensor<xpu, 1, double> _grad_wordEmb_ij = NewTensor<xpu>(
					Shape1(_wordDim), 0.0);
			Tensor<xpu, 1, double> tmp_normaize_alpha = NewTensor<xpu>(
					Shape1(_wordDim), 0.0);
			Tensor<xpu, 1, double> tmp_alpha = NewTensor<xpu>(Shape1(_wordDim),
					0.0);
			Tensor<xpu, 1, double> _ft_wordEmb_ij = NewTensor<xpu>(
					Shape1(_wordDim), 0.0);

			for (it = _word_indexers.begin(); it != _word_indexers.end();
					++it) {
				int index = *it;
				_grad_wordEmb_ij = _grad_wordEmb[index]
						+ nnRegular * _wordEmb[index] / _ft_wordEmb[index];
				_eg2_wordEmb[index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
				tmp_normaize_alpha = F<nl_sqrt>(_eg2_wordEmb[index] + adaEps);
				tmp_alpha = adaAlpha / tmp_normaize_alpha;
				_ft_wordEmb_ij = _ft_wordEmb[index] * tmp_alpha * nnRegular;
				_ft_wordEmb[index] -= _ft_wordEmb_ij;
				_wordEmb[index] -= tmp_alpha * _grad_wordEmb[index]
						/ _ft_wordEmb[index];
				_grad_wordEmb[index] = 0.0;
			}

			FreeSpace(&_grad_wordEmb_ij);
			FreeSpace(&tmp_normaize_alpha);
			FreeSpace(&tmp_alpha);
			FreeSpace(&_ft_wordEmb_ij);
		}

	}

	void writeModel(const string& outputModelFile){
		LStream lstream(outputModelFile, "w");
		_wordCnnProject.writeModel(lstream);
		_wordGatedPooling.writeModel(lstream);
		_wordTanhProject.writeModel(lstream);
		_olayer_linear.writeModel(lstream);

		SaveBinary(lstream, _wordEmb);
		SaveBinary(lstream, _grad_wordEmb);
		SaveBinary(lstream, _eg2_wordEmb);
		SaveBinary(lstream, _ft_wordEmb);

		WriteBinary(lstream, _word_indexers);
		WriteBinary(lstream, _wordcontext);
		WriteBinary(lstream, _wordSize);
		WriteBinary(lstream, _wordDim);
		WriteBinary(lstream, _b_wordEmb_finetune);
		WriteBinary(lstream, _wordHiddenSize);
		WriteBinary(lstream, _word_cnn_iSize);
		WriteBinary(lstream, _token_representation_size);
		WriteBinary(lstream, _hiddenSize);
		WriteBinary(lstream, _labelSize);
		WriteBinary(lstream, _dropOut);
		WriteBinary(lstream, _remove);
		WriteBinary(lstream, _poolmanners);
		WriteBinary(lstream, topK);

	}

	void loadModel(const string& inputModelFile){
		LStream lstream(inputModelFile, "r");
		_wordCnnProject.loadModel(lstream);
		_wordGatedPooling.loadModel(lstream);
		_wordTanhProject.loadModel(lstream);
		_olayer_linear.loadModel(lstream);

		LoadBinary(lstream, &_wordEmb, false);
		LoadBinary(lstream, &_grad_wordEmb, false);
		LoadBinary(lstream, &_eg2_wordEmb, false);
		LoadBinary(lstream, &_ft_wordEmb, false);

		ReadBinary(lstream, _word_indexers);
		ReadBinary(lstream, _wordcontext);
		ReadBinary(lstream, _wordSize);
		ReadBinary(lstream, _wordDim);
		ReadBinary(lstream, _b_wordEmb_finetune);
		ReadBinary(lstream, _wordHiddenSize);
		ReadBinary(lstream, _word_cnn_iSize);
		ReadBinary(lstream, _token_representation_size);
		ReadBinary(lstream, _hiddenSize);
		ReadBinary(lstream, _labelSize);
		ReadBinary(lstream, _dropOut);
		ReadBinary(lstream, _remove);
		ReadBinary(lstream, _poolmanners);
		ReadBinary(lstream, topK);
	}

	void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd,
			const Tensor<xpu, 2, double>& gradWd, const string& mark,
			int iter) {
		int charseed = mark.length();
		for (int i = 0; i < mark.length(); i++) {
			charseed = (int) (mark[i]) * 5 + charseed;
		}
		srand(iter + charseed);
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		for (int i = 0; i < Wd.size(0); ++i)
			idRows.push_back(i);
		for (int idx = 0; idx < Wd.size(1); idx++)
			idCols.push_back(idx);

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		int check_i = idRows[0], check_j = idCols[0];

		double orginValue = Wd[check_i][check_j];

		Wd[check_i][check_j] = orginValue + 0.001;
		double lossAdd = 0.0;
		for (int i = 0; i < examples.size(); i++) {
			Example oneExam = examples[i];
			lossAdd += computeScore(oneExam);
		}

		Wd[check_i][check_j] = orginValue - 0.001;
		double lossPlus = 0.0;
		for (int i = 0; i < examples.size(); i++) {
			Example oneExam = examples[i];
			lossPlus += computeScore(oneExam);
		}

		double mockGrad = (lossAdd - lossPlus) / 0.002;
		mockGrad = mockGrad / examples.size();
		double computeGrad = gradWd[check_i][check_j];

		printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter,
				mark.c_str(), check_i, check_j);
		printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad,
				computeGrad);

		Wd[check_i][check_j] = orginValue;
	}

	void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd,
			const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter,
			const hash_set<int>& indexes, bool bRow = true) {
		int charseed = mark.length();
		for (int i = 0; i < mark.length(); i++) {
			charseed = (int) (mark[i]) * 5 + charseed;
		}
		srand(iter + charseed);
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		static hash_set<int>::iterator it;
		if (bRow) {
			for (it = indexes.begin(); it != indexes.end(); ++it)
				idRows.push_back(*it);
			for (int idx = 0; idx < Wd.size(1); idx++)
				idCols.push_back(idx);
		} else {
			for (it = indexes.begin(); it != indexes.end(); ++it)
				idCols.push_back(*it);
			for (int idx = 0; idx < Wd.size(0); idx++)
				idRows.push_back(idx);
		}

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		int check_i = idRows[0], check_j = idCols[0];

		double orginValue = Wd[check_i][check_j];

		Wd[check_i][check_j] = orginValue + 0.001;
		double lossAdd = 0.0;
		for (int i = 0; i < examples.size(); i++) {
			Example oneExam = examples[i];
			lossAdd += computeScore(oneExam);
		}

		Wd[check_i][check_j] = orginValue - 0.001;
		double lossPlus = 0.0;
		for (int i = 0; i < examples.size(); i++) {
			Example oneExam = examples[i];
			lossPlus += computeScore(oneExam);
		}

		double mockGrad = (lossAdd - lossPlus) / 0.002;
		mockGrad = mockGrad / examples.size();
		double computeGrad = gradWd[check_i][check_j];

		printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter,
				mark.c_str(), check_i, check_j);
		printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad,
				computeGrad);

		Wd[check_i][check_j] = orginValue;

	}

	void checkgrads(const vector<Example>& examples, int iter) {

		checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW,
				"_olayer_linear._W", iter);
		checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb,
				"_olayer_linear._b", iter);

		checkgrad(examples, _wordGatedPooling._W, _wordGatedPooling._gradW,
				"_gated_pooling._W", iter);
		checkgrad(examples, _wordGatedPooling._b, _wordGatedPooling._gradb,
				"_gated_pooling._b", iter);

		checkgrad(examples, _wordTanhProject._W, _wordTanhProject._gradW,
				"_tanh_project._W", iter);
		checkgrad(examples, _wordTanhProject._b, _wordTanhProject._gradb,
				"_tanh_project._b", iter);

		checkgrad(examples, _wordCnnProject._W, _wordCnnProject._gradW,
				"_cnn_project._W", iter);
		checkgrad(examples, _wordCnnProject._b, _wordCnnProject._gradb,
				"_cnn_project._b", iter);

		if (_word_indexers.size() > 0)
			checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter,
					_word_indexers);

	}

public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(double dropOut) {
		_dropOut = dropOut;
	}

	inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
		_b_wordEmb_finetune = b_wordEmb_finetune;
	}

	inline void resetRemove(int remove) {
		_remove = remove;
	}

};

#endif /* SRC_CNNWordClassifier_H_ */
