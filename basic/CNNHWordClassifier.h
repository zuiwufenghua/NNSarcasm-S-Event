/*
 * CNNHWordClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_CNNHWordClassifier_H_
#define SRC_CNNHWordClassifier_H_

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
class CNNHWordClassifier {
public:
  CNNHWordClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~CNNHWordClassifier() {

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

  UniHidderLayer<xpu> _cnn_project;
  UniHidderLayer<xpu> _gated_sent_pooling;
  UniHidderLayer<xpu> _gated_history_pooling;
  UniHidderLayer<xpu> _tanh_project;
  UniHidderLayer<xpu> _olayer_linear;

  int _labelSize;

  Metric _eval;

  double _dropOut;

  int _remove; // 1, avg, 2, max, 3 min, 4 gated

  int _poolmanners;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int labelSize, int wordHiddenSize, int hiddenSize) {
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

    _cnn_project.initial(_wordHiddenSize, _word_cnn_iSize, true, 20, 0);
    _gated_sent_pooling.initial(_wordHiddenSize, _wordHiddenSize, true, 30, 3);
    _gated_history_pooling.initial(_token_representation_size, _token_representation_size, true, 40, 3);
    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size, true, 50, 0);
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
    _cnn_project.release();
    _gated_sent_pooling.release();
    _gated_history_pooling.release();
    _tanh_project.release();
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

      //word CNN


      //sentence CNN

      int seq_size = example.m_features.size();
     

      Tensor<xpu, 3, double> wordprime[seq_size], wordprimeLoss[seq_size], wordprimeMask[seq_size];
      Tensor<xpu, 3, double> wordrepresent[seq_size], wordrepresentLoss[seq_size];
      Tensor<xpu, 4, double> wordinputcontext[seq_size], wordinputcontextLoss[seq_size];
      Tensor<xpu, 3, double> input[seq_size], inputLoss[seq_size];
      Tensor<xpu, 3, double> hidden[seq_size], hiddenLoss[seq_size], hiddenLossTmp[seq_size];
      vector<Tensor<xpu, 2, double> > pool(seq_size * _poolmanners), poolLoss(seq_size * _poolmanners);
      vector<Tensor<xpu, 3, double> > poolIndex(seq_size * _poolmanners), poolIndexLoss(seq_size * _poolmanners);
      vector<Tensor<xpu, 3, double> > gateweight(seq_size), gateweightLoss(seq_size), gateweightIndex(seq_size);
      vector<Tensor<xpu, 2, double> > gateweightsum(seq_size), gateweightsumLoss(seq_size);

      Tensor<xpu, 2, double> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, double> project, projectLoss;
      Tensor<xpu, 2, double> output, outputLoss;
      Tensor<xpu, 2, double> scores;

      //initialize

      for (int idx = 0; idx < seq_size; idx++) {
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();
        int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
        int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

        wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
        wordprimeMask[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 1.0);
        wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
        wordinputcontext[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
        wordinputcontextLoss[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
        input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
        hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        hiddenLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        hiddenLossTmp[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

        offset = idx * _poolmanners;
        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
          poolLoss[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
          poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          poolIndexLoss[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        }
        gateweight[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        gateweightIndex[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        gateweightLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

        gateweightsum[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        gateweightsumLoss[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
      }

      poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
      project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
      projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

        const vector<int>& words = feature.words;
        int word_num = words.size();
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        for (int idy = 0; idy < word_num; idy++) {
          offset = words[idy];
          wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
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
          wordprime[idx][idy] = wordprime[idx][idy] * wordprimeMask[idx][idy];
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresent[idx][idy] +=  wordprime[idx][idy];
        }

        //word context
        for (int idy = 0; idy < word_num; idy++) {
          wordinputcontext[idx][idy][0] += wordrepresent[idx][idy];
          for (int idc = 1; idc <= curcontext; idc++) {
            if (idy - idc >= 0) {
              wordinputcontext[idx][idy][2 * idc - 1] += wordrepresent[idx][idy - idc];
            }
            if (idy + idc < word_num) {
              wordinputcontext[idx][idy][2 * idc] += wordrepresent[idx][idy + idc];
            }
          }
        }

        //word reshape, concatenate to one-dim vector
        for (int idy = 0; idy < word_num; idy++) {
          offset = 0;
          for (int i = 0; i < window; i++) {
            for (int j = 0; j < _token_representation_size; j++) {
              input[idx][idy][0][offset] = wordinputcontext[idx][idy][i][0][j];
              offset++;
            }
          }
        }

        //word convolution
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1)
            _cnn_project.ComputeForwardScore(input[idx][idy], hidden[idx][idy]);
          else
            hidden[idx][idy] += input[idx][idy];
        }

        //word pooling
        offset = idx * _poolmanners;
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden[idx], pool[offset+1], poolIndex[offset+1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden[idx], pool[offset+2], poolIndex[offset+2]);
        }

        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            if(idx == seq_size - 1)_gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
            else _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          }
          sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
          for (int idy = 0; idy < word_num; idy++) {
            poolIndex[offset+3][idy] = gateweight[idx][idy] / gateweightsum[idx];
          }
          for (int idy = 0; idy < word_num; idy++) {
            pool[offset+3] += hidden[idx][idy] * poolIndex[offset+3][idy];
          }
        }

      }

      // sentence
      offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
      concat(pool, poolmerge, offset);
      _tanh_project.ComputeForwardScore(poolmerge, project);

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
      cost += (log(sum2) - log(sum1)) / example_num;
      if (example.m_labels[optLabel] == 1)
        _eval.correct_label_count++;
      _eval.overall_label_count++;

      for (int i = 0; i < _labelSize; ++i) {
        outputLoss[0][i] = 0.0;
        if (example.m_labels[i] >= 0) {
          outputLoss[0][i] = (scores[0][i] / sum2 - example.m_labels[i]) / example_num;
        }
      }

      // loss backward propagation
      //sentence
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
      _tanh_project.ComputeBackwardLoss(poolmerge, project, projectLoss, poolmergeLoss);

      offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
      unconcat(poolLoss, poolmergeLoss, offset);

      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

        const vector<int>& words = feature.words;
        int word_num = words.size();

        //word pooling
        offset = idx * _poolmanners;
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          hiddenLossTmp[idx] = 0.0;
          pool_backward(poolLoss[offset], poolIndex[offset], hiddenLossTmp[idx]);
          hiddenLoss[idx] = hiddenLoss[idx] + hiddenLossTmp[idx];
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          hiddenLossTmp[idx] = 0.0;
          pool_backward(poolLoss[offset+1], poolIndex[offset+1], hiddenLossTmp[idx]);
          hiddenLoss[idx] = hiddenLoss[idx] + hiddenLossTmp[idx];
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          hiddenLossTmp[idx] = 0.0;
          pool_backward(poolLoss[offset+2], poolIndex[offset+2], hiddenLossTmp[idx]);
          hiddenLoss[idx] = hiddenLoss[idx] + hiddenLossTmp[idx];
        }

        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            poolIndexLoss[offset+3][idy] = poolLoss[offset+3] * hidden[idx][idy];
            hiddenLoss[idx][idy] += poolLoss[offset+3] * poolIndex[offset+3][idy];
          }

          for (int idy = 0; idy < word_num; idy++) {
            gateweightLoss[idx][idy] += poolIndexLoss[offset+3][idy] / gateweightsum[idx];
            gateweightsumLoss[idx] -= poolIndexLoss[offset+3][idy] * gateweight[idx][idy] / gateweightsum[idx] / gateweightsum[idx];
          }

          pool_backward(gateweightsumLoss[idx], gateweightIndex[idx], gateweightLoss[idx]);

          hiddenLossTmp[idx] = 0.0;
          for (int idy = 0; idy < word_num; idy++) {
            if(idx == seq_size - 1)_gated_sent_pooling.ComputeBackwardLoss(hidden[idx][idy], gateweight[idx][idy], gateweightLoss[idx][idy], hiddenLossTmp[idx][idy]);
            else _gated_history_pooling.ComputeBackwardLoss(hidden[idx][idy], gateweight[idx][idy], gateweightLoss[idx][idy], hiddenLossTmp[idx][idy]);
            hiddenLoss[idx][idy] += hiddenLossTmp[idx][idy];
          }
        }

        //word convolution
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1)
            _cnn_project.ComputeBackwardLoss(input[idx][idy], hidden[idx][idy], hiddenLoss[idx][idy], inputLoss[idx][idy]);
          else
            inputLoss[idx][idy] += hiddenLoss[idx][idy];
        }

        //word reshape
        for (int idy = 0; idy < word_num; idy++) {
          offset = 0;
          for (int i = 0; i < window; i++) {
            for (int j = 0; j < _token_representation_size; j++) {
              wordinputcontextLoss[idx][idy][i][0][j] = inputLoss[idx][idy][0][offset];
              offset++;
            }
          }
        }

        //word context
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresentLoss[idx][idy] += wordinputcontextLoss[idx][idy][0];
          for (int idc = 1; idc <= curcontext; idc++) {
            if (idy - idc >= 0) {
              wordrepresentLoss[idx][idy - idc] += wordinputcontextLoss[idx][idy][2 * idc - 1];
            }
            if (idy + idc < word_num) {
              wordrepresentLoss[idx][idy + idc] += wordinputcontextLoss[idx][idy][2 * idc];
            }
          }
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          wordprimeLoss[idx][idy] += wordrepresentLoss[idx][idy];
        }

        //word dropout
        for (int idy = 0; idy < word_num; idy++) {
          wordprimeLoss[idx][idy] = wordprimeLoss[idx][idy] * wordprimeMask[idx][idy];
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
      for (int idx = 0; idx < seq_size; idx++) {
        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));
        FreeSpace(&(wordinputcontext[idx]));
        FreeSpace(&(wordinputcontextLoss[idx]));
        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(hidden[idx]));
        FreeSpace(&(hiddenLoss[idx]));
        FreeSpace(&(hiddenLossTmp[idx]));

        offset = idx * _poolmanners;
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[offset + idm]));
          FreeSpace(&(poolLoss[offset + idm]));
          FreeSpace(&(poolIndex[offset + idm]));
          FreeSpace(&(poolIndexLoss[offset + idm]));
        }
        FreeSpace(&(gateweight[idx]));
        FreeSpace(&(gateweightIndex[idx]));
        FreeSpace(&(gateweightLoss[idx]));

        FreeSpace(&(gateweightsum[idx]));
        FreeSpace(&(gateweightsumLoss[idx]));
      }
      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&project);
      FreeSpace(&projectLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);
      FreeSpace(&scores);

    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  inline double processText(const vector<Example>& examples, int iter) {
      _eval.reset();
      _word_indexers.clear();

      int example_num = examples.size();
      double cost = 0.0;
      int offset = 0;
      for (int count = 0; count < example_num; count++) {
        const Example& example = examples[count];

        int seq_size = example.m_features.size();

        //word related cache
        Tensor<xpu, 3, double> wordPrime[seq_size], wordPrimeLoss[seq_size], wordPrimeMask[seq_size];
        Tensor<xpu, 3, double> wordRepresent[seq_size], wordRepresentLoss[seq_size];
        Tensor<xpu, 4, double> wordInputContext[seq_size], wordInputContextLoss[seq_size];
        Tensor<xpu, 3, double> wordInput[seq_size], wordInputLoss[seq_size];
        Tensor<xpu, 3, double> wordHidden[seq_size], wordHiddenLoss[seq_size], wordHiddenLossTmp[seq_size];
        vector<Tensor<xpu, 2, double> > wordPool(seq_size * _poolmanners), wordPoolLoss(seq_size * _poolmanners);
        vector<Tensor<xpu, 3, double> > wordPoolIndex(seq_size * _poolmanners), wordPoolIndexLoss(seq_size * _poolmanners);
        vector<Tensor<xpu, 3, double> > wordGateWeight(seq_size), wordGateWeightLoss(seq_size), wordGateweightIndex(seq_size);
        vector<Tensor<xpu, 2, double> > wordGateWeightSum(seq_size), wordGateWeightSumLoss(seq_size);

        Tensor<xpu, 2, double> wordPoolmerge, wordPoolmergeLoss;
        Tensor<xpu, 2, double> wordProject, wordProjectLoss;
        Tensor<xpu, 2, double> wordOutput, wordOutputLoss;
        Tensor<xpu, 2, double> wordScores;

        //initialize

        for (int idx = 0; idx < seq_size; idx++) {
          int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
          const Feature& feature = example.m_features[idx];
          int word_num = feature.words.size();
          int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
          int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

          wordPrime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
          wordPrimeLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
          wordPrimeMask[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 1.0);
          wordRepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
          wordRepresentLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
          wordInputContext[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
          wordInputContextLoss[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
          wordInput[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
          wordInputLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
          wordHidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          wordHiddenLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          wordHiddenLossTmp[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

          offset = idx * _poolmanners;
          for (int idm = 0; idm < _poolmanners; idm++) {
            wordPool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
            wordPoolLoss[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
            wordPoolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
            wordPoolIndexLoss[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          }
          wordGateWeight[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          wordGateweightIndex[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          wordGateWeightLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

          wordGateWeightSum[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
          wordGateWeightSumLoss[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        }

        wordPoolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
        wordPoolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
        wordProject = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
        wordProjectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
        wordOutput = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        wordOutputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        wordScores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

        //forward propagation
        //input setting, and linear setting
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
          int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

          const vector<int>& words = feature.words;
          int word_num = words.size();
          //linear features should not be dropped out

          srand(iter * example_num + count * seq_size + idx);

          for (int idy = 0; idy < word_num; idy++) {
            offset = words[idy];
            wordPrime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
          }

          //word dropout
          for (int idy = 0; idy < word_num; idy++) {
            for (int j = 0; j < _wordDim; j++) {
              if (1.0 * rand() / RAND_MAX >= _dropOut) {
                wordPrimeMask[idx][idy][0][j] = 1.0;
              } else {
                wordPrimeMask[idx][idy][0][j] = 0.0;
              }
            }
            wordPrime[idx][idy] = wordPrime[idx][idy] * wordPrimeMask[idx][idy];
          }

          //word representation
          for (int idy = 0; idy < word_num; idy++) {
            wordRepresent[idx][idy] +=  wordPrime[idx][idy];
          }

          //word context
          for (int idy = 0; idy < word_num; idy++) {
            wordInputContext[idx][idy][0] += wordRepresent[idx][idy];
            for (int idc = 1; idc <= curcontext; idc++) {
              if (idy - idc >= 0) {
                wordInputContext[idx][idy][2 * idc - 1] += wordRepresent[idx][idy - idc];
              }
              if (idy + idc < word_num) {
                wordInputContext[idx][idy][2 * idc] += wordRepresent[idx][idy + idc];
              }
            }
          }

          //word reshape
          for (int idy = 0; idy < word_num; idy++) {
            offset = 0;
            for (int i = 0; i < window; i++) {
              for (int j = 0; j < _token_representation_size; j++) {
                wordInput[idx][idy][0][offset] = wordInputContext[idx][idy][i][0][j];
                offset++;
              }
            }
          }

          //word convolution
          for (int idy = 0; idy < word_num; idy++) {
            if (idx == seq_size - 1)
              _cnn_project.ComputeForwardScore(wordInput[idx][idy], wordHidden[idx][idy]);
            else
              wordHidden[idx][idy] += wordInput[idx][idy];
          }

          //word pooling
          offset = idx * _poolmanners;
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            avgpool_forward(wordHidden[idx], wordPool[offset], wordPoolIndex[offset]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            maxpool_forward(wordHidden[idx], wordPool[offset+1], wordPoolIndex[offset+1]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            minpool_forward(wordHidden[idx], wordPool[offset+2], wordPoolIndex[offset+2]);
          }

          //gated pooling
          if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
            for (int idy = 0; idy < word_num; idy++) {
              if(idx == seq_size - 1)_gated_sent_pooling.ComputeForwardScore(wordHidden[idx][idy], wordGateWeight[idx][idy]);
              else _gated_history_pooling.ComputeForwardScore(wordHidden[idx][idy], wordGateWeight[idx][idy]);
            }
            sumpool_forward(wordGateWeight[idx], wordGateWeightSum[idx], wordGateweightIndex[idx]);
            for (int idy = 0; idy < word_num; idy++) {
              wordPoolIndex[offset+3][idy] = wordGateWeight[idx][idy] / wordGateWeightSum[idx];
            }
            for (int idy = 0; idy < word_num; idy++) {
              wordPool[offset+3] += wordHidden[idx][idy] * wordPoolIndex[offset+3][idy];
            }
          }

        }

        // sentence
        offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
        concat(wordPool, wordPoolmerge, offset);
        _tanh_project.ComputeForwardScore(wordPoolmerge, wordProject);
        _olayer_linear.ComputeForwardScore(wordProject, wordOutput);

        int optLabel = -1;
        for (int i = 0; i < _labelSize; ++i) {
          if (example.m_labels[i] >= 0) {
            if (optLabel < 0 || wordOutput[0][i] > wordOutput[0][optLabel])
              optLabel = i;
          }
        }

        double sum1 = 0.0;
        double sum2 = 0.0;
        double maxScore = wordOutput[0][optLabel];
        for (int i = 0; i < _labelSize; ++i) {
          wordScores[0][i] = -1e10;
          if (example.m_labels[i] >= 0) {
            wordScores[0][i] = exp(wordOutput[0][i] - maxScore);
            if (example.m_labels[i] == 1)
              sum1 += wordScores[0][i];
            sum2 += wordScores[0][i];
          }
        }
        cost += (log(sum2) - log(sum1)) / example_num;
        if (example.m_labels[optLabel] == 1)
          _eval.correct_label_count++;
        _eval.overall_label_count++;

        for (int i = 0; i < _labelSize; ++i) {
          wordOutputLoss[0][i] = 0.0;
          if (example.m_labels[i] >= 0) {
            wordOutputLoss[0][i] = (wordScores[0][i] / sum2 - example.m_labels[i]) / example_num;
          }
        }

        // loss backward propagation
        //sentence
        _olayer_linear.ComputeBackwardLoss(wordProject, wordOutput, wordOutputLoss, wordProjectLoss);
        _tanh_project.ComputeBackwardLoss(wordPoolmerge, wordProject, wordProjectLoss, wordPoolmergeLoss);

        offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
        unconcat(wordPoolLoss, wordPoolmergeLoss, offset);

        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
          int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

          const vector<int>& words = feature.words;
          int word_num = words.size();

          //word pooling
          offset = idx * _poolmanners;
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            wordHiddenLossTmp[idx] = 0.0;
            pool_backward(wordPoolLoss[offset], wordPoolIndex[offset], wordHiddenLossTmp[idx]);
            wordHiddenLoss[idx] = wordHiddenLoss[idx] + wordHiddenLossTmp[idx];
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            wordHiddenLossTmp[idx] = 0.0;
            pool_backward(wordPoolLoss[offset+1], wordPoolIndex[offset+1], wordHiddenLossTmp[idx]);
            wordHiddenLoss[idx] = wordHiddenLoss[idx] + wordHiddenLossTmp[idx];
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            wordHiddenLossTmp[idx] = 0.0;
            pool_backward(wordPoolLoss[offset+2], wordPoolIndex[offset+2], wordHiddenLossTmp[idx]);
            wordHiddenLoss[idx] = wordHiddenLoss[idx] + wordHiddenLossTmp[idx];
          }

          //gated pooling
          if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
            for (int idy = 0; idy < word_num; idy++) {
              wordPoolIndexLoss[offset+3][idy] = wordPoolLoss[offset+3] * wordHidden[idx][idy];
              wordHiddenLoss[idx][idy] += wordPoolLoss[offset+3] * wordPoolIndex[offset+3][idy];
            }

            for (int idy = 0; idy < word_num; idy++) {
              wordGateWeightLoss[idx][idy] += wordPoolIndexLoss[offset+3][idy] / wordGateWeightSum[idx];
              wordGateWeightSumLoss[idx] -= wordPoolIndexLoss[offset+3][idy] * wordGateWeight[idx][idy] / wordGateWeightSum[idx] / wordGateWeightSum[idx];
            }

            pool_backward(wordGateWeightSumLoss[idx], wordGateweightIndex[idx], wordGateWeightLoss[idx]);

            wordHiddenLossTmp[idx] = 0.0;
            for (int idy = 0; idy < word_num; idy++) {
              if(idx == seq_size - 1)_gated_sent_pooling.ComputeBackwardLoss(wordHidden[idx][idy], wordGateWeight[idx][idy], wordGateWeightLoss[idx][idy], wordHiddenLossTmp[idx][idy]);
              else _gated_history_pooling.ComputeBackwardLoss(wordHidden[idx][idy], wordGateWeight[idx][idy], wordGateWeightLoss[idx][idy], wordHiddenLossTmp[idx][idy]);
              wordHiddenLoss[idx][idy] += wordHiddenLossTmp[idx][idy];
            }
          }

          //word convolution
          for (int idy = 0; idy < word_num; idy++) {
            if (idx == seq_size - 1)
              _cnn_project.ComputeBackwardLoss(wordInput[idx][idy], wordHidden[idx][idy], wordHiddenLoss[idx][idy], wordInputLoss[idx][idy]);
            else
              wordInputLoss[idx][idy] += wordHiddenLoss[idx][idy];
          }

          //word reshape
          for (int idy = 0; idy < word_num; idy++) {
            offset = 0;
            for (int i = 0; i < window; i++) {
              for (int j = 0; j < _token_representation_size; j++) {
                wordInputContextLoss[idx][idy][i][0][j] = wordInputLoss[idx][idy][0][offset];
                offset++;
              }
            }
          }

          //word context
          for (int idy = 0; idy < word_num; idy++) {
            wordRepresentLoss[idx][idy] += wordInputContextLoss[idx][idy][0];
            for (int idc = 1; idc <= curcontext; idc++) {
              if (idy - idc >= 0) {
                wordRepresentLoss[idx][idy - idc] += wordInputContextLoss[idx][idy][2 * idc - 1];
              }
              if (idy + idc < word_num) {
                wordRepresentLoss[idx][idy + idc] += wordInputContextLoss[idx][idy][2 * idc];
              }
            }
          }

          //word representation
          for (int idy = 0; idy < word_num; idy++) {
            wordPrimeLoss[idx][idy] += wordRepresentLoss[idx][idy];
          }

          //word dropout
          for (int idy = 0; idy < word_num; idy++) {
            wordPrimeLoss[idx][idy] = wordPrimeLoss[idx][idy] * wordPrimeMask[idx][idy];
          }

          //word finetune
          if (_b_wordEmb_finetune) {
            for (int idy = 0; idy < word_num; idy++) {
              offset = words[idy];
              _grad_wordEmb[offset] += wordPrimeLoss[idx][idy][0];
              _word_indexers.insert(offset);
            }
          }
        }

        //release
        for (int idx = 0; idx < seq_size; idx++) {
          FreeSpace(&(wordPrime[idx]));
          FreeSpace(&(wordPrimeLoss[idx]));
          FreeSpace(&(wordPrimeMask[idx]));
          FreeSpace(&(wordRepresent[idx]));
          FreeSpace(&(wordRepresentLoss[idx]));
          FreeSpace(&(wordInputContext[idx]));
          FreeSpace(&(wordInputContextLoss[idx]));
          FreeSpace(&(wordInput[idx]));
          FreeSpace(&(wordInputLoss[idx]));
          FreeSpace(&(wordHidden[idx]));
          FreeSpace(&(wordHiddenLoss[idx]));
          FreeSpace(&(wordHiddenLossTmp[idx]));

          offset = idx * _poolmanners;
          for (int idm = 0; idm < _poolmanners; idm++) {
            FreeSpace(&(wordPool[offset + idm]));
            FreeSpace(&(wordPoolLoss[offset + idm]));
            FreeSpace(&(wordPoolIndex[offset + idm]));
            FreeSpace(&(wordPoolIndexLoss[offset + idm]));
          }
          FreeSpace(&(wordGateWeight[idx]));
          FreeSpace(&(wordGateweightIndex[idx]));
          FreeSpace(&(wordGateWeightLoss[idx]));

          FreeSpace(&(wordGateWeightSum[idx]));
          FreeSpace(&(wordGateWeightSumLoss[idx]));
        }
        FreeSpace(&wordPoolmerge);
        FreeSpace(&wordPoolmergeLoss);
        FreeSpace(&wordProject);
        FreeSpace(&wordProjectLoss);
        FreeSpace(&wordOutput);
        FreeSpace(&wordOutputLoss);
        FreeSpace(&wordScores);

      }

      if (_eval.getAccuracy() < 0) {
        std::cout << "strange" << std::endl;
      }

      return cost;
    }

  int predict(const vector<Feature>& features, vector<double>& results) {
    int seq_size = features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }


    Tensor<xpu, 3, double> wordprime[seq_size];
    Tensor<xpu, 3, double> wordrepresent[seq_size];
    Tensor<xpu, 4, double> wordinputcontext[seq_size];
    Tensor<xpu, 3, double> input[seq_size];
    Tensor<xpu, 3, double> hidden[seq_size];
    vector<Tensor<xpu, 2, double> > pool(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, double> > poolIndex(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, double> > gateweight(seq_size), gateweightIndex(seq_size);
    vector<Tensor<xpu, 2, double> > gateweightsum(seq_size);


    Tensor<xpu, 2, double> poolmerge;
    Tensor<xpu, 2, double> project;
    Tensor<xpu, 2, double> output;
    Tensor<xpu, 2, double> scores;

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;
      const Feature& feature = features[idx];
      int word_num = feature.words.size();

      wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
      wordinputcontext[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
      input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
      hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      }
      gateweight[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightIndex[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightsum[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      int word_num = words.size();
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }
      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        wordrepresent[idx][idy] +=  wordprime[idx][idy];
      }

      //word context
      for (int idy = 0; idy < word_num; idy++) {
        wordinputcontext[idx][idy][0] += wordrepresent[idx][idy];
        for (int idc = 1; idc <= curcontext; idc++) {
          if (idy - idc >= 0) {
            wordinputcontext[idx][idy][2 * idc - 1] += wordrepresent[idx][idy - idc];
          }
          if (idy + idc < word_num) {
            wordinputcontext[idx][idy][2 * idc] += wordrepresent[idx][idy + idc];
          }
        }
      }

      //word reshape
      for (int idy = 0; idy < word_num; idy++) {
        offset = 0;
        for (int i = 0; i < window; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            input[idx][idy][0][offset] = wordinputcontext[idx][idy][i][0][j];
            offset++;
          }
        }
      }

      //word convolution
      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1)
          _cnn_project.ComputeForwardScore(input[idx][idy], hidden[idx][idy]);
        else
          hidden[idx][idy] += input[idx][idy];
      }

      //word pooling
      offset = idx * _poolmanners;
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden[idx], pool[offset+1], poolIndex[offset+1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset+2], poolIndex[offset+2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          if(idx == seq_size - 1)_gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          else _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
        }
        sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
        for (int idy = 0; idy < word_num; idy++) {
          poolIndex[offset+3][idy] = gateweight[idx][idy] / gateweightsum[idx];
        }
        for (int idy = 0; idy < word_num; idy++) {
          pool[offset+3] += hidden[idx][idy] * poolIndex[offset+3][idy];
        }
      }
    }

    // sentence
    offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
    concat(pool, poolmerge, offset);
    _tanh_project.ComputeForwardScore(poolmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    int optLabel = 0;
    for (int i = 1; i < _labelSize; ++i) {
      if (output[0][i] > output[0][optLabel])
        optLabel = i;
    }

    double sum = 0.0;
    double maxScore = output[0][optLabel];
    for (int i = 0; i < _labelSize; ++i) {
      scores[0][i] = exp(output[0][i] - maxScore);
      sum += scores[0][i];
    }

    results.resize(_labelSize);
    for (int i = 0; i < _labelSize; ++i) {
      results[i] = scores[0][i] / sum;
    }

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(wordinputcontext[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(hidden[idx]));

      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[offset + idm]));
        FreeSpace(&(poolIndex[offset + idm]));
      }
      FreeSpace(&(gateweight[idx]));
      FreeSpace(&(gateweightIndex[idx]));
      FreeSpace(&(gateweightsum[idx]));
    }
    FreeSpace(&poolmerge);
    FreeSpace(&project);
    FreeSpace(&output);
    FreeSpace(&scores);

    return optLabel;
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }

    Tensor<xpu, 3, double> wordprime[seq_size];
    Tensor<xpu, 3, double> wordrepresent[seq_size];
    Tensor<xpu, 4, double> wordinputcontext[seq_size];
    Tensor<xpu, 3, double> input[seq_size];
    Tensor<xpu, 3, double> hidden[seq_size];
    vector<Tensor<xpu, 2, double> > pool(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, double> > poolIndex(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, double> > gateweight(seq_size), gateweightIndex(seq_size);
    vector<Tensor<xpu, 2, double> > gateweightsum(seq_size);

    Tensor<xpu, 2, double> poolmerge;
    Tensor<xpu, 2, double> project;
    Tensor<xpu, 2, double> output;
    Tensor<xpu, 2, double> scores;

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();

      wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
      wordinputcontext[idx] = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
      input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
      hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      }
      gateweight[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightIndex[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightsum[idx] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), 0.0);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      int word_num = words.size();
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        wordrepresent[idx][idy] +=  wordprime[idx][idy];
      }

      //word context
      for (int idy = 0; idy < word_num; idy++) {
        wordinputcontext[idx][idy][0] += wordrepresent[idx][idy];
        for (int idc = 1; idc <= curcontext; idc++) {
          if (idy - idc >= 0) {
            wordinputcontext[idx][idy][2 * idc - 1] += wordrepresent[idx][idy - idc];
          }
          if (idy + idc < word_num) {
            wordinputcontext[idx][idy][2 * idc] += wordrepresent[idx][idy + idc];
          }
        }
      }

      //word reshape
      for (int idy = 0; idy < word_num; idy++) {
        offset = 0;
        for (int i = 0; i < window; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            input[idx][idy][0][offset] = wordinputcontext[idx][idy][i][0][j];
            offset++;
          }
        }
      }

      //word convolution
      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1)
          _cnn_project.ComputeForwardScore(input[idx][idy], hidden[idx][idy]);
        else
          hidden[idx][idy] += input[idx][idy];
      }

      //word pooling
      offset = idx * _poolmanners;
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden[idx], pool[offset+1], poolIndex[offset+1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset+2], poolIndex[offset+2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          if(idx == seq_size - 1)_gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          else _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
        }
        sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
        for (int idy = 0; idy < word_num; idy++) {
          poolIndex[offset+3][idy] = gateweight[idx][idy] / gateweightsum[idx];
        }
        for (int idy = 0; idy < word_num; idy++) {
          pool[offset+3] += hidden[idx][idy] * poolIndex[offset+3][idy];
        }
      }
    }

    // sentence
    offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
    concat(pool, poolmerge, offset);

    _tanh_project.ComputeForwardScore(poolmerge, project);
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
    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(wordinputcontext[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(hidden[idx]));
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[offset + idm]));
        FreeSpace(&(poolIndex[offset + idm]));
      }
      FreeSpace(&(gateweight[idx]));
      FreeSpace(&(gateweightIndex[idx]));
      FreeSpace(&(gateweightsum[idx]));
    }
    FreeSpace(&poolmerge);
    FreeSpace(&project);
    FreeSpace(&output);
    FreeSpace(&scores);

    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gated_sent_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gated_history_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    if (_b_wordEmb_finetune) {
      static hash_set<int>::iterator it;
      Tensor<xpu, 1, double> _grad_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_normaize_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> _ft_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);

      for (it = _word_indexers.begin(); it != _word_indexers.end(); ++it) {
        int index = *it;
        _grad_wordEmb_ij = _grad_wordEmb[index] + nnRegular * _wordEmb[index] / _ft_wordEmb[index];
        _eg2_wordEmb[index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
        tmp_normaize_alpha = F<nl_sqrt>(_eg2_wordEmb[index] + adaEps);
        tmp_alpha = adaAlpha / tmp_normaize_alpha;
        _ft_wordEmb_ij = _ft_wordEmb[index] * tmp_alpha * nnRegular;
        _ft_wordEmb[index] -= _ft_wordEmb_ij;
        _wordEmb[index] -= tmp_alpha * _grad_wordEmb[index] / _ft_wordEmb[index];
        _grad_wordEmb[index] = 0.0;
      }

      FreeSpace(&_grad_wordEmb_ij);
      FreeSpace(&tmp_normaize_alpha);
      FreeSpace(&tmp_alpha);
      FreeSpace(&_ft_wordEmb_ij);
    }

  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd, const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter) {
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

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd, const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter,
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

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _gated_sent_pooling._W, _gated_sent_pooling._gradW, "_gated_sent_pooling._W", iter);
    checkgrad(examples, _gated_sent_pooling._b, _gated_sent_pooling._gradb, "_gated_sent_pooling._b", iter);

    checkgrad(examples, _gated_history_pooling._W, _gated_history_pooling._gradW, "_gated_history_pooling._W", iter);
    checkgrad(examples, _gated_history_pooling._b, _gated_history_pooling._gradb, "_gated_history_pooling._b", iter);

    checkgrad(examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
    checkgrad(examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

    if (_word_indexers.size() > 0)
      checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter, _word_indexers);

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

#endif /* SRC_CNNHWordClassifier_H_ */
