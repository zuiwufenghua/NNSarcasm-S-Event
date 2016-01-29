/*
 * CNNHCharClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_CNNHCharClassifier_H_
#define SRC_CNNHCharClassifier_H_

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
class CNNHCharClassifier {
public:
  CNNHCharClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~CNNHCharClassifier() {

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

  Tensor<xpu, 2, double> _charEmb;
  Tensor<xpu, 2, double> _grad_charEmb;
  Tensor<xpu, 2, double> _eg2_charEmb;
  Tensor<xpu, 2, double> _ft_charEmb;
  hash_set<int> _char_indexers;

  int _charcontext;
  int _charSize;
  int _charDim;
  int _charHiddenSize;
  int _char_cnn_iSize;
  bool _b_charEmb_finetune;

  int _hiddenSize;

  UniHidderLayer<xpu> _cnnchar_project;
  UniHidderLayer<xpu> _cnn_project;
  UniHidderLayer<xpu> _gated_sent_pooling;
  UniHidderLayer<xpu> _gatedchar_pooling;
  UniHidderLayer<xpu> _gated_history_pooling;
  UniHidderLayer<xpu> _tanh_project;
  UniHidderLayer<xpu> _olayer_linear;

  int _labelSize;

  Metric _eval;

  double _dropOut;

  int _remove, _charremove; // 1, avg, 2, max, 3 min, 4 gated

  int _poolmanners;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, const NRMat<double>& charEmb, int charcontext, int labelSize, int wordHiddenSize,
      int charHiddenSize, int hiddenSize) {
    _wordcontext = wordcontext;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();
    _poolmanners = 4;

    _charcontext = charcontext;
    _charSize = charEmb.nrows();
    _charDim = charEmb.ncols();

    _labelSize = labelSize;
    _hiddenSize = hiddenSize;
    _wordHiddenSize = wordHiddenSize;
    _charHiddenSize = charHiddenSize;
    _token_representation_size = _wordDim + _poolmanners * _charHiddenSize;

    _char_cnn_iSize = _charDim * (2 * _charcontext + 1);
    _word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

    _wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
    assign(_wordEmb, wordEmb);
    for (int idx = 0; idx < _wordSize; idx++) {
      norm2one(_wordEmb, idx);
    }

    _charEmb = NewTensor<xpu>(Shape2(_charSize, _charDim), 0.0);
    _grad_charEmb = NewTensor<xpu>(Shape2(_charSize, _charDim), 0.0);
    _eg2_charEmb = NewTensor<xpu>(Shape2(_charSize, _charDim), 0.0);
    _ft_charEmb = NewTensor<xpu>(Shape2(_charSize, _charDim), 1.0);
    assign(_charEmb, charEmb);
    for (int idx = 0; idx < _charSize; idx++) {
      norm2one(_charEmb, idx);
    }

    _cnnchar_project.initial(_charHiddenSize, _char_cnn_iSize, true, 20, 0);
    _cnn_project.initial(_wordHiddenSize, _word_cnn_iSize, true, 30, 0);
    _gatedchar_pooling.initial(_charHiddenSize, _charHiddenSize, true, 40, 3);
    _gated_sent_pooling.initial(_wordHiddenSize, _wordHiddenSize, true, 50, 3);
    _gated_history_pooling.initial(_token_representation_size, _token_representation_size, true, 60, 3);
    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size, true, 70, 0);
    _olayer_linear.initial(_labelSize, _hiddenSize, false, 80, 2);

    _eval.reset();
    _word_indexers.clear();
    _char_indexers.clear();

    _remove = 0;
    _charremove = 0;

  }

  inline void release() {
    FreeSpace(&_wordEmb);
    FreeSpace(&_grad_wordEmb);
    FreeSpace(&_eg2_wordEmb);
    FreeSpace(&_ft_wordEmb);
    FreeSpace(&_charEmb);
    FreeSpace(&_grad_charEmb);
    FreeSpace(&_eg2_charEmb);
    FreeSpace(&_ft_charEmb);
    _cnnchar_project.release();
    _cnn_project.release();
    _gatedchar_pooling.release();
    _gated_sent_pooling.release();
    _gated_history_pooling.release();
    _tanh_project.release();
    _olayer_linear.release();
  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval.reset();
    _word_indexers.clear();
    _char_indexers.clear();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      if (seq_size > 2) {
        std::cout << "error" << std::endl;
      }

      vector<Tensor<xpu, 3, double> > charprime[seq_size], charprimeLoss[seq_size], charprimeMask[seq_size];
      vector<Tensor<xpu, 4, double> > charinputcontext[seq_size], charinputcontextLoss[seq_size];
      vector<Tensor<xpu, 3, double> > charinput[seq_size], charinputLoss[seq_size];
      vector<Tensor<xpu, 3, double> > charhidden[seq_size], charhiddenLoss[seq_size], charhiddenLossTmp[seq_size];
      vector<Tensor<xpu, 3, double> > charavgpoolIndex[seq_size], charmaxpoolIndex[seq_size], charminpoolIndex[seq_size];
      vector<Tensor<xpu, 2, double> > charavgpool[seq_size], charavgpoolLoss[seq_size];
      vector<Tensor<xpu, 2, double> > charmaxpool[seq_size], charmaxpoolLoss[seq_size];
      vector<Tensor<xpu, 2, double> > charminpool[seq_size], charminpoolLoss[seq_size];
      vector<Tensor<xpu, 3, double> > chargatedpoolIndex[seq_size], chargatedpoolIndexLoss[seq_size];
      vector<Tensor<xpu, 2, double> > chargatedpool[seq_size], chargatedpoolLoss[seq_size];
      vector<Tensor<xpu, 3, double> > chargateweight[seq_size], chargateweightLoss[seq_size], chargateweightIndex[seq_size];
      vector<Tensor<xpu, 2, double> > chargateweightsum[seq_size], chargateweightsumLoss[seq_size];

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
        int charwindow = 2 * _charcontext + 1;
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();
        int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
        int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

        charprime[idx].resize(word_num);
        charprimeLoss[idx].resize(word_num);
        charprimeMask[idx].resize(word_num);
        charinputcontext[idx].resize(word_num);
        charinputcontextLoss[idx].resize(word_num);
        charinput[idx].resize(word_num);
        charinputLoss[idx].resize(word_num);
        charhidden[idx].resize(word_num);
        charhiddenLoss[idx].resize(word_num);
        charhiddenLossTmp[idx].resize(word_num);
        charavgpool[idx].resize(word_num);
        charavgpoolLoss[idx].resize(word_num);
        charavgpoolIndex[idx].resize(word_num);
        charmaxpool[idx].resize(word_num);
        charmaxpoolLoss[idx].resize(word_num);
        charmaxpoolIndex[idx].resize(word_num);
        charminpool[idx].resize(word_num);
        charminpoolLoss[idx].resize(word_num);
        charminpoolIndex[idx].resize(word_num);
        chargatedpool[idx].resize(word_num);
        chargatedpoolLoss[idx].resize(word_num);
        chargatedpoolIndex[idx].resize(word_num);
        chargatedpoolIndexLoss[idx].resize(word_num);
        chargateweight[idx].resize(word_num);
        chargateweightLoss[idx].resize(word_num);
        chargateweightIndex[idx].resize(word_num);
        chargateweightsum[idx].resize(word_num);
        chargateweightsumLoss[idx].resize(word_num);

        for (int idy = 0; idy < word_num; idy++) {
          int char_num = feature.chars[idy].size();
          charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
          charprimeLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
          charprimeMask[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 1.0);
          charinputcontext[idx][idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
          charinputcontextLoss[idx][idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
          charinput[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
          charinputLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
          charhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charhiddenLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charhiddenLossTmp[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charavgpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charavgpoolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charavgpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charmaxpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charmaxpoolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charmaxpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charminpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charminpoolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charminpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargatedpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargatedpoolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargatedpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargatedpoolIndexLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweight[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightsum[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargateweightsumLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        }

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
        int charwindow = 2 * _charcontext + 1;

        const vector<int>& words = feature.words;
        const vector<vector<int> >& chars = feature.chars;
        int word_num = words.size();
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        for (int idy = 0; idy < word_num; idy++) {
          int char_num = chars[idy].size();

          //charprime
          for (int idz = 0; idz < char_num; idz++) {
            offset = chars[idy][idz];
            charprime[idx][idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
          }

          //char dropout
          for (int idz = 0; idz < char_num; idz++) {
            for (int j = 0; j < _charDim; j++) {
              if (1.0 * rand() / RAND_MAX >= _dropOut) {
                charprimeMask[idx][idy][idz][0][j] = 1.0;
              } else {
                charprimeMask[idx][idy][idz][0][j] = 0.0;
              }
            }
            charprime[idx][idy][idz] = charprime[idx][idy][idz] * charprimeMask[idx][idy][idz];
          }

          //char context
          for (int idz = 0; idz < char_num; idz++) {
            charinputcontext[idx][idy][idz][0] += charprime[idx][idy][idz];
            for (int idc = 1; idc <= _charcontext; idc++) {
              if (idz - idc >= 0) {
                charinputcontext[idx][idy][idz][2 * idc - 1] += charprime[idx][idy][idz - idc];
              }
              if (idz + idc < char_num) {
                charinputcontext[idx][idy][idz][2 * idc] += charprime[idx][idy][idz + idc];
              }
            }
          }

          //char reshape
          for (int idz = 0; idz < char_num; idz++) {
            offset = 0;
            for (int i = 0; i < charwindow; i++) {
              for (int j = 0; j < _charDim; j++) {
                charinput[idx][idy][idz][0][offset] = charinputcontext[idx][idy][idz][i][0][j];
                offset++;
              }
            }
          }

          //char convolution
          for (int idz = 0; idz < char_num; idz++) {
            _cnnchar_project.ComputeForwardScore(charinput[idx][idy][idz], charhidden[idx][idy][idz]);
          }

          //char pooling
          if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
            avgpool_forward(charhidden[idx][idy], charavgpool[idx][idy], charavgpoolIndex[idx][idy]);
          }
          if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
            maxpool_forward(charhidden[idx][idy], charmaxpool[idx][idy], charmaxpoolIndex[idx][idy]);
          }
          if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
            minpool_forward(charhidden[idx][idy], charminpool[idx][idy], charminpoolIndex[idx][idy]);
          }
          if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
            for (int idz = 0; idz < char_num; idz++) {
              _gatedchar_pooling.ComputeForwardScore(charhidden[idx][idy][idz], chargateweight[idx][idy][idz]);
            }
            sumpool_forward(chargateweight[idx][idy], chargateweightsum[idx][idy], chargateweightIndex[idx][idy]);
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpoolIndex[idx][idy][idz] = chargateweight[idx][idy][idz] / chargateweightsum[idx][idy];
            }
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpool[idx][idy] += charhidden[idx][idy][idz] * chargatedpoolIndex[idx][idy][idz];
            }
          }

        }

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
          concat(wordprime[idx][idy], charavgpool[idx][idy], charmaxpool[idx][idy], charminpool[idx][idy], chargatedpool[idx][idy], wordrepresent[idx][idy]);
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
          maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
        }

        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            if (idx == seq_size - 1)
              _gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
            else {
              _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
            }
          }
          sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
          for (int idy = 0; idy < word_num; idy++) {
            poolIndex[offset + 3][idy] = gateweight[idx][idy] / gateweightsum[idx];
          }
          for (int idy = 0; idy < word_num; idy++) {
            pool[offset + 3] += hidden[idx][idy] * poolIndex[offset + 3][idy];
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
        int charwindow = 2 * _charcontext + 1;

        const vector<int>& words = feature.words;
        const vector<vector<int> >& chars = feature.chars;
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
          pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], hiddenLossTmp[idx]);
          hiddenLoss[idx] = hiddenLoss[idx] + hiddenLossTmp[idx];
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          hiddenLossTmp[idx] = 0.0;
          pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], hiddenLossTmp[idx]);
          hiddenLoss[idx] = hiddenLoss[idx] + hiddenLossTmp[idx];
        }

        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            poolIndexLoss[offset + 3][idy] = poolLoss[offset + 3] * hidden[idx][idy];
            hiddenLoss[idx][idy] += poolLoss[offset + 3] * poolIndex[offset + 3][idy];
          }

          for (int idy = 0; idy < word_num; idy++) {
            gateweightLoss[idx][idy] += poolIndexLoss[offset + 3][idy] / gateweightsum[idx];
            gateweightsumLoss[idx] -= poolIndexLoss[offset + 3][idy] * gateweight[idx][idy] / gateweightsum[idx] / gateweightsum[idx];
          }

          pool_backward(gateweightsumLoss[idx], gateweightIndex[idx], gateweightLoss[idx]);

          hiddenLossTmp[idx] = 0.0;
          for (int idy = 0; idy < word_num; idy++) {
            if (idx == seq_size - 1)
              _gated_sent_pooling.ComputeBackwardLoss(hidden[idx][idy], gateweight[idx][idy], gateweightLoss[idx][idy], hiddenLossTmp[idx][idy]);
            else {
              _gated_history_pooling.ComputeBackwardLoss(hidden[idx][idy], gateweight[idx][idy], gateweightLoss[idx][idy], hiddenLossTmp[idx][idy]);
            }
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
          unconcat(wordprimeLoss[idx][idy], charavgpoolLoss[idx][idy], charmaxpoolLoss[idx][idy], charminpoolLoss[idx][idy], chargatedpoolLoss[idx][idy],
              wordrepresentLoss[idx][idy]);
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

        for (int idy = 0; idy < word_num; idy++) {
          int char_num = chars[idy].size();

          //char pooling
          if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
            charhiddenLossTmp[idx][idy] = 0.0;
            pool_backward(charavgpoolLoss[idx][idy], charavgpoolIndex[idx][idy], charhiddenLossTmp[idx][idy]);
            charhiddenLoss[idx][idy] = charhiddenLoss[idx][idy] + charhiddenLossTmp[idx][idy];
          }
          if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
            charhiddenLossTmp[idx][idy] = 0.0;
            pool_backward(charmaxpoolLoss[idx][idy], charmaxpoolIndex[idx][idy], charhiddenLossTmp[idx][idy]);
            charhiddenLoss[idx][idy] = charhiddenLoss[idx][idy] + charhiddenLossTmp[idx][idy];
          }
          if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
            charhiddenLossTmp[idx][idy] = 0.0;
            pool_backward(charminpoolLoss[idx][idy], charminpoolIndex[idx][idy], charhiddenLossTmp[idx][idy]);
            charhiddenLoss[idx][idy] = charhiddenLoss[idx][idy] + charhiddenLossTmp[idx][idy];
          }

          if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpoolIndexLoss[idx][idy][idz] = chargatedpoolLoss[idx][idy] * charhidden[idx][idy][idz];
              charhiddenLoss[idx][idy][idz] += chargatedpoolLoss[idx][idy] * chargatedpoolIndex[idx][idy][idz];
            }

            for (int idz = 0; idz < char_num; idz++) {
              chargateweightLoss[idx][idy][idz] += chargatedpoolIndexLoss[idx][idy][idz] / chargateweightsum[idx][idy];
              chargateweightsumLoss[idx][idy] -= chargatedpoolIndexLoss[idx][idy][idz] / chargateweightsum[idx][idy] * chargateweight[idx][idy][idz]
                  / chargateweightsum[idx][idy];
            }

            pool_backward(chargateweightsumLoss[idx][idy], chargateweightIndex[idx][idy], chargateweightLoss[idx][idy]);

            charhiddenLossTmp[idx][idy] = 0.0;
            for (int idz = 0; idz < char_num; idz++) {
              _gatedchar_pooling.ComputeBackwardLoss(charhidden[idx][idy][idz], chargateweight[idx][idy][idz], chargateweightLoss[idx][idy][idz],
                  charhiddenLossTmp[idx][idy][idz]);
              charhiddenLoss[idx][idy][idz] += charhiddenLossTmp[idx][idy][idz];
            }
          }

          //char convolution
          for (int idz = 0; idz < char_num; idz++) {
            _cnnchar_project.ComputeBackwardLoss(charinput[idx][idy][idz], charhidden[idx][idy][idz], charhiddenLoss[idx][idy][idz],
                charinputLoss[idx][idy][idz]);
          }

          //reshape
          for (int idz = 0; idz < char_num; idz++) {
            offset = 0;
            for (int i = 0; i < charwindow; i++) {
              for (int j = 0; j < _charDim; j++) {
                charinputcontextLoss[idx][idy][idz][i][0][j] = charinputLoss[idx][idy][idz][0][offset];
                offset++;
              }
            }
          }

          //char context
          for (int idz = 0; idz < char_num; idz++) {
            charprimeLoss[idx][idy][idz] += charinputcontextLoss[idx][idy][idz][0];
            for (int idc = 1; idc <= _charcontext; idc++) {
              if (idz - idc >= 0) {
                charprimeLoss[idx][idy][idz - idc] += charinputcontextLoss[idx][idy][idz][2 * idc - 1];
              }
              if (idz + idc < char_num) {
                charprimeLoss[idx][idy][idz + idc] += charinputcontextLoss[idx][idy][idz][2 * idc];
              }
            }
          }

          //char dropout
          for (int idz = 0; idz < char_num; idz++) {
            charprimeLoss[idx][idy][idz] = charprimeLoss[idx][idy][idz] * charprimeMask[idx][idy][idz];
          }

          //char finetune
          if (_b_charEmb_finetune) {
            for (int idz = 0; idz < char_num; idz++) {
              offset = chars[idy][idz];
              _grad_charEmb[offset] += charprimeLoss[idx][idy][idz][0];
              _char_indexers.insert(offset);
            }
          }
        }

      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();

        for (int idy = 0; idy < word_num; idy++) {
          FreeSpace(&(charprime[idx][idy]));
          FreeSpace(&(charprimeLoss[idx][idy]));
          FreeSpace(&(charprimeMask[idx][idy]));
          FreeSpace(&(charinputcontext[idx][idy]));
          FreeSpace(&(charinputcontextLoss[idx][idy]));
          FreeSpace(&(charinput[idx][idy]));
          FreeSpace(&(charinputLoss[idx][idy]));
          FreeSpace(&(charhidden[idx][idy]));
          FreeSpace(&(charhiddenLoss[idx][idy]));
          FreeSpace(&(charhiddenLossTmp[idx][idy]));
          FreeSpace(&(charavgpool[idx][idy]));
          FreeSpace(&(charavgpoolLoss[idx][idy]));
          FreeSpace(&(charavgpoolIndex[idx][idy]));
          FreeSpace(&(charmaxpool[idx][idy]));
          FreeSpace(&(charmaxpoolLoss[idx][idy]));
          FreeSpace(&(charmaxpoolIndex[idx][idy]));
          FreeSpace(&(charminpool[idx][idy]));
          FreeSpace(&(charminpoolLoss[idx][idy]));
          FreeSpace(&(charminpoolIndex[idx][idy]));
          FreeSpace(&(chargatedpool[idx][idy]));
          FreeSpace(&(chargatedpoolLoss[idx][idy]));
          FreeSpace(&(chargatedpoolIndex[idx][idy]));
          FreeSpace(&(chargatedpoolIndexLoss[idx][idy]));
          FreeSpace(&(chargateweight[idx][idy]));
          FreeSpace(&(chargateweightIndex[idx][idy]));
          FreeSpace(&(chargateweightLoss[idx][idy]));
          FreeSpace(&(chargateweightsum[idx][idy]));
          FreeSpace(&(chargateweightsumLoss[idx][idy]));
        }

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

  int predict(const vector<Feature>& features, vector<double>& results) {
    int seq_size = features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }

    vector<Tensor<xpu, 3, double> > charprime[seq_size];
    vector<Tensor<xpu, 4, double> > charinputcontext[seq_size];
    vector<Tensor<xpu, 3, double> > charinput[seq_size];
    vector<Tensor<xpu, 3, double> > charhidden[seq_size];
    vector<Tensor<xpu, 3, double> > charavgpoolIndex[seq_size], charmaxpoolIndex[seq_size], charminpoolIndex[seq_size];
    vector<Tensor<xpu, 2, double> > charavgpool[seq_size];
    vector<Tensor<xpu, 2, double> > charmaxpool[seq_size];
    vector<Tensor<xpu, 2, double> > charminpool[seq_size];
    vector<Tensor<xpu, 3, double> > chargatedpoolIndex[seq_size];
    vector<Tensor<xpu, 2, double> > chargatedpool[seq_size];
    vector<Tensor<xpu, 3, double> > chargateweight[seq_size], chargateweightIndex[seq_size];
    vector<Tensor<xpu, 2, double> > chargateweightsum[seq_size];

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
      int charwindow = 2 * _charcontext + 1;
      const Feature& feature = features[idx];
      int word_num = feature.words.size();
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      charprime[idx].resize(word_num);
      charinputcontext[idx].resize(word_num);
      charinput[idx].resize(word_num);
      charhidden[idx].resize(word_num);
      charavgpool[idx].resize(word_num);
      charavgpoolIndex[idx].resize(word_num);
      charmaxpool[idx].resize(word_num);
      charmaxpoolIndex[idx].resize(word_num);
      charminpool[idx].resize(word_num);
      charminpoolIndex[idx].resize(word_num);
      chargatedpool[idx].resize(word_num);
      chargatedpoolIndex[idx].resize(word_num);
      chargateweight[idx].resize(word_num);
      chargateweightIndex[idx].resize(word_num);
      chargateweightsum[idx].resize(word_num);

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
        charinputcontext[idx][idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
        charinput[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
        charhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charavgpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charavgpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charmaxpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charmaxpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charminpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charminpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargatedpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        chargatedpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweight[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightsum[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
      }

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
      int charwindow = 2 * _charcontext + 1;

      const vector<int>& words = feature.words;
      const vector<vector<int> >& chars = feature.chars;
      int word_num = words.size();
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = chars[idy].size();

        //charprime
        for (int idz = 0; idz < char_num; idz++) {
          offset = chars[idy][idz];
          charprime[idx][idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
        }

        //char context
        for (int idz = 0; idz < char_num; idz++) {
          charinputcontext[idx][idy][idz][0] += charprime[idx][idy][idz];
          for (int idc = 1; idc <= _charcontext; idc++) {
            if (idz - idc >= 0) {
              charinputcontext[idx][idy][idz][2 * idc - 1] += charprime[idx][idy][idz - idc];
            }
            if (idz + idc < char_num) {
              charinputcontext[idx][idy][idz][2 * idc] += charprime[idx][idy][idz + idc];
            }
          }
        }

        //char reshape
        for (int idz = 0; idz < char_num; idz++) {
          offset = 0;
          for (int i = 0; i < charwindow; i++) {
            for (int j = 0; j < _charDim; j++) {
              charinput[idx][idy][idz][0][offset] = charinputcontext[idx][idy][idz][i][0][j];
              offset++;
            }
          }
        }

        //char convolution
        for (int idz = 0; idz < char_num; idz++) {
          _cnnchar_project.ComputeForwardScore(charinput[idx][idy][idz], charhidden[idx][idy][idz]);
        }

        //char pooling
        if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
          avgpool_forward(charhidden[idx][idy], charavgpool[idx][idy], charavgpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
          maxpool_forward(charhidden[idx][idy], charmaxpool[idx][idy], charmaxpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
          minpool_forward(charhidden[idx][idy], charminpool[idx][idy], charminpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
          for (int idz = 0; idz < char_num; idz++) {
            _gatedchar_pooling.ComputeForwardScore(charhidden[idx][idy][idz], chargateweight[idx][idy][idz]);
          }
          sumpool_forward(chargateweight[idx][idy], chargateweightsum[idx][idy], chargateweightIndex[idx][idy]);
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpoolIndex[idx][idy][idz] = chargateweight[idx][idy][idz] / chargateweightsum[idx][idy];
          }
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpool[idx][idy] += charhidden[idx][idy][idz] * chargatedpoolIndex[idx][idy][idz];
          }
        }

      }

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        concat(wordprime[idx][idy], charavgpool[idx][idy], charmaxpool[idx][idy], charminpool[idx][idy], chargatedpool[idx][idy], wordrepresent[idx][idy]);
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
        maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1)
            _gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          else {
            _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          }
        }
        sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
        for (int idy = 0; idy < word_num; idy++) {
          poolIndex[offset + 3][idy] = gateweight[idx][idy] / gateweightsum[idx];
        }
        for (int idy = 0; idy < word_num; idy++) {
          pool[offset + 3] += hidden[idx][idy] * poolIndex[offset + 3][idy];
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
      const Feature& feature = features[idx];
      int word_num = feature.words.size();

      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idx][idy]));
        FreeSpace(&(charinputcontext[idx][idy]));
        FreeSpace(&(charinput[idx][idy]));
        FreeSpace(&(charhidden[idx][idy]));
        FreeSpace(&(charavgpool[idx][idy]));
        FreeSpace(&(charavgpoolIndex[idx][idy]));
        FreeSpace(&(charmaxpool[idx][idy]));
        FreeSpace(&(charmaxpoolIndex[idx][idy]));
        FreeSpace(&(charminpool[idx][idy]));
        FreeSpace(&(charminpoolIndex[idx][idy]));
        FreeSpace(&(chargatedpool[idx][idy]));
        FreeSpace(&(chargatedpoolIndex[idx][idy]));
        FreeSpace(&(chargateweight[idx][idy]));
        FreeSpace(&(chargateweightIndex[idx][idy]));
        FreeSpace(&(chargateweightsum[idx][idy]));
      }

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

    vector<Tensor<xpu, 3, double> > charprime[seq_size];
    vector<Tensor<xpu, 4, double> > charinputcontext[seq_size];
    vector<Tensor<xpu, 3, double> > charinput[seq_size];
    vector<Tensor<xpu, 3, double> > charhidden[seq_size];
    vector<Tensor<xpu, 3, double> > charavgpoolIndex[seq_size], charmaxpoolIndex[seq_size], charminpoolIndex[seq_size];
    vector<Tensor<xpu, 2, double> > charavgpool[seq_size];
    vector<Tensor<xpu, 2, double> > charmaxpool[seq_size];
    vector<Tensor<xpu, 2, double> > charminpool[seq_size];
    vector<Tensor<xpu, 3, double> > chargatedpoolIndex[seq_size];
    vector<Tensor<xpu, 2, double> > chargatedpool[seq_size];
    vector<Tensor<xpu, 3, double> > chargateweight[seq_size], chargateweightIndex[seq_size];
    vector<Tensor<xpu, 2, double> > chargateweightsum[seq_size];

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
      int charwindow = 2 * _charcontext + 1;
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      charprime[idx].resize(word_num);
      charinputcontext[idx].resize(word_num);
      charinput[idx].resize(word_num);
      charhidden[idx].resize(word_num);
      charavgpool[idx].resize(word_num);
      charavgpoolIndex[idx].resize(word_num);
      charmaxpool[idx].resize(word_num);
      charmaxpoolIndex[idx].resize(word_num);
      charminpool[idx].resize(word_num);
      charminpoolIndex[idx].resize(word_num);
      chargatedpool[idx].resize(word_num);
      chargatedpoolIndex[idx].resize(word_num);
      chargateweight[idx].resize(word_num);
      chargateweightIndex[idx].resize(word_num);
      chargateweightsum[idx].resize(word_num);

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
        charinputcontext[idx][idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
        charinput[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
        charhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charavgpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charavgpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charmaxpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charmaxpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charminpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charminpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargatedpool[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        chargatedpoolIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweight[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightIndex[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightsum[idx][idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
      }

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
      int charwindow = 2 * _charcontext + 1;

      const vector<int>& words = feature.words;
      const vector<vector<int> >& chars = feature.chars;
      int word_num = words.size();
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = chars[idy].size();

        //charprime
        for (int idz = 0; idz < char_num; idz++) {
          offset = chars[idy][idz];
          charprime[idx][idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
        }

        //char context
        for (int idz = 0; idz < char_num; idz++) {
          charinputcontext[idx][idy][idz][0] += charprime[idx][idy][idz];
          for (int idc = 1; idc <= _charcontext; idc++) {
            if (idz - idc >= 0) {
              charinputcontext[idx][idy][idz][2 * idc - 1] += charprime[idx][idy][idz - idc];
            }
            if (idz + idc < char_num) {
              charinputcontext[idx][idy][idz][2 * idc] += charprime[idx][idy][idz + idc];
            }
          }
        }

        //char reshape
        for (int idz = 0; idz < char_num; idz++) {
          offset = 0;
          for (int i = 0; i < charwindow; i++) {
            for (int j = 0; j < _charDim; j++) {
              charinput[idx][idy][idz][0][offset] = charinputcontext[idx][idy][idz][i][0][j];
              offset++;
            }
          }
        }

        //char convolution
        for (int idz = 0; idz < char_num; idz++) {
          _cnnchar_project.ComputeForwardScore(charinput[idx][idy][idz], charhidden[idx][idy][idz]);
        }

        //char pooling
        if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
          avgpool_forward(charhidden[idx][idy], charavgpool[idx][idy], charavgpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
          maxpool_forward(charhidden[idx][idy], charmaxpool[idx][idy], charmaxpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
          minpool_forward(charhidden[idx][idy], charminpool[idx][idy], charminpoolIndex[idx][idy]);
        }
        if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
          for (int idz = 0; idz < char_num; idz++) {
            _gatedchar_pooling.ComputeForwardScore(charhidden[idx][idy][idz], chargateweight[idx][idy][idz]);
          }
          sumpool_forward(chargateweight[idx][idy], chargateweightsum[idx][idy], chargateweightIndex[idx][idy]);
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpoolIndex[idx][idy][idz] = chargateweight[idx][idy][idz] / chargateweightsum[idx][idy];
          }
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpool[idx][idy] += charhidden[idx][idy][idz] * chargatedpoolIndex[idx][idy][idz];
          }
        }

      }

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idx][idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        concat(wordprime[idx][idy], charavgpool[idx][idy], charmaxpool[idx][idy], charminpool[idx][idy], chargatedpool[idx][idy], wordrepresent[idx][idy]);
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
        maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1)
            _gated_sent_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          else {
            _gated_history_pooling.ComputeForwardScore(hidden[idx][idy], gateweight[idx][idy]);
          }
        }
        sumpool_forward(gateweight[idx], gateweightsum[idx], gateweightIndex[idx]);
        for (int idy = 0; idy < word_num; idy++) {
          poolIndex[offset + 3][idy] = gateweight[idx][idy] / gateweightsum[idx];
        }
        for (int idy = 0; idy < word_num; idy++) {
          pool[offset + 3] += hidden[idx][idy] * poolIndex[offset + 3][idy];
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
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();

      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idx][idy]));
        FreeSpace(&(charinputcontext[idx][idy]));
        FreeSpace(&(charinput[idx][idy]));
        FreeSpace(&(charhidden[idx][idy]));
        FreeSpace(&(charavgpool[idx][idy]));
        FreeSpace(&(charavgpoolIndex[idx][idy]));
        FreeSpace(&(charmaxpool[idx][idy]));
        FreeSpace(&(charmaxpoolIndex[idx][idy]));
        FreeSpace(&(charminpool[idx][idy]));
        FreeSpace(&(charminpoolIndex[idx][idy]));
        FreeSpace(&(chargatedpool[idx][idy]));
        FreeSpace(&(chargatedpoolIndex[idx][idy]));
        FreeSpace(&(chargateweight[idx][idy]));
        FreeSpace(&(chargateweightIndex[idx][idy]));
        FreeSpace(&(chargateweightsum[idx][idy]));
      }

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
    _cnnchar_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gatedchar_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gated_sent_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gated_history_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);

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

    if (_b_charEmb_finetune) {
      static hash_set<int>::iterator it;
      Tensor<xpu, 1, double> _grad_charEmb_ij = NewTensor<xpu>(Shape1(_charDim), 0.0);
      Tensor<xpu, 1, double> tmp_normaize_alpha = NewTensor<xpu>(Shape1(_charDim), 0.0);
      Tensor<xpu, 1, double> tmp_alpha = NewTensor<xpu>(Shape1(_charDim), 0.0);
      Tensor<xpu, 1, double> _ft_charEmb_ij = NewTensor<xpu>(Shape1(_charDim), 0.0);

      for (it = _char_indexers.begin(); it != _char_indexers.end(); ++it) {
        int index = *it;
        _grad_charEmb_ij = _grad_charEmb[index] + nnRegular * _charEmb[index] / _ft_charEmb[index];
        _eg2_charEmb[index] += _grad_charEmb_ij * _grad_charEmb_ij;
        tmp_normaize_alpha = F<nl_sqrt>(_eg2_charEmb[index] + adaEps);
        tmp_alpha = adaAlpha / tmp_normaize_alpha;
        _ft_charEmb_ij = _ft_charEmb[index] * tmp_alpha * nnRegular;
        _ft_charEmb[index] -= _ft_charEmb_ij;
        _charEmb[index] -= tmp_alpha * _grad_charEmb[index] / _ft_charEmb[index];
        _grad_charEmb[index] = 0.0;
      }

      FreeSpace(&_grad_charEmb_ij);
      FreeSpace(&tmp_normaize_alpha);
      FreeSpace(&tmp_alpha);
      FreeSpace(&_ft_charEmb_ij);
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

    checkgrad(examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
    checkgrad(examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

    checkgrad(examples, _cnnchar_project._W, _cnnchar_project._gradW, "_cnnchar_project._W", iter);
    checkgrad(examples, _cnnchar_project._b, _cnnchar_project._gradb, "_cnnchar_project._b", iter);

    checkgrad(examples, _gated_sent_pooling._W, _gated_sent_pooling._gradW, "_gated_sent_pooling._W", iter);
    checkgrad(examples, _gated_sent_pooling._b, _gated_sent_pooling._gradb, "_gated_sent_pooling._b", iter);

    checkgrad(examples, _gated_history_pooling._W, _gated_history_pooling._gradW, "_gated_history_pooling._W", iter);
    checkgrad(examples, _gated_history_pooling._b, _gated_history_pooling._gradb, "_gated_history_pooling._b", iter);

    checkgrad(examples, _gatedchar_pooling._W, _gatedchar_pooling._gradW, "_gatedchar_pooling._W", iter);
    checkgrad(examples, _gatedchar_pooling._b, _gatedchar_pooling._gradb, "_gatedchar_pooling._b", iter);

    if (_word_indexers.size() > 0)
      checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter, _word_indexers);

    if (_char_indexers.size() > 0)
      checkgrad(examples, _charEmb, _grad_charEmb, "_charEmb", iter, _char_indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(double dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune, bool b_charEmb_finetune) {
    _b_wordEmb_finetune = b_wordEmb_finetune;
    _b_charEmb_finetune = b_charEmb_finetune;
  }

  inline void resetRemove(int remove, int charremove) {
    _remove = remove;
    _charremove = charremove;
  }

};

#endif /* SRC_CNNHCharClassifier_H_ */
