/*
 * CNNCharClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_CNNCharClassifier_H_
#define SRC_CNNCharClassifier_H_

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
class CNNCharClassifier {
public:
  CNNCharClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~CNNCharClassifier() {

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
  UniHidderLayer<xpu> _gated_pooling;
  UniHidderLayer<xpu> _gatedchar_pooling;
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
    _gated_pooling.initial(_wordHiddenSize, _wordHiddenSize, true, 50, 3);
    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize, true, 60, 0);
    _olayer_linear.initial(_labelSize, _hiddenSize, false, 70, 2);

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
    _gated_pooling.release();
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

      vector<Tensor<xpu, 3, double> > charprime, charprimeLoss, charprimeMask;
      vector<Tensor<xpu, 4, double> > charinputcontext, charinputcontextLoss;
      vector<Tensor<xpu, 3, double> > charinput, charinputLoss;
      vector<Tensor<xpu, 3, double> > charhidden, charhiddenLoss, charhiddenLossTmp;
      vector<Tensor<xpu, 3, double> > charavgpoolIndex, charmaxpoolIndex, charminpoolIndex;
      vector<Tensor<xpu, 2, double> > charavgpool, charavgpoolLoss;
      vector<Tensor<xpu, 2, double> > charmaxpool, charmaxpoolLoss;
      vector<Tensor<xpu, 2, double> > charminpool, charminpoolLoss;
      vector<Tensor<xpu, 3, double> > chargatedpoolIndex, chargatedpoolIndexLoss;
      vector<Tensor<xpu, 2, double> > chargatedpool, chargatedpoolLoss;
      vector<Tensor<xpu, 3, double> > chargateweight, chargateweightLoss, chargateweightIndex;
      vector<Tensor<xpu, 2, double> > chargateweightsum, chargateweightsumLoss;

      Tensor<xpu, 3, double> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, double> wordrepresent, wordrepresentLoss;
      Tensor<xpu, 4, double> wordinputcontext, wordinputcontextLoss;
      Tensor<xpu, 3, double> input, inputLoss;
      Tensor<xpu, 3, double> hidden, hiddenLoss, hiddenLossTmp;
      vector<Tensor<xpu, 2, double> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, double> >  poolIndex(_poolmanners), poolIndexLoss(_poolmanners);
      Tensor<xpu, 3, double> gateweight, gateweightLoss, gateweightIndex;
      Tensor<xpu, 2, double> gateweightsum, gateweightsumLoss;

      Tensor<xpu, 2, double> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, double> project, projectLoss;
      Tensor<xpu, 2, double> output, outputLoss;
      Tensor<xpu, 2, double> scores;

      //initialize
      int idx = seq_size - 1;

      {
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int charwindow = 2 * _charcontext + 1;
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();
        int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
        int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

        charprime.resize(word_num);
        charprimeLoss.resize(word_num);
        charprimeMask.resize(word_num);
        charinputcontext.resize(word_num);
        charinputcontextLoss.resize(word_num);
        charinput.resize(word_num);
        charinputLoss.resize(word_num);
        charhidden.resize(word_num);
        charhiddenLoss.resize(word_num);
        charhiddenLossTmp.resize(word_num);
        charavgpool.resize(word_num);
        charavgpoolLoss.resize(word_num);
        charavgpoolIndex.resize(word_num);
        charmaxpool.resize(word_num);
        charmaxpoolLoss.resize(word_num);
        charmaxpoolIndex.resize(word_num);
        charminpool.resize(word_num);
        charminpoolLoss.resize(word_num);
        charminpoolIndex.resize(word_num);
        chargatedpool.resize(word_num);
        chargatedpoolLoss.resize(word_num);
        chargatedpoolIndex.resize(word_num);
        chargatedpoolIndexLoss.resize(word_num);
        chargateweight.resize(word_num);
        chargateweightLoss.resize(word_num);
        chargateweightIndex.resize(word_num);
        chargateweightsum.resize(word_num);
        chargateweightsumLoss.resize(word_num);

        for (int idy = 0; idy < word_num; idy++) {
          int char_num = feature.chars[idy].size();
          charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
          charprimeLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
          charprimeMask[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 1.0);
          charinputcontext[idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
          charinputcontextLoss[idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
          charinput[idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
          charinputLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
          charhidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charhiddenLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charhiddenLossTmp[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charavgpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charavgpoolLoss[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charavgpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charmaxpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charmaxpoolLoss[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charmaxpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          charminpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charminpoolLoss[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          charminpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargatedpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargatedpoolLoss[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargatedpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargatedpoolIndexLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweight[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
          chargateweightsum[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
          chargateweightsumLoss[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        }

        wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
        wordprimeLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
        wordprimeMask = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 1.0);
        wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
        wordrepresentLoss = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
        wordinputcontext = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
        wordinputcontextLoss = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
        input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
        inputLoss = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
        hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        hiddenLoss = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        hiddenLossTmp = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
          poolLoss[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
          poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
          poolIndexLoss[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        }
        gateweight = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        gateweightIndex = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
        gateweightLoss = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

        gateweightsum = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        gateweightsumLoss = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
      }
      poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
      project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
      projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

      //forward propagation
      //input setting, and linear setting
      {
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
            charprime[idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
          }

          //char dropout
          for (int idz = 0; idz < char_num; idz++) {
            for (int j = 0; j < _charDim; j++) {
              if (1.0 * rand() / RAND_MAX >= _dropOut) {
                charprimeMask[idy][idz][0][j] = 1.0;
              } else {
                charprimeMask[idy][idz][0][j] = 0.0;
              }
            }
            charprime[idy][idz] = charprime[idy][idz] * charprimeMask[idy][idz];
          }


          //char context
          for (int idz = 0; idz < char_num; idz++) {
            charinputcontext[idy][idz][0] += charprime[idy][idz];
            for (int idc = 1; idc <= _charcontext; idc++) {
              if (idz - idc >= 0) {
                charinputcontext[idy][idz][2 * idc - 1] += charprime[idy][idz - idc];
              }
              if (idz + idc < char_num) {
                charinputcontext[idy][idz][2 * idc] += charprime[idy][idz + idc];
              }
            }
          }

          //char reshape
          for (int idz = 0; idz < char_num; idz++) {
            offset = 0;
            for (int i = 0; i < charwindow; i++) {
              for (int j = 0; j < _charDim; j++) {
                charinput[idy][idz][0][offset] = charinputcontext[idy][idz][i][0][j];
                offset++;
              }
            }
          }

          //char convolution
          for (int idz = 0; idz < char_num; idz++) {
            _cnnchar_project.ComputeForwardScore(charinput[idy][idz], charhidden[idy][idz]);
          }


          //char pooling
          if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
            avgpool_forward(charhidden[idy], charavgpool[idy], charavgpoolIndex[idy]);
          }
          if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
            maxpool_forward(charhidden[idy], charmaxpool[idy], charmaxpoolIndex[idy]);
          }
          if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
            minpool_forward(charhidden[idy], charminpool[idy], charminpoolIndex[idy]);
          }
          if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
            for (int idz = 0; idz < char_num; idz++) {
              _gatedchar_pooling.ComputeForwardScore(charhidden[idy][idz], chargateweight[idy][idz]);
            }
            sumpool_forward(chargateweight[idy], chargateweightsum[idy], chargateweightIndex[idy]);
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpoolIndex[idy][idz] = chargateweight[idy][idz] / chargateweightsum[idy];
            }
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpool[idy] += charhidden[idy][idz] * chargatedpoolIndex[idy][idz];
            }
          }

        }

        for (int idy = 0; idy < word_num; idy++) {
          offset = words[idy];
          wordprime[idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
        }

        //word dropout
        for (int idy = 0; idy < word_num; idy++) {
          for (int j = 0; j < _wordDim; j++) {
            if (1.0 * rand() / RAND_MAX >= _dropOut) {
              wordprimeMask[idy][0][j] = 1.0;
            } else {
              wordprimeMask[idy][0][j] = 0.0;
            }
          }
          wordprime[idy] = wordprime[idy] * wordprimeMask[idy];
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          concat(wordprime[idy], charavgpool[idy], charmaxpool[idy], charminpool[idy], chargatedpool[idy], wordrepresent[idy]);
        }

        //word context
        for (int idy = 0; idy < word_num; idy++) {
          wordinputcontext[idy][0] += wordrepresent[idy];
          for (int idc = 1; idc <= curcontext; idc++) {
            if (idy - idc >= 0) {
              wordinputcontext[idy][2 * idc - 1] += wordrepresent[idy - idc];
            }
            if (idy + idc < word_num) {
              wordinputcontext[idy][2 * idc] += wordrepresent[idy + idc];
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
          _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
        }

        //word pooling
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden, pool[0], poolIndex[0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden, pool[1], poolIndex[1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden, pool[2], poolIndex[2]);
        }

        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            _gated_pooling.ComputeForwardScore(hidden[idy], gateweight[idy]);
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

      unconcat(poolLoss, poolmergeLoss);

      {

        const Feature& feature = example.m_features[idx];
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;
        int charwindow = 2 * _charcontext + 1;
        const vector<int>& words = feature.words;
        const vector<vector<int> >& chars = feature.chars;
        int word_num = words.size();

        //word pooling
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          hiddenLossTmp = 0.0;
          pool_backward(poolLoss[0], poolIndex[0], hiddenLossTmp);
          hiddenLoss = hiddenLoss + hiddenLossTmp;
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          hiddenLossTmp = 0.0;
          pool_backward(poolLoss[1], poolIndex[1], hiddenLossTmp);
          hiddenLoss = hiddenLoss + hiddenLossTmp;
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          hiddenLossTmp = 0.0;
          pool_backward(poolLoss[2], poolIndex[2], hiddenLossTmp);
          hiddenLoss = hiddenLoss + hiddenLossTmp;
        }


        //gated pooling
        if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
          for (int idy = 0; idy < word_num; idy++) {
            poolIndexLoss[3][idy] = poolLoss[3] * hidden[idy];
            hiddenLoss[idy] += poolLoss[3] * poolIndex[3][idy];
          }

          for (int idy = 0; idy < word_num; idy++) {
            gateweightLoss[idy] += poolIndexLoss[3][idy] / gateweightsum;
            gateweightsumLoss -= poolIndexLoss[3][idy] * gateweight[idy] / gateweightsum / gateweightsum;
          }

          pool_backward(gateweightsumLoss, gateweightIndex, gateweightLoss);

          hiddenLossTmp = 0.0;
          for (int idy = 0; idy < word_num; idy++) {
            _gated_pooling.ComputeBackwardLoss(hidden[idy], gateweight[idy], gateweightLoss[idy], hiddenLossTmp[idy]);
            hiddenLoss[idy] += hiddenLossTmp[idy];
          }
        }

        //word convolution
        for (int idy = 0; idy < word_num; idy++) {
          _cnn_project.ComputeBackwardLoss(input[idy], hidden[idy], hiddenLoss[idy], inputLoss[idy]);
        }

        //word reshape
        for (int idy = 0; idy < word_num; idy++) {
          offset = 0;
          for (int i = 0; i < window; i++) {
            for (int j = 0; j < _token_representation_size; j++) {
              wordinputcontextLoss[idy][i][0][j] = inputLoss[idy][0][offset];
              offset++;
            }
          }
        }

        //word context
        for (int idy = 0; idy < word_num; idy++) {
          wordrepresentLoss[idy] += wordinputcontextLoss[idy][0];
          for (int idc = 1; idc <= curcontext; idc++) {
            if (idy - idc >= 0) {
              wordrepresentLoss[idy - idc] += wordinputcontextLoss[idy][2 * idc - 1];
            }
            if (idy + idc < word_num) {
              wordrepresentLoss[idy + idc] += wordinputcontextLoss[idy][2 * idc];
            }
          }
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          unconcat(wordprimeLoss[idy], charavgpoolLoss[idy], charmaxpoolLoss[idy], charminpoolLoss[idy], chargatedpoolLoss[idy], wordrepresentLoss[idy]);
        }

        //word dropout
        for (int idy = 0; idy < word_num; idy++) {
          wordprimeLoss[idy] = wordprimeLoss[idy] * wordprimeMask[idy];
        }

        //word finetune
        if (_b_wordEmb_finetune) {
          for (int idy = 0; idy < word_num; idy++) {
            offset = words[idy];
            _grad_wordEmb[offset] += wordprimeLoss[idy][0];
            _word_indexers.insert(offset);
          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          int char_num = chars[idy].size();

          //char pooling
          if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
            charhiddenLossTmp[idy] = 0.0;
            pool_backward(charavgpoolLoss[idy], charavgpoolIndex[idy], charhiddenLossTmp[idy]);
            charhiddenLoss[idy] = charhiddenLoss[idy] + charhiddenLossTmp[idy];
          }
          if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
            charhiddenLossTmp[idy] = 0.0;
            pool_backward(charmaxpoolLoss[idy], charmaxpoolIndex[idy], charhiddenLossTmp[idy]);
            charhiddenLoss[idy] = charhiddenLoss[idy] + charhiddenLossTmp[idy];
          }
          if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
            charhiddenLossTmp[idy] = 0.0;
            pool_backward(charminpoolLoss[idy], charminpoolIndex[idy], charhiddenLossTmp[idy]);
            charhiddenLoss[idy] = charhiddenLoss[idy] + charhiddenLossTmp[idy];
          }

          if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
            for (int idz = 0; idz < char_num; idz++) {
              chargatedpoolIndexLoss[idy][idz] = chargatedpoolLoss[idy] * charhidden[idy][idz];
              charhiddenLoss[idy][idz] += chargatedpoolLoss[idy] * chargatedpoolIndex[idy][idz];
            }

            for (int idz = 0; idz < char_num; idz++) {
              chargateweightLoss[idy][idz] += chargatedpoolIndexLoss[idy][idz] / chargateweightsum[idy];
              chargateweightsumLoss[idy] -= chargatedpoolIndexLoss[idy][idz] * chargateweight[idy][idz] / chargateweightsum[idy] / chargateweightsum[idy];
            }

            pool_backward(chargateweightsumLoss[idy], chargateweightIndex[idy], chargateweightLoss[idy]);

            charhiddenLossTmp[idy] = 0.0;
            for (int idz = 0; idz < char_num; idz++) {
              _gatedchar_pooling.ComputeBackwardLoss(charhidden[idy][idz], chargateweight[idy][idz], chargateweightLoss[idy][idz], charhiddenLossTmp[idy][idz]);
              charhiddenLoss[idy][idz] += charhiddenLossTmp[idy][idz];
            }
          }

          //char convolution
          for (int idz = 0; idz < char_num; idz++) {
            _cnnchar_project.ComputeBackwardLoss(charinput[idy][idz], charhidden[idy][idz], charhiddenLoss[idy][idz], charinputLoss[idy][idz]);
          }

          //reshape
          for (int idz = 0; idz < char_num; idz++) {
            offset = 0;
            for (int i = 0; i < charwindow; i++) {
              for (int j = 0; j < _charDim; j++) {
                charinputcontextLoss[idy][idz][i][0][j] = charinputLoss[idy][idz][0][offset];
                offset++;
              }
            }
          }

          //char context
          for (int idz = 0; idz < char_num; idz++) {
            charprimeLoss[idy][idz] += charinputcontextLoss[idy][idz][0];
            for (int idc = 1; idc <= _charcontext; idc++) {
              if (idz - idc >= 0) {
                charprimeLoss[idy][idz - idc] += charinputcontextLoss[idy][idz][2 * idc - 1];
              }
              if (idz + idc < char_num) {
                charprimeLoss[idy][idz + idc] += charinputcontextLoss[idy][idz][2 * idc];
              }
            }
          }

          //char dropout
          for (int idz = 0; idz < char_num; idz++) {
            charprimeLoss[idy][idz] = charprimeLoss[idy][idz] * charprimeMask[idy][idz];
          }

          //char finetune
          if (_b_charEmb_finetune) {
            for (int idz = 0; idz < char_num; idz++) {
              offset = chars[idy][idz];
              _grad_charEmb[offset] += charprimeLoss[idy][idz][0];
              _char_indexers.insert(offset);
            }
          }
        }

      }

      //release
      {
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();

        for (int idy = 0; idy < word_num; idy++) {
          FreeSpace(&(charprime[idy]));
          FreeSpace(&(charprimeLoss[idy]));
          FreeSpace(&(charprimeMask[idy]));
          FreeSpace(&(charinputcontext[idy]));
          FreeSpace(&(charinputcontextLoss[idy]));

          FreeSpace(&(charinput[idy]));
          FreeSpace(&(charinputLoss[idy]));
          FreeSpace(&(charhidden[idy]));
          FreeSpace(&(charhiddenLoss[idy]));
          FreeSpace(&(charhiddenLossTmp[idy]));
          FreeSpace(&(charavgpool[idy]));
          FreeSpace(&(charavgpoolLoss[idy]));
          FreeSpace(&(charavgpoolIndex[idy]));
          FreeSpace(&(charmaxpool[idy]));
          FreeSpace(&(charmaxpoolLoss[idy]));
          FreeSpace(&(charmaxpoolIndex[idy]));
          FreeSpace(&(charminpool[idy]));
          FreeSpace(&(charminpoolLoss[idy]));
          FreeSpace(&(charminpoolIndex[idy]));
          FreeSpace(&(chargatedpool[idy]));
          FreeSpace(&(chargatedpoolLoss[idy]));
          FreeSpace(&(chargatedpoolIndex[idy]));
          FreeSpace(&(chargatedpoolIndexLoss[idy]));
          FreeSpace(&(chargateweight[idy]));
          FreeSpace(&(chargateweightIndex[idy]));
          FreeSpace(&(chargateweightLoss[idy]));
          FreeSpace(&(chargateweightsum[idy]));
          FreeSpace(&(chargateweightsumLoss[idy]));
        }

        FreeSpace(&wordprime);
        FreeSpace(&wordprimeLoss);
        FreeSpace(&wordprimeMask);
        FreeSpace(&wordrepresent);
        FreeSpace(&wordrepresentLoss);
        FreeSpace(&wordinputcontext);
        FreeSpace(&wordinputcontextLoss);
        FreeSpace(&input);
        FreeSpace(&inputLoss);
        FreeSpace(&hidden);
        FreeSpace(&hiddenLoss);
        FreeSpace(&hiddenLossTmp);
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[idm]));
          FreeSpace(&(poolLoss[idm]));
          FreeSpace(&(poolIndex[idm]));
          FreeSpace(&(poolIndexLoss[idm]));
        }
        FreeSpace(&gateweight);
        FreeSpace(&gateweightLoss);
        FreeSpace(&gateweightIndex);
        FreeSpace(&gateweightsum);
        FreeSpace(&gateweightsumLoss);
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

    vector<Tensor<xpu, 3, double> > charprime;
    vector<Tensor<xpu, 4, double> > charinputcontext;
    vector<Tensor<xpu, 3, double> > charinput;
    vector<Tensor<xpu, 3, double> > charhidden;
    vector<Tensor<xpu, 3, double> > charavgpoolIndex, charmaxpoolIndex, charminpoolIndex;
    vector<Tensor<xpu, 2, double> > charavgpool;
    vector<Tensor<xpu, 2, double> > charmaxpool;
    vector<Tensor<xpu, 2, double> > charminpool;
    vector<Tensor<xpu, 2, double> > chargatedpool;
    vector<Tensor<xpu, 3, double> > chargatedpoolIndex;
    vector<Tensor<xpu, 3, double> > chargateweight, chargateweightIndex;
    vector<Tensor<xpu, 2, double> > chargateweightsum;

    Tensor<xpu, 3, double> wordprime;
    Tensor<xpu, 3, double> wordrepresent;
    Tensor<xpu, 4, double> wordinputcontext;
    Tensor<xpu, 3, double> input;
    Tensor<xpu, 3, double> hidden;
    vector<Tensor<xpu, 2, double> > pool(_poolmanners);
    vector<Tensor<xpu, 3, double> >  poolIndex(_poolmanners);
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
      int charwindow = 2 * _charcontext + 1;
      const Feature& feature = features[idx];
      int word_num = feature.words.size();
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      charprime.resize(word_num);
      charinputcontext.resize(word_num);
      charinput.resize(word_num);
      charhidden.resize(word_num);
      charavgpool.resize(word_num);
      charavgpoolIndex.resize(word_num);
      charmaxpool.resize(word_num);
      charmaxpoolIndex.resize(word_num);
      charminpool.resize(word_num);
      charminpoolIndex.resize(word_num);
      chargatedpool.resize(word_num);
      chargatedpoolIndex.resize(word_num);
      chargateweight.resize(word_num);
      chargateweightIndex.resize(word_num);
      chargateweightsum.resize(word_num);

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
        charinputcontext[idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
        charinput[idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
        charhidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charavgpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charavgpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charmaxpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charmaxpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charminpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charminpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargatedpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        chargatedpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweight[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightsum[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
      }

      wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
      wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
      wordinputcontext = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
      input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
      hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      }
      gateweight = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightIndex = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightsum = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
    }
    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    {
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
          charprime[idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
        }


        //char context
        for (int idz = 0; idz < char_num; idz++) {
          charinputcontext[idy][idz][0] += charprime[idy][idz];
          for (int idc = 1; idc <= _charcontext; idc++) {
            if (idz - idc >= 0) {
              charinputcontext[idy][idz][2 * idc - 1] += charprime[idy][idz - idc];
            }
            if (idz + idc < char_num) {
              charinputcontext[idy][idz][2 * idc] += charprime[idy][idz + idc];
            }
          }
        }

        //char reshape
        for (int idz = 0; idz < char_num; idz++) {
          offset = 0;
          for (int i = 0; i < charwindow; i++) {
            for (int j = 0; j < _charDim; j++) {
              charinput[idy][idz][0][offset] = charinputcontext[idy][idz][i][0][j];
              offset++;
            }
          }
        }

        //char convolution
        for (int idz = 0; idz < char_num; idz++) {
          _cnnchar_project.ComputeForwardScore(charinput[idy][idz], charhidden[idy][idz]);
        }


        //char pooling
        if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
          avgpool_forward(charhidden[idy], charavgpool[idy], charavgpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
          maxpool_forward(charhidden[idy], charmaxpool[idy], charmaxpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
          minpool_forward(charhidden[idy], charminpool[idy], charminpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
          for (int idz = 0; idz < char_num; idz++) {
            _gatedchar_pooling.ComputeForwardScore(charhidden[idy][idz], chargateweight[idy][idz]);
          }
          sumpool_forward(chargateweight[idy], chargateweightsum[idy], chargateweightIndex[idy]);
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpoolIndex[idy][idz] = chargateweight[idy][idz] / chargateweightsum[idy];
          }
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpool[idy] += charhidden[idy][idz] * chargatedpoolIndex[idy][idz];
          }
        }

      }

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }


      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        concat(wordprime[idy], charavgpool[idy], charmaxpool[idy], charminpool[idy], chargatedpool[idy], wordrepresent[idy]);
      }

      //word context
      for (int idy = 0; idy < word_num; idy++) {
        wordinputcontext[idy][0] += wordrepresent[idy];
        for (int idc = 1; idc <= curcontext; idc++) {
          if (idy - idc >= 0) {
            wordinputcontext[idy][2 * idc - 1] += wordrepresent[idy - idc];
          }
          if (idy + idc < word_num) {
            wordinputcontext[idy][2 * idc] += wordrepresent[idy + idc];
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
        _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
      }

      //word pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden, pool[0], poolIndex[0]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden, pool[1], poolIndex[1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden, pool[2], poolIndex[2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          _gated_pooling.ComputeForwardScore(hidden[idy], gateweight[idy]);
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
    {
      const Feature& feature = features[idx];
      int word_num = feature.words.size();

      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idy]));
        FreeSpace(&(charinputcontext[idy]));
        FreeSpace(&(charinput[idy]));
        FreeSpace(&(charhidden[idy]));
        FreeSpace(&(charavgpool[idy]));
        FreeSpace(&(charavgpoolIndex[idy]));
        FreeSpace(&(charmaxpool[idy]));
        FreeSpace(&(charmaxpoolIndex[idy]));
        FreeSpace(&(charminpool[idy]));
        FreeSpace(&(charminpoolIndex[idy]));
        FreeSpace(&(chargatedpool[idy]));
        FreeSpace(&(chargatedpoolIndex[idy]));
        FreeSpace(&(chargateweight[idy]));
        FreeSpace(&(chargateweightIndex[idy]));
        FreeSpace(&(chargateweightsum[idy]));
      }

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

    return optLabel;
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }

    vector<Tensor<xpu, 3, double> > charprime;
    vector<Tensor<xpu, 4, double> > charinputcontext;
    vector<Tensor<xpu, 3, double> > charinput;
    vector<Tensor<xpu, 3, double> > charhidden;
    vector<Tensor<xpu, 3, double> > charavgpoolIndex, charmaxpoolIndex, charminpoolIndex;
    vector<Tensor<xpu, 2, double> > charavgpool;
    vector<Tensor<xpu, 2, double> > charmaxpool;
    vector<Tensor<xpu, 2, double> > charminpool;
    vector<Tensor<xpu, 2, double> > chargatedpool;
    vector<Tensor<xpu, 3, double> > chargatedpoolIndex;
    vector<Tensor<xpu, 3, double> > chargateweight, chargateweightIndex;
    vector<Tensor<xpu, 2, double> > chargateweightsum;

    Tensor<xpu, 3, double> wordprime;
    Tensor<xpu, 3, double> wordrepresent;
    Tensor<xpu, 4, double> wordinputcontext;
    Tensor<xpu, 3, double> input;
    Tensor<xpu, 3, double> hidden;
    vector<Tensor<xpu, 2, double> > pool(_poolmanners);
    vector<Tensor<xpu, 3, double> >  poolIndex(_poolmanners);
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
      int charwindow = 2 * _charcontext + 1;
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      charprime.resize(word_num);
      charinputcontext.resize(word_num);
      charinput.resize(word_num);
      charhidden.resize(word_num);
      charavgpool.resize(word_num);
      charavgpoolIndex.resize(word_num);
      charmaxpool.resize(word_num);
      charmaxpoolIndex.resize(word_num);
      charminpool.resize(word_num);
      charminpoolIndex.resize(word_num);
      chargatedpool.resize(word_num);
      chargatedpoolIndex.resize(word_num);
      chargateweight.resize(word_num);
      chargateweightIndex.resize(word_num);
      chargateweightsum.resize(word_num);

      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), 0.0);
        charinputcontext[idy] = NewTensor<xpu>(Shape4(char_num, charwindow, 1, _charDim), 0.0);
        charinput[idy] = NewTensor<xpu>(Shape3(char_num, 1, _char_cnn_iSize), 0.0);
        charhidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charavgpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charavgpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charmaxpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charmaxpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        charminpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        charminpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargatedpool[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
        chargatedpoolIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweight[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightIndex[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charHiddenSize), 0.0);
        chargateweightsum[idy] = NewTensor<xpu>(Shape2(1, _charHiddenSize), 0.0);
      }

      wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), 0.0);
      wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), 0.0);
      wordinputcontext = NewTensor<xpu>(Shape4(word_num, window, 1, _token_representation_size), 0.0);
      input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), 0.0);
      hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
        poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      }
      gateweight = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightIndex = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), 0.0);
      gateweightsum = NewTensor<xpu>(Shape2(1, wordHiddenSize), 0.0);
    }
    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), 0.0);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    scores = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    {
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
          charprime[idy][idz][0] = _charEmb[offset] / _ft_charEmb[offset];
        }


        //char context
        for (int idz = 0; idz < char_num; idz++) {
          charinputcontext[idy][idz][0] += charprime[idy][idz];
          for (int idc = 1; idc <= _charcontext; idc++) {
            if (idz - idc >= 0) {
              charinputcontext[idy][idz][2 * idc - 1] += charprime[idy][idz - idc];
            }
            if (idz + idc < char_num) {
              charinputcontext[idy][idz][2 * idc] += charprime[idy][idz + idc];
            }
          }
        }

        //char reshape
        for (int idz = 0; idz < char_num; idz++) {
          offset = 0;
          for (int i = 0; i < charwindow; i++) {
            for (int j = 0; j < _charDim; j++) {
              charinput[idy][idz][0][offset] = charinputcontext[idy][idz][i][0][j];
              offset++;
            }
          }
        }

        //char convolution
        for (int idz = 0; idz < char_num; idz++) {
          _cnnchar_project.ComputeForwardScore(charinput[idy][idz], charhidden[idy][idz]);
        }


        //char pooling
        if ((_charremove > 0 && _charremove != 1) || (_charremove < 0 && _charremove == -1) || _charremove == 0) {
          avgpool_forward(charhidden[idy], charavgpool[idy], charavgpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 2) || (_charremove < 0 && _charremove == -2) || _charremove == 0) {
          maxpool_forward(charhidden[idy], charmaxpool[idy], charmaxpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 3) || (_charremove < 0 && _charremove == -3) || _charremove == 0) {
          minpool_forward(charhidden[idy], charminpool[idy], charminpoolIndex[idy]);
        }
        if ((_charremove > 0 && _charremove != 4) || (_charremove < 0 && _charremove == -4) || _charremove == 0) {
          for (int idz = 0; idz < char_num; idz++) {
            _gatedchar_pooling.ComputeForwardScore(charhidden[idy][idz], chargateweight[idy][idz]);
          }
          sumpool_forward(chargateweight[idy], chargateweightsum[idy], chargateweightIndex[idy]);
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpoolIndex[idy][idz] = chargateweight[idy][idz] / chargateweightsum[idy];
          }
          for (int idz = 0; idz < char_num; idz++) {
            chargatedpool[idy] += charhidden[idy][idz] * chargatedpoolIndex[idy][idz];
          }
        }

      }

      for (int idy = 0; idy < word_num; idy++) {
        offset = words[idy];
        wordprime[idy][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }


      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        concat(wordprime[idy], charavgpool[idy], charmaxpool[idy], charminpool[idy], chargatedpool[idy], wordrepresent[idy]);
      }

      //word context
      for (int idy = 0; idy < word_num; idy++) {
        wordinputcontext[idy][0] += wordrepresent[idy];
        for (int idc = 1; idc <= curcontext; idc++) {
          if (idy - idc >= 0) {
            wordinputcontext[idy][2 * idc - 1] += wordrepresent[idy - idc];
          }
          if (idy + idc < word_num) {
            wordinputcontext[idy][2 * idc] += wordrepresent[idy + idc];
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
        _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
      }

      //word pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden, pool[0], poolIndex[0]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden, pool[1], poolIndex[1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden, pool[2], poolIndex[2]);
      }

      //gated pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        for (int idy = 0; idy < word_num; idy++) {
          _gated_pooling.ComputeForwardScore(hidden[idy], gateweight[idy]);
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
    {
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();

      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idy]));
        FreeSpace(&(charinputcontext[idy]));
        FreeSpace(&(charinput[idy]));
        FreeSpace(&(charhidden[idy]));
        FreeSpace(&(charavgpool[idy]));
        FreeSpace(&(charavgpoolIndex[idy]));
        FreeSpace(&(charmaxpool[idy]));
        FreeSpace(&(charmaxpoolIndex[idy]));
        FreeSpace(&(charminpool[idy]));
        FreeSpace(&(charminpoolIndex[idy]));
        FreeSpace(&(chargatedpool[idy]));
        FreeSpace(&(chargatedpoolIndex[idy]));
        FreeSpace(&(chargateweight[idy]));
        FreeSpace(&(chargateweightIndex[idy]));
        FreeSpace(&(chargateweightsum[idy]));
      }

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
    _cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _cnnchar_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gatedchar_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _gated_pooling.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

    checkgrad(examples, _gated_pooling._W, _gated_pooling._gradW, "_gated_pooling._W", iter);
    checkgrad(examples, _gated_pooling._b, _gated_pooling._gradb, "_gated_pooling._b", iter);

    checkgrad(examples, _gatedchar_pooling._W, _gatedchar_pooling._gradW, "_gatedchar_pooling._W", iter);
    checkgrad(examples, _gatedchar_pooling._b, _gatedchar_pooling._gradb, "_gatedchar_pooling._b", iter);

    checkgrad(examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
    checkgrad(examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

    checkgrad(examples, _cnnchar_project._W, _cnnchar_project._gradW, "_cnnchar_project._W", iter);
    checkgrad(examples, _cnnchar_project._b, _cnnchar_project._gradb, "_cnnchar_project._b", iter);

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

#endif /* SRC_CNNCharClassifier_H_ */
