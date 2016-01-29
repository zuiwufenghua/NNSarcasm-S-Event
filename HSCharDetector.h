/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_Detector_H_
#define SRC_Detector_H_



#include "Alphabet.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"

#include "Pipe.h"
#include "Utf.h"
#include "NRMat.h"
#include "basic/CNNHSCharClassifier.h"


using namespace nr;
using namespace std;


class Labeler {

public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:
  Alphabet m_labelAlphabet;
  Alphabet m_featAlphabet;
  Alphabet m_wordAlphabet;
  Alphabet m_charAlphabet;


public:
  Options m_options;

  Pipe m_pipe;

  int m_linearfeat;


#if USE_CUDA==1
  CNNHSCharClassifier<gpu> m_classifier;
#else
  CNNHSCharClassifier<cpu> m_classifier;
#endif


public:
  void readWordEmbeddings(const string& inFile, NRMat<double>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);

  void extractFeature(Feature& feat, const Instance* pInstance, int idx);
  void extractLinearFeatures(vector<string>& features, const Instance* pInstance);
  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile);
  double predict(const vector<Feature>& features, string& output);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);


};

#endif /* SRC_Detector_H_ */
