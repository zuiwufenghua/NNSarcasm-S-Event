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
#include "basic/CNNWordClassifier.h"


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

public:
  const string labelSpliter = "|";


#if USE_CUDA==1
  CNNWordClassifier<gpu> m_classifier;
#else
  CNNWordClassifier<cpu> m_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<double>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  string getRealLabel(const string rawLabel);
  int createAlphabet(const string& insFile);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);

  void extractFeature(Feature& feat, const Instance* pInstance, int idx);
  void extractLinearFeatures(vector<string>& features, const Instance* pInstance);
  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const string& insFile, vector<Example>& vecExams);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void evaluate(const Example& example, vector<double>& results, Metric& eval);
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile);
  double predict(const Example& example, vector<double>& labelprobs);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);


};

#endif /* SRC_Detector_H_ */
