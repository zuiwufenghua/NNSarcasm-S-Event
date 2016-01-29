/*
 * Labeler.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "SEventDetector.h"

#include "Argument_helper.h"

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";

}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
}

string Labeler::getRealLabel(const string rawLabel){
  string firstWord, secondWord;
  string::size_type pos = rawLabel.find("@@", 0);
  if(pos != string::npos){
    firstWord = rawLabel.substr(0, pos);
    string::size_type pos2 = rawLabel.find("%%%", pos+2);
    string::size_type pos3 = rawLabel.find("@@", pos2+3);
    if(pos2 != string::npos && pos3 != string::npos){
      secondWord = rawLabel.substr(pos2+3, pos3 - pos2 -3);
    }
  }
	//make sure dictionary order
	string realLabel = "";
	if (firstWord.compare(secondWord) > 0) {
		realLabel = secondWord + "%%%" + firstWord;
	} else {
		realLabel = firstWord + "%%%" + secondWord;
	}
	return realLabel;
}

int Labeler::createAlphabet(const string& insFile) {
  cout << "Creating Alphabet..." << endl;

  int numInstance = 0, labelId;
  hash_map<string, int> word_stat;
  m_labelAlphabet.clear();

  m_pipe.initInputFile(insFile.c_str());
  Instance *pInstance = m_pipe.nextInstance();
  while (pInstance) {

    if (pInstance->size() < 300) {
      const vector<vector<string> > &words = pInstance->words;

      const string &label = pInstance->label;
      //multiple label per instance, split by '##'
      vector<string> labels;
      split_bystr(label, labels, "##");
      for (int i = 0; i < labels.size(); i++) {
        string realLabel = getRealLabel(labels[i]);
        //      cout << "realLabel: " << realLabel << endl;
        m_labelAlphabet.from_string(realLabel);
      }

      int sentNum = words.size();
      for (int i = 0; i < sentNum; ++i) {
        int curSentLength = words[i].size();
        for (int j = 0; j < curSentLength; j++) {
          string curword = normalize_to_lowerwithdigit(words[i][j]);
          word_stat[curword]++;
        }
      }
      numInstance++;
    } else {
      cout << "sentNum exceed 300!" << endl;
    }
    if (numInstance % 1000 == 0)
      cout << "have read instances num: " << numInstance << endl;

    pInstance = m_pipe.nextInstance();
  }
  m_pipe.uninitInputFile();

  cout << "m_strInFile: " << insFile << ", instance num: " << numInstance << endl;

  cout << "Label num: " << m_labelAlphabet.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;

  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);

  hash_map<string, int>::iterator feat_iter;

  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  m_labelAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding word Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> word_stat;
  m_wordAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<string> > &words = pInstance->words;

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      int curWordSize = words[i].size();
      for (int j = 0; j < curWordSize; j++) {
        string curword = normalize_to_lowerwithdigit(words[i][j]);
        word_stat[curword]++;
      }
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestCharAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding char Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> char_stat;
  m_charAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  m_charAlphabet.set_fixed_flag(true);

  return 0;
}

void Labeler::extractLinearFeatures(vector<string>& features, const Instance* pInstance) {
  features.clear();
  const vector<vector<string> >& words = pInstance->words;
  int seq_size = words.size();

  string feat = "";
  const vector<string> lastWords = words[seq_size - 1];
  int wordnumber = lastWords.size();
  for (int i = 0; i < wordnumber; i++) {
    feat = "F1U=" + lastWords[i];
    features.push_back(feat);
    string prevword = i - 1 >= 0 ? lastWords[i - 1] : nullkey;
    feat = "F2B=" + prevword + seperateKey + lastWords[i];
    features.push_back(feat);
//    string prev2word = i - 2 >= 0 ? lastWords[i - 2] : nullkey;
//    feat = "F3T=" + prev2word + seperateKey + prevword + seperateKey + lastWords[i];
//    features.push_back(feat);
  }

  if (m_linearfeat > 1 && seq_size == 2) {
    vector<string> lastWords = words[seq_size - 2];
    wordnumber = lastWords.size();
    for (int i = 0; i < wordnumber; i++) {
      feat = "F4U=" + lastWords[i];
      features.push_back(feat);
    }
  }

}

void Labeler::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<vector<string> >& words = pInstance->words;

  int sentsize = words.size();

  if (idx < 0 || idx >= sentsize)
    return;

  int wordNum = words[idx].size();
  feat.words.resize(wordNum);

  int unknownWordId = m_wordAlphabet.from_string(unknownkey);

  for (int i = 0; i < wordNum; i++) {
    string curWord = normalize_to_lowerwithdigit(words[idx][i]);
    int curWordId = m_wordAlphabet.from_string(curWord);
    if (curWordId >= 0)
      feat.words[i] = curWordId;
    else
      feat.words[i] = unknownWordId;
  }
}

void Labeler::convert2Example(const Instance* pInstance, Example& exam) {
  exam.clear();
  const string &label = pInstance->label;
  vector<string> labels;
  hash_set<string> labelSet;
  split_bystr(label, labels, "##");
  for(int i = 0; i < labels.size(); i++){
    string realLabel = getRealLabel(labels[i]);
    labelSet.insert(realLabel);
  }
  const vector<vector<string> > &words = pInstance->words;

  int numLabels = m_labelAlphabet.size();
  exam.m_labels.resize(numLabels);
  for (int j = 0; j < numLabels; ++j) {
    string str = m_labelAlphabet.from_id(j);
    if(labelSet.find(str) == labelSet.end()){
      //not found
      exam.m_labels[j] = false;
    }else
      exam.m_labels[j] = true;
  }

  int sentNum = words.size();
  exam.m_features.resize(sentNum);
  for (int i = 0; i < sentNum; ++i) {
    Feature& feat = exam.m_features[i];
    extractFeature(feat, pInstance, i);
  }
}

void Labeler::initialExamples(const string& insFile, vector<Example>& vecExams) {

  m_pipe.initInputFile(insFile.c_str());
  Instance *pInstance = m_pipe.nextInstance();
  int numInstance = 0;
  while (pInstance) {

    if (pInstance->size() < 300) {
      Example& curExam = vecExams[numInstance];
      convert2Example(pInstance, curExam);
      numInstance++;
    } else {
      cout << "sentNum exceed 300!" << endl;
    }
    if (numInstance % 1000 == 0)
      cout << "have build example num: " << numInstance << endl;

    pInstance = m_pipe.nextInstance();
  }
  m_pipe.uninitInputFile();

  cout << "m_strInFile: " << insFile << ", example num: " << numInstance << endl;
}

void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}

void Labeler::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile, const string& charEmbFile) {
  if (optionFile != "")
    m_options.load(optionFile);

  m_options.showOptions();

  m_linearfeat = 0;

  int trainNum = 188450, devNum = 23556, testNum = 23540;

  vector<Instance> trainInsts(trainNum), devInsts(devNum), testInsts(testNum);
  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool bCurIterBetter = false;

  std::cout << "TrainIns num: " << trainNum << std::endl;
  std::cout << "DevIns num: " << devNum << std::endl;
  std::cout << "TestIns num: " << testNum << std::endl;

  createAlphabet(trainFile);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(devInsts);
    addTestWordAlpha(testInsts);
    cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  }

  if (!m_options.charEmbFineTune) {
    addTestCharAlpha(devInsts);
    addTestCharAlpha(testInsts);
    cout << "Remain char num: " << m_charAlphabet.size() << endl;
  }

  NRMat<double> wordEmb;
  if (wordEmbFile != "") {
    readWordEmbeddings(wordEmbFile, wordEmb);
  } else {
    wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
    wordEmb.randu(1000);
  }

  NRMat<double> charEmb;
  if (charEmbFile != "") {
    readWordEmbeddings(charEmbFile, charEmb);
  } else {
    charEmb.resize(m_charAlphabet.size(), m_options.charEmbSize);
    charEmb.randu(1001);
  }

  m_classifier.init(wordEmb, m_options.wordcontext, m_labelAlphabet.size(), m_options.wordHiddenSize, m_options.hiddenSize);
  m_classifier.resetRemove(m_options.removePool);
  m_classifier.setDropValue(m_options.dropProb);
  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);

  cout << "init examples..." << endl;
  vector<Example> trainExamples(trainNum), devExamples(devNum), testExamples(testNum);
  initialExamples(trainFile, trainExamples);
  initialExamples(devFile, devExamples);
  initialExamples(testFile, testExamples);

  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  vector<int> otherInstNums(m_options.testFiles.size());
  vector<vector<Example> > otherExamples(m_options.testFiles.size());
  for (int idx = 0; idx < otherInstNums.size(); idx++) {
    initialExamples(otherInsts[idx], otherExamples[idx]);
    otherInstNums[idx] = otherExamples[idx].size();
  }

  double bestDIS = 0;

  int inputSize = trainExamples.size();
  int batchBlock = inputSize / m_options.batchSize;
    if (inputSize % m_options.batchSize != 0)
      batchBlock++;

  srand(0);
  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;
  static vector<Example> subExamples;

  double cost = 0.0;
  for (int iter = 0; iter < m_options.maxIter; ++iter) {

    std::cout << "##### Iteration " << iter << "/" << m_options.maxIter << std::endl;

//    random_shuffle(indexes.begin(), indexes.end());
    eval.reset();
    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
      subExamples.clear();
//      int start_pos = updateIter * m_options.batchSize;
//      int end_pos = (updateIter + 1) * m_options.batchSize;
//      if (end_pos > inputSize)
//        end_pos = inputSize;

//      for (int idy = start_pos; idy < end_pos; idy++) {
//        subExamples.push_back(trainExamples[indexes[idy]]);
//      }
      random_shuffle(indexes.begin(), indexes.end());
      for (int idy = 0; idy < m_options.batchSize; idy++) {
        subExamples.push_back(trainExamples[indexes[idy]]);
      }

      int curUpdateIter = iter * batchBlock + updateIter;
      double cost = m_classifier.process(subExamples, curUpdateIter);

      eval.overall_label_count += m_classifier._eval.overall_label_count;
      eval.correct_label_count += m_classifier._eval.correct_label_count;

      std:cout << "have finished " << updateIter << "/" << batchBlock << " batches" << endl;

      if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
        //m_classifier.checkgrads(subExamples, curUpdateIter+1);
        std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
        std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
      }
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);

    }

    if (devNum > 0) {

      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devExamples.size(); idx++) {

        vector<double> results;
        double confidence = predict(devExamples[idx], results);
        evaluate(devExamples[idx], results, metric_dev);

        if(idx % 1000 == 0) cout << "have evaluated " << idx << " devIns" << endl;
      }
      std::cout << "dev: " << endl;
      metric_dev.print();

      if ((!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS)) {
//        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testExamples.size(); idx++) {

        	vector<double> results;
			double confidence = predict(testExamples[idx], results);
			evaluate(testExamples[idx], results, metric_dev);

          if(idx % 1000 == 0) cout << "have evaluated " << idx << " testIns" << endl;
        }
        std::cout << "test:" << std::endl;
        metric_test.print();
      }

      for (int idx = 0; idx < otherExamples.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idy = 0; idy < otherExamples[idx].size(); idy++) {

        	vector<double> results;
			double confidence = predict(otherExamples[idx][idy], results);
			evaluate(otherExamples[idx][idy], results, metric_dev);

          if(idx % 1000 == 0) cout << "have evaluated " << idx << " otherIns" << endl;
        }
        std::cout << "other test:" << std::endl;
        metric_test.print();
      }

      if ((m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS)) {
        if (metric_dev.getAccuracy() > bestDIS) {
          std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
          bestDIS = metric_dev.getAccuracy();
        }
        writeModelFile(modelFile);
      }

    }
    // Clear gradients
  }

  //finish training, then valid
  if (devNum > 0) {
    bCurIterBetter = false;
    if (!m_options.outBest.empty())
      decodeInstResults.clear();
    metric_dev.reset();
    for (int idx = 0; idx < devExamples.size(); idx++) {
    	vector<double> results;
    	double confidence = predict(devExamples[idx], results);
    	evaluate(devExamples[idx], results, metric_dev);
    	if(idx % 1000 == 0) cout << "have evaluated " << idx << " devIns" << endl;
    }
    std::cout << "have finished training, show dev result: " << endl;
    metric_dev.print();

    if ((!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS)) {
      m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
      bCurIterBetter = true;
    }

    if (testNum > 0) {
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_test.reset();
      for (int idx = 0; idx < testExamples.size(); idx++) {
    	  vector<double> results;
    	  double confidence = predict(testExamples[idx], results);
    	  evaluate(testExamples[idx], results, metric_dev);

    	  if(idx % 1000 == 0) cout << "have evaluated " << idx << " testIns" << endl;
      }
      std::cout << "have finished training, show test result: " << endl;
      metric_test.print();

      if ((!m_options.outBest.empty() && bCurIterBetter)) {
        m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
      }
    }

    for (int idx = 0; idx < otherExamples.size(); idx++) {
      std::cout << "processing " << m_options.testFiles[idx] << std::endl;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_test.reset();
      for (int idy = 0; idy < otherExamples[idx].size(); idy++) {

    	  vector<double> results;
    	  double confidence = predict(otherExamples[idx][idy], results);
    	  evaluate(otherExamples[idx][idy], results, metric_dev);

    	  if(idx % 1000 == 0) cout << "have evaluated " << idx << " otherIns" << endl;
      }
      std::cout << "have finished training, show other test result:" << std::endl;
      metric_test.print();

      if ((!m_options.outBest.empty() && bCurIterBetter)) {
        m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
      }
    }

    if ((m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS)) {
      if (metric_dev.getAccuracy() > bestDIS) {
        std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
        bestDIS = metric_dev.getAccuracy();
      }
      writeModelFile(modelFile);
    }

  } else {
    writeModelFile(modelFile);
  }
}

void Labeler::evaluate(const Example& example, vector<double>& results, Metric& eval){
	//topK score
	int topK = m_classifier.topK;
	int _labelSize = m_classifier._labelSize;
	int topKIndex[topK];
	double topKScores[topK];
	for (int i = 0; i < topK; i++) {
		topKScores[i] = -1e10;
		for (int j = 0; j < _labelSize; j++) {
			if (results[j] > topKScores[i]) {
				topKScores[i] = results[j];
				topKIndex[i] = j;
			}
		}
		results[topKIndex[i]] = -1e12;   //remove temporarily
	}
	for (int i = 0; i < topK; i++)
		results[topKIndex[i]] = topKScores[i];  //recovery

	//correctness judge
	bool correct = true;
	int hitPosNum = 0;
	for (int i = 0; _labelSize; i++) {
		if (example.m_labels[i]) {
			bool find = false;
			for (int j = 0; j < topK; j++) {
				if (i == topKIndex[j]) {
					find = true;
					hitPosNum++;
					break;
				}
			}
			if (!find) {
				if (hitPosNum >= topK)
					cout << "hitPosNum exceed topK: " << topK << endl;
				correct = false;
				break;
			}
		}
	}
	if (correct)
      eval.correct_label_count++;
    eval.overall_label_count++;
  }

double Labeler::predict(const Example& example, vector<double>& labelprobs) {
  int label = m_classifier.predict(example.m_features, labelprobs);
  return labelprobs[label];
}

void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
	  vector<double> results;
	  double confidence = predict(testExamples[idx], results);
	  evaluate(testExamples[idx], results, metric_test);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  m_pipe.outputAllInstances(outputFile, testInstResults);

}

void Labeler::readWordEmbeddings(const string& inFile, NRMat<double>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = atoi(vecInfo[1].c_str());

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  hash_set<int> indexers;
  double sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        double curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        wordEmb[wordId][idx] = curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Labeler::loadModelFile(const string& inputModelFile) {
	m_classifier.loadModel(inputModelFile);
}

void Labeler::writeModelFile(const string& outputModelFile) {
	m_classifier.writeModel(outputModelFile);
}

int main(int argc, char* argv[]) {
  std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
  std::string wordEmbFile = "", charEmbFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = true;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if (bTrain) {
    tagger.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmbFile, charEmbFile);
  } else {
    tagger.test(testFile, outputFile, modelFile);
  }

}
