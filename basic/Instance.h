#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "MyLib.h"
#include "Metric.h"

using namespace std;

class Instance {
public:
  Instance() {
  }
  ~Instance() {
  }

  int size() const {
    return words.size();
  }

  void clear() {
    label = "";
    for (int i = 0; i < size(); i++) {
      words[i].clear();
    }
    words.clear();
    confidence = -1.0;
  }

  void allocate(int length) {
    clear();
    label = "";
    words.resize(length);
    confidence = -1.0;
  }

  void copyValuesFrom(const Instance& anInstance) {
    allocate(anInstance.size());
    for (int i = 0; i < anInstance.size(); i++) {
      for (int j = 0; j < anInstance.words[i].size(); j++) {
        words[i].push_back(anInstance.words[i][j]);
      }
    }
    label = anInstance.label;
    confidence = anInstance.confidence;
  }

  void assignLabel(const string& resulted_label) {
    label = resulted_label;
  }

  void assignLabel(const string& resulted_label, double resulted_confidence) {
    label = resulted_label;
    confidence = resulted_confidence;
  }

  void Evaluate(const string& resulted_label, Metric& eval) const {
    if (resulted_label.compare(label) == 0)
      eval.correct_label_count++;
    eval.overall_label_count++;
  }

public:
  string label;
  vector<vector<string> > words;
  double confidence;

};

#endif

