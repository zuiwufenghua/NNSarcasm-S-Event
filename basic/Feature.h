/*
 * Feature.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;
class Feature {

public:
  vector<int> words;

public:
  Feature() {
  }
  virtual ~Feature() {

  }

  void clear() {
    words.clear();
  }
};

#endif /* SRC_FEATURE_H_ */
