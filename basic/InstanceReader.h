#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "MyLib.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
  InstanceReader() {
  }
  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();

    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

    if(vecLine.size() < 3) {
//      string errIns;
//      join_bystr(vecLine, errIns, "@");
//      cout << "vecLine.size = " << vecLine.size() << ", error ins: " << errIns << endl;
      return NULL;
    }

    int counter = 0;
    string filename = vecLine[counter++];
    string headline = vecLine[counter++];
    string labels = vecLine[counter++];
    int sentNum = vecLine.size() - counter;
//    cout << "filename: " << filename << ", sentNum: " << sentNum << endl;

    m_instance.allocate(1);
    m_instance.label = labels;
    for(int i = 0; i < sentNum; i++){
      vector<string> vecInfo;
      split_bychar(vecLine[i+counter], vecInfo, ' ');
      for(int j = 0; j < vecInfo.size(); j++){
        m_instance.words[0].push_back(vecInfo[j]);
      }
    }

    return &m_instance;
  }
};

#endif

