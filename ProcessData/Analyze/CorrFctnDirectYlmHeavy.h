
#ifndef _CORRFCTNDIRECTYLMHEAVY_H_
#define _CORRFCTNDIRECTYLMHEAVY_H_

#include "CorrFctnDirectYlmLite.h"

using namespace std;

class CorrFctnDirectYlmHeavy{
 public:
  CorrFctnDirectYlmHeavy(vector<CorrFctnDirectYlmLite*> &aYlmCfLiteCollection);
  ~CorrFctnDirectYlmHeavy();


  TH1D* GetYlmCfnHist(YlmComponent aComponent, int al, int am);

  //inline
  vector<CorrFctnDirectYlmLite*> GetYlmCfLiteCollection() const;

 private:
  vector<CorrFctnDirectYlmLite*> fYlmCfLiteCollection;


  
};

inline vector<CorrFctnDirectYlmLite*> CorrFctnDirectYlmHeavy::GetYlmCfLiteCollection() const {return fYlmCfLiteCollection;}

#endif

