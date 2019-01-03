
#ifndef _CORRFCTNDIRECTYLMTHERM_H_
#define _CORRFCTNDIRECTYLMTHERM_H_

// Correlation function that is binned in Ylms directly
// Provides a way to store the numerator and denominator
// in Ylms directly and correctly calculate the correlation
// function from them

#include <assert.h>
#include <math.h>
#include <complex>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TFile.h>
#include <TObjArray.h>

#include "Types.h"
#include "CorrFctnDirectYlm.h"

using namespace std;

class CorrFctnDirectYlmTherm : public CorrFctnDirectYlm {
 public:
  CorrFctnDirectYlmTherm(TString aFileNameBase, int aImpactParam, AnalysisType aAnType, int aMaxl, int aNbins, double aKStarMin, double aKStarMax, int aRebin, double aNumScale=0.);
  ~CorrFctnDirectYlmTherm();

  TH1* Get1dHisto(TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aHistoName, TString aNewName);
  TH3* Get3dHisto(TString aHistoName, TString aNewName);

  void ReadFromThermFile(int aRebin=1);

  TH1D* GetYlmHist(YlmComponent aComponent, YlmHistType aHistType, int al, int am);

  //inline--------------------------
  double GetNumScale();
  AnalysisType GetAnalysisType();
  int GetImpactParam();

 private:
  TFile *fFile;
  int fImpactParam;
  AnalysisType fAnalysisType;
  TString fSavedNameMod;
  TString fNewNameMod;
  double fNumScale; //Taken from normal KStarCf
  int fRebin;
  
};

inline double CorrFctnDirectYlmTherm::GetNumScale() {return fNumScale;}
inline AnalysisType CorrFctnDirectYlmTherm::GetAnalysisType() {return fAnalysisType;}
inline int CorrFctnDirectYlmTherm::GetImpactParam() {return fImpactParam;}

#endif

