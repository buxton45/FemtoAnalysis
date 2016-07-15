///////////////////////////////////////////////////////////////////////////
// FitChi2Histograms:                                                    //
///////////////////////////////////////////////////////////////////////////

#ifndef FITCHI2HISTOGRAMS_H
#define FITCHI2HISTOGRAMS_H

#include "TObject.h"
#include "TObjArray.h"
#include "TH2D.h"
#include "TH3D.h"
#include <cassert>
#include "TFile.h"

#include "FitParameter.h"
class FitParameter;

class FitChi2Histograms {

public:

  FitChi2Histograms();
  virtual ~FitChi2Histograms();

  int Factorial(int aInput);
  int nChoosek(int aN, int aK);

  BinInfo GetBinInfo(ParameterType aParamType);
  void SetupBinInfo(ParameterType aParamType, int aNBins, double aMin, double aMax);
 // void SetupBinInfo(ParameterType aParamType, int aNBins, double aMin, double aMax);
  void SetupChi2BinInfo(int aNBins, double aMin, double aMax);
  void AddParameter(int aMinuitParamNumber, FitParameter* aParam);
  void InitiateHistograms();

  void FillHistograms(double aChi2, double *aParams);

  void SaveHistograms(TString aFileName);




private:

  vector<FitParameter*> fMinuitFitParameters;
  TObjArray *fChi2HistCollection;
  TObjArray *fInvChi2HistCollection;
  TObjArray *fChi2CountsHistCollection;  //will be used to normalize fChi2HistCollection

  BinInfo fChi2BinInfo;

  BinInfo fLambdaBinInfo;
  BinInfo fRadiusBinInfo;
  BinInfo fReF0sBinInfo;
  BinInfo fImF0sBinInfo;
  BinInfo fD0sBinInfo;
  BinInfo fReF0tBinInfo;
  BinInfo fImF0tBinInfo;
  BinInfo fD0tBinInfo;





#ifdef __ROOT__
  ClassDef(FitChi2Histograms, 1)
#endif
};


#endif

