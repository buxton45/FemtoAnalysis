/* SimpleChargedResidualCf.h */

#ifndef SIMPLECHARGEDRESIDUALCF_H
#define SIMPLECHARGEDRESIDUALCF_H

#include <cassert>
#include <iostream>
#include <math.h>

#include "TH2.h"
#include "TMath.h"

#include "Types.h"

class FitPairAnalysis;

using namespace std;

class SimpleChargedResidualCf {

public:
  SimpleChargedResidualCf(AnalysisType aResidualType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters,
                          CentralityType aCentType=k0010, double aMaxKStar=1.0, 
                          TString aFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/Results_cXicKch_20170505_ignoreOnFlyStatus");
  virtual ~SimpleChargedResidualCf();

  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");

  td1dVec GetChargedResidualCorrelation(double aMaxKStar=1.0);

  td1dVec GetTransformedChargedResidualCorrelation(double aMaxKStar=1.0);

  TH1D* GetChargedResidualCorrelationHistogram(TString aTitle, double aMaxKStar=1.0);
  TH1D* GetTransformedChargedResidualCorrelationHistogram(TString aTitle, double aMaxKStar=1.0);

  td1dVec GetContributionToFitCf(double aOverallLambda, double aMaxKStar=1.0);
  TH1D* GetChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aMaxKStar=1.0);
  TH1D* GetTransformedChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aMaxKStar=1.0);

  //inline
  AnalysisType GetResidualType();
  TH1D* GetChargedResidualCorrelationHistogram(TString aTitle="fResCf");
  TH1D* GetTransformedChargedResidualCorrelationHistogram(TString aTitle="fTransformedResCf");
  double GetLambdaFactor();
protected:
AnalysisType fResidualType;
FitPairAnalysis* fPairAn;
TH1D* fExpXiHist;
double fLambdaFactor;
TH2D* fTransformMatrix;
td1dVec fKStarBinCenters;
td1dVec fResCf;
td1dVec fTransformedResCf;



#ifdef __ROOT__
  ClassDef(SimpleChargedResidualCf, 1)
#endif
};

inline AnalysisType SimpleChargedResidualCf::GetResidualType() {return fResidualType;}
inline TH1D* SimpleChargedResidualCf::GetChargedResidualCorrelationHistogram(TString aTitle) {return Convert1dVecToHist(fResCf, fKStarBinCenters, aTitle);}
inline TH1D* SimpleChargedResidualCf::GetTransformedChargedResidualCorrelationHistogram(TString aTitle) {return Convert1dVecToHist(fTransformedResCf, fKStarBinCenters, aTitle);}
inline double SimpleChargedResidualCf::GetLambdaFactor() {return fLambdaFactor;}
#endif
