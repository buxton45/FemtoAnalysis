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
                          CentralityType aCentType=k0010, 
                          TString aFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/Results_cXicKch_20170505_ignoreOnFlyStatus");
  virtual ~SimpleChargedResidualCf();

  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");
  static td1dVec ConvertHistTo1dVec(TH1* aHist);
  void SetDaughtersAndMothers();

  void LoadCoulombOnlyInterpCfs(TString aFileDirectory="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/", bool aUseCoulombOnlyInterpCfs=true, double aRadiusFactor=1.);
  td1dVec ExtractCfFrom2dInterpCfs(double aRadius);

  td1dVec GetChargedResidualCorrelation(double aRadiusParam=-1.);

  td1dVec GetTransformedChargedResidualCorrelation(double aRadiusParam=-1.);

  TH1D* GetChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam=-1.);
  TH1D* GetTransformedChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam=-1.);

  td1dVec GetContributionToFitCf(double aOverallLambda, double aRadiusParam=-1.);
  TH1D* GetChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aRadiusParam=-1.);
  TH1D* GetTransformedChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aRadiusParam=-1.);

  void SetUsemTScalingOfRadii(AnalysisType aParentAnType, double aPower=-0.5);

  //inline
  AnalysisType GetResidualType();
  double GetLambdaFactor();
  void SetUseCoulombOnlyInterpCfs(bool aUse);
  TH2D* GetTransformMatrix();

  ParticlePDGType GetMotherType1();
  ParticlePDGType GetDaughterType1();
  ParticlePDGType GetMotherType2();
  ParticlePDGType GetDaughterType2();

  void SetRadiusFactor(double aFactor);

protected:
  AnalysisType fResidualType;
  ParticlePDGType fDaughterType1, fMotherType1;
  ParticlePDGType fDaughterType2, fMotherType2;
  FitPairAnalysis* fPairAn;
  TH1D* fExpXiHist;
  double fLambdaFactor;
  TH2D* fTransformMatrix;
  td1dVec fKStarBinCenters;
  td1dVec fResCf;
  td1dVec fTransformedResCf;

  bool fUseCoulombOnlyInterpCfs;
  TH2D* f2dCoulombOnlyInterpCfs;
  double fRadiusFactor;  //Allow me to give SigSt different radii


#ifdef __ROOT__
  ClassDef(SimpleChargedResidualCf, 1)
#endif
};

inline AnalysisType SimpleChargedResidualCf::GetResidualType() {return fResidualType;}
inline double SimpleChargedResidualCf::GetLambdaFactor() {return fLambdaFactor;}
inline void SimpleChargedResidualCf::SetUseCoulombOnlyInterpCfs(bool aUse) {fUseCoulombOnlyInterpCfs = aUse;}
inline TH2D* SimpleChargedResidualCf::GetTransformMatrix() {return fTransformMatrix;}

inline ParticlePDGType SimpleChargedResidualCf::GetMotherType1() {return fMotherType1;}
inline ParticlePDGType SimpleChargedResidualCf::GetDaughterType1() {return fDaughterType1;}
inline ParticlePDGType SimpleChargedResidualCf::GetMotherType2() {return fMotherType2;}
inline ParticlePDGType SimpleChargedResidualCf::GetDaughterType2() {return fDaughterType2;}

inline void SimpleChargedResidualCf::SetRadiusFactor(double aFactor) {fRadiusFactor = aFactor;}
#endif
