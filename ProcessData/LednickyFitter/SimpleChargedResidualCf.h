/* SimpleChargedResidualCf.h */

#ifndef SIMPLECHARGEDRESIDUALCF_H
#define SIMPLECHARGEDRESIDUALCF_H

#include <cassert>
#include <iostream>
#include <math.h>

#include <omp.h>

#include "TH2.h"
#include "TMath.h"

#include "Types.h"
#include "Types_LambdaValues.h"

class FitPairAnalysis;

using namespace std;

class SimpleChargedResidualCf {

public:
  SimpleChargedResidualCf(AnalysisType aResidualType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters,
                          CentralityType aCentType=k0010, 
                          TString aFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/Results_cXicKch_20170505_ignoreOnFlyStatus");
  virtual ~SimpleChargedResidualCf();

  static TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle = "tCf");
  static td1dVec ConvertHistTo1dVec(TH1* aHist);

  void SetDaughtersAndMothers();

  void BuildExpXiHist(TString aFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/Results_cXicKch_20170505_ignoreOnFlyStatus");
  void LoadCoulombOnlyInterpCfs(TString aFileDirectory="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/", double aRadiusFactor=1.);

  td1dVec ExtractCfFrom2dInterpCfs(double aRadius);

  bool IsRadiusParamSame(double aRadiusParam);
  td1dVec GetChargedResidualCorrelation(double aRadiusParam);

  td1dVec GetTransformedChargedResidualCorrelation(double aRadiusParam);

  TH1D* GetChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam);
  TH1D* GetTransformedChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam);

  td1dVec GetContributionToFitCf(double *aParamsOverall);
  TH1D* GetResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall);
  TH1D* GetTransformedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall);

  void SetUsemTScalingOfRadii(AnalysisType aParentAnType, double aPower=-0.5);

  //inline
  AnalysisType GetResidualType();
  double GetLambdaFactor();
  TH2D* GetTransformMatrix();

  ParticlePDGType GetMotherType1();
  ParticlePDGType GetDaughterType1();
  ParticlePDGType GetMotherType2();
  ParticlePDGType GetDaughterType2();

  void SetRadiusFactor(double aFactor);

protected:
  AnalysisType fResidualType;
  CentralityType fCentralityType;
  IncludeResidualsType fIncludeResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;
  ParticlePDGType fDaughterType1, fMotherType1;
  ParticlePDGType fDaughterType2, fMotherType2;
  FitPairAnalysis* fPairAn;  //TODO can I delete this after fExpXiHist is built?
  TH1D* fExpXiHist;
  double fLambdaFactor;
  TH2D* fTransformMatrix;
  td1dVec fKStarBinCenters;
  td1dVec fResCf;
  td1dVec fTransformedResCf;

  bool fUseCoulombOnlyInterpCfs;
  TH2D* f2dCoulombOnlyInterpCfs;
  double fCurrentRadiusParam;
  double fRadiusFactor;  //Allow me to give SigSt different radii


#ifdef __ROOT__
  ClassDef(SimpleChargedResidualCf, 1)
#endif
};

inline AnalysisType SimpleChargedResidualCf::GetResidualType() {return fResidualType;}
inline double SimpleChargedResidualCf::GetLambdaFactor() {return fLambdaFactor;}
inline TH2D* SimpleChargedResidualCf::GetTransformMatrix() {return fTransformMatrix;}

inline ParticlePDGType SimpleChargedResidualCf::GetMotherType1() {return fMotherType1;}
inline ParticlePDGType SimpleChargedResidualCf::GetDaughterType1() {return fDaughterType1;}
inline ParticlePDGType SimpleChargedResidualCf::GetMotherType2() {return fMotherType2;}
inline ParticlePDGType SimpleChargedResidualCf::GetDaughterType2() {return fDaughterType2;}

inline void SimpleChargedResidualCf::SetRadiusFactor(double aFactor) {fRadiusFactor = aFactor;}
#endif
