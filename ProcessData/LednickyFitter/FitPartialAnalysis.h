///////////////////////////////////////////////////////////////////////////
// FitPartialAnalysis:                                                   //
//     This object represents a particular pair system (LamKchP),        //
//     in a particular centrality bin (i.e. 0010),                       //
//     and a particular division of the dataset (i.e. Bp1)               //
//                                                                       //
//     ex. FitPartialAnalysis = LamKchP0010_Bp1                          //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef FITPARTIALANALYSIS_H
#define FITPARTIALANALYSIS_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Faddeeva.hh"

#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TDirectoryFile.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitParameter.h"
class FitParameter;

#include "CfLite.h"
class CfLite;

#include "Types.h"

class FitPartialAnalysis {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitPartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType=kTrain, TString aDirNameModifier="");
  FitPartialAnalysis(TString aFileLocation, TString aFileLocationMC, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType=kTrain, TString aDirNameModifier="");
  virtual ~FitPartialAnalysis();

  static double NonFlatBackgroundFitFunctionLinear(double *x, double *par);
  static double NonFlatBackgroundFitFunctionQuadratic(double *x, double *par);
  static double NonFlatBackgroundFitFunctionGaussian(double *x, double *par);

  TObjArray* ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName);

  void SetParticleTypes();

  TH1* Get1dHisto(TString aHistoName, TString aNewName);
  TH1* Get1dHisto(TString aFileLocation, TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aHistoName, TString aNewName);
  TH2* Get2dHisto(TString aFileLocation, TString aDirectoryName, TString aHistoName, TString aNewName);

  void BuildKStarCf(double aMinNorm=0.32, double aMaxNorm=0.4);
  void RebinKStarCf(int aRebinFactor, double aMinNorm=0.32, double aMaxNorm=0.4);
  static TF1* FitNonFlatBackground(TH1* aCf, NonFlatBgdFitType aFitType=kLinear, double aMinFit=0.6, double aMaxFit=0.9);
  TF1* GetNonFlatBackground(NonFlatBgdFitType aFitType=kLinear, double aMinFit=0.60, double aMaxFit=0.90);

  void SetFitParameter(FitParameter* aParam);

  CfLite* GetModelKStarCf(double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=1);
  CfLite* GetModelKStarCfFake(double aMinNorm, double aMaxNorm, int aRebin=1);
  CfLite* GetModelKStarCfFakeIdeal(double aMinNorm, double aMaxNorm, int aRebin=1);

  //inline (i.e. simple) functions
  AnalysisType GetAnalysisType();
  BFieldType GetBFieldType();
  CentralityType GetCentralityType();

  void SetFitPartialAnalysisNumber(int aAnalysisNumber);  //will be set by the FitPairAnalysis object
  int GetFitPartialAnalysisNumber();

  CfLite* GetKStarCfLite();
  TH1* GetKStarCf();
  TH1* GetNumKStarCf();
  TH1* GetDenKStarCf();
  double GetKStarNumScale();
  double GetKStarDenScale();

  vector<ParticleType> GetParticleTypes();

  int GetNFitParams();
  vector<FitParameter*> GetFitParameters();
  FitParameter* GetFitNormParameter();
  FitParameter* GetFitParameter(ParameterType aParamType);

  void SetRejectOmega(bool aRejectOmega);
  bool RejectOmega();

  TH2* GetModelKStarTrueVsRecMixed();

private:
  AnalysisRunType fAnalysisRunType;
  TString fFileLocation;
  TString fFileLocationMC;
  TString fAnalysisName;
  TString fDirectoryName;

  AnalysisType fAnalysisType;
  BFieldType fBFieldType;
  CentralityType fCentralityType;

  int fFitPartialAnalysisNumber;  //to help identify in FitPairAnalysis (set in FitPairAnalysis);

  vector<ParticleType> fParticleTypes;

  CfLite *fKStarCfLite;
  TH1 *fKStarCf, *fKStarCfNum, *fKStarCfDen;
  double fKStarMinNorm, fKStarMaxNorm;
  double fKStarNumScale, fKStarDenScale;

  int fNFitParams;
  FitParameter* fLambda;
  FitParameter* fRadius;
  FitParameter* fRef0;
  FitParameter* fImf0;
  FitParameter* fd0;
  FitParameter* fRef02;
  FitParameter* fImf02;
  FitParameter* fd02;
  vector<FitParameter*> fFitParameters;
  FitParameter* fNorm;

  bool fRejectOmega;

  TH2* fModelKStarTrueVsRecMixed;
  CfLite* fModelKStarCfFake;
  CfLite* fModelKStarCfFakeIdeal;

  TF1* fNonFlatBackground;

#ifdef __ROOT__
  ClassDef(FitPartialAnalysis, 1)
#endif
};


//inline stuff
inline AnalysisType FitPartialAnalysis::GetAnalysisType() {return fAnalysisType;}
inline BFieldType FitPartialAnalysis::GetBFieldType() {return fBFieldType;}
inline CentralityType FitPartialAnalysis::GetCentralityType() {return fCentralityType;}

inline void FitPartialAnalysis::SetFitPartialAnalysisNumber(int aAnalysisNumber) {fFitPartialAnalysisNumber = aAnalysisNumber;}
inline int FitPartialAnalysis::GetFitPartialAnalysisNumber() {return fFitPartialAnalysisNumber;}

inline CfLite* FitPartialAnalysis::GetKStarCfLite() {return fKStarCfLite;}
inline TH1* FitPartialAnalysis::GetKStarCf() {return fKStarCf;}
inline TH1* FitPartialAnalysis::GetNumKStarCf() {return fKStarCfNum;}
inline TH1* FitPartialAnalysis::GetDenKStarCf() {return fKStarCfDen;}
inline double FitPartialAnalysis::GetKStarNumScale() {return fKStarNumScale;}
inline double FitPartialAnalysis::GetKStarDenScale() {return fKStarDenScale;}

inline vector<ParticleType> FitPartialAnalysis::GetParticleTypes() {return fParticleTypes;}

inline int FitPartialAnalysis::GetNFitParams() {return fNFitParams;}
inline vector<FitParameter*> FitPartialAnalysis::GetFitParameters() {return fFitParameters;}
inline FitParameter* FitPartialAnalysis::GetFitNormParameter() {return fNorm;}
inline FitParameter* FitPartialAnalysis::GetFitParameter(ParameterType aParamType) {return fFitParameters[aParamType];}

inline void FitPartialAnalysis::SetRejectOmega(bool aRejectOmega) {fRejectOmega = aRejectOmega;}
inline bool FitPartialAnalysis::RejectOmega() {return fRejectOmega;}

inline TH2* FitPartialAnalysis::GetModelKStarTrueVsRecMixed() {return fModelKStarTrueVsRecMixed;}

#endif
