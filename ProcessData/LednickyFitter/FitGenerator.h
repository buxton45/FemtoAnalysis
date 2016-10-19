///////////////////////////////////////////////////////////////////////////
// FitGenerator:                                                         //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef FITGENERATOR_H
#define FITGENERATOR_H


#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

#include "CanvasPartition.h"
class CanvasPartition;


class FitGenerator {

public:
  FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, int aNPartialAnalysis=5, bool aIsTrainResults=false, CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj);
  virtual ~FitGenerator();

  void SetNAnalyses();
  void SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);
  void SetupAxis(TAxis* aAxis, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);

  void DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  void DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin=0.9, double aYmax=1.1, double aXmin=0.0, double aXmax=0.5, int aMarkerColor=1, TString aOption = "", int aMarkerStyle=20);
  virtual TCanvas* DrawKStarCfs();
  virtual TCanvas* DrawKStarCfswFits();


  void SetDefaultSharedParameters();
  void DoFit();


  //inline 
  void SetSharedParameter(ParameterType aParamType);  //share amongst all
  void SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound=0., double aUpperBound=0.);  //share amongst all

  void SetSharedParameter(ParameterType aParamType, vector<int> &aSharedAnalyses); //share amongst analyses selected in aSharedAnalyses
  void SetSharedParameter(ParameterType aParamType, vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound=0., double aUpperBound=0.);

protected:
  bool fContainsMC;
  int fNAnalyses;  //should be 1, 2, 3 or 6
  FitGeneratorType fGeneratorType;
  AnalysisType fPairType, fConjPairType;
  CentralityType fCentralityType;  //Note kMB means include all


  FitSharedAnalyses* fSharedAn;
  LednickyFitter* fLednickyFitter;



#ifdef __ROOT__
  ClassDef(FitGenerator, 1)
#endif
};


inline void FitGenerator::SetSharedParameter(ParameterType aParamType) 
  {fSharedAn->SetSharedParameter(aParamType);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, double aStartValue, double aLowerBound, double aUpperBound) 
  {fSharedAn->SetSharedParameter(aParamType,aStartValue,aLowerBound,aUpperBound);}

inline void FitGenerator::SetSharedParameter(ParameterType aParamType, vector<int> &aSharedAnalyses) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses);}
inline void FitGenerator::SetSharedParameter(ParameterType aParamType, vector<int> &aSharedAnalyses, double aStartValue, double aLowerBound, double aUpperBound) 
  {fSharedAn->SetSharedParameter(aParamType,aSharedAnalyses,aStartValue,aLowerBound,aUpperBound);}

#endif






