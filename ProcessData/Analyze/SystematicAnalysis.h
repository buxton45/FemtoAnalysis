///////////////////////////////////////////////////////////////////////////
// SystematicAnalysis:                                                   //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef SYSTEMATICANALYSIS_H
#define SYSTEMATICANALYSIS_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Analysis.h"
class Analysis;


class SystematicAnalysis {

public:
  enum DiffHistFitType {kExpDecay=0, kDampedHarmOsc=1};

  SystematicAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   TString aDirNameModifierBase2, vector<double> &aModifierValues2);

  SystematicAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1);

  virtual ~SystematicAnalysis();

  int Factorial(int aInput);
  int nChoosek(int aN, int aK);

  TH1* GetDiffHist(TH1* aHist1, TH1* aHist2);
  double GetPValueCorrelated(TH1* aHist1, TH1* aHist2);
  void GetAllPValues(ostream &aOut=std::cout);

  TF1* FitDiffHist(TH1* aDiffHist, DiffHistFitType aFitType=kExpDecay);
  void GetAllFits(ostream &aOut=std::cout, double aNSigma=2.);

  void DrawAll();
  void DrawAllDiffs(bool aDrawFits=false, bool aSaveImages=false);


  void SetSaveDirectory(TString aDirectory);
protected:
  TString fSaveDirectory;
  TString fFileLocationBase;
  AnalysisType fAnalysisType;
  CentralityType fCentralityType;

  TString fDirNameModifierBase1;
  TString fDirNameModifierBase2;

  vector<double> fModifierValues1;
  vector<double> fModifierValues2;

  vector<Analysis> fAnalyses;





#ifdef __ROOT__
  ClassDef(SystematicAnalysis, 1)
#endif
};

inline void SystematicAnalysis::SetSaveDirectory(TString aDirectory) {fSaveDirectory = aDirectory;}


#endif
