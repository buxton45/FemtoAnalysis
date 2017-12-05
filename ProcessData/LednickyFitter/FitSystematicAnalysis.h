///////////////////////////////////////////////////////////////////////////
// FitSystematicAnalysis.h:                                              //
///////////////////////////////////////////////////////////////////////////

#ifndef FITSYSTEMATICANALYSIS_H
#define FITSYSTEMATICANALYSIS_H

#include "FitGenerator.h"
#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

#include "CanvasPartition.h"
class CanvasPartition;


class FitSystematicAnalysis {

public:
  FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false);

  FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false);

  FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false);

  virtual ~FitSystematicAnalysis();
  TString GetCutValues(int aIndex);
  void OutputCutValues(int aIndex, ostream &aOut=std::cout);

  double ExtractParamValue(TString aString);
  void AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber);
  void PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut=std::cout);
  void RunAllFits(bool aSave=false, ostream &aOut=std::cout);

  void RunVaryFitRange(bool aSave=false, ostream &aOut=std::cout, double aMaxKStar1=0.225, double aMaxKStar2=0.300, double aMaxKStar3=0.375);
  void RunVaryNonFlatBackgroundFit(bool aSave=false, ostream &aOut=std::cout);

  //inline
  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetApplyMomResCorrection(bool aApply);

  void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);

  void SetSaveDirectory(TString aDirectory);

protected:
  TString fFileLocationBase;
  TString fFileLocationBaseMC;
  AnalysisType fAnalysisType;
  CentralityType fCentralityType;
  FitGeneratorType fFitGeneratorType;
  bool fShareLambdaParams;
  bool fAllShareSingleLambdaParam;
  bool fApplyNonFlatBackgroundCorrection;
  bool fApplyMomResCorrection;
  IncludeResidualsType fIncludeResidualsType;
  ChargedResidualsType fChargedResidualsType;

  TString fSaveDirectory;

  TString fDirNameModifierBase1;
  TString fDirNameModifierBase2;

  vector<double> fModifierValues1;
  vector<double> fModifierValues2;

#ifdef __ROOT__
  ClassDef(FitSystematicAnalysis, 1)
#endif
};

inline void FitSystematicAnalysis::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection=aApply;}
inline void FitSystematicAnalysis::SetApplyMomResCorrection(bool aApply) {fApplyMomResCorrection=aApply;}

inline void FitSystematicAnalysis::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType) {fIncludeResidualsType = aIncludeResidualsType;}
inline void FitSystematicAnalysis::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}

inline void FitSystematicAnalysis::SetSaveDirectory(TString aDirectory) {fSaveDirectory = aDirectory;}

#endif
