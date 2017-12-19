///////////////////////////////////////////////////////////////////////////
// FitSystematicAnalysis.h:                                              //
///////////////////////////////////////////////////////////////////////////

#ifndef FITSYSTEMATICANALYSIS_H
#define FITSYSTEMATICANALYSIS_H

#include "TSystem.h"

#include "FitGenerator.h"
#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

#include "CanvasPartition.h"
class CanvasPartition;

#include "Types_FitParamValues.h"

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
  void SetConjAnalysisType();
  TString GetCutValues(int aIndex);
  void OutputCutValues(int aIndex, ostream &aOut=std::cout);

  double ExtractParamValue(TString aString);
  void AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber);
  void PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut=std::cout);

  static void AppendFitInfo(TString &aSaveName, bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, IncludeResidualsType aIncludeResidualsType, 
                            ResPrimMaxDecayType aResPrimMaxDecayType=k5fm, ChargedResidualsType aChargedResidualsType=kUseXiDataAndCoulombOnlyInterp, bool aFixD0=false);
  void AppendFitInfo(TString &aSaveName);
  FitGenerator* BuildFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, NonFlatBgdFitType aNonFlatBgdFitType);

  void RunAllFits(bool aSaveImages=false, bool aWriteToTxtFile=false);
  void RunVaryFitRange(bool aSaveImages=false, bool aWriteToTxtFile=false, double aMaxKStar1=0.225, double aMaxKStar2=0.300, double aMaxKStar3=0.375);
  void RunVaryNonFlatBackgroundFit(bool aSaveImages=false, bool aWriteToTxtFile=false);

  //inline
  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetApplyMomResCorrection(bool aApply);

  void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);

  void SetFixD0(bool aFix);

  void SetSaveDirectory(TString aDirectory);

protected:
  TString fFileLocationBase;
  TString fFileLocationBaseMC;
  AnalysisType fAnalysisType;
  AnalysisType fConjAnalysisType;
  CentralityType fCentralityType;
  FitGeneratorType fFitGeneratorType;
  bool fShareLambdaParams;
  bool fAllShareSingleLambdaParam;
  bool fApplyNonFlatBackgroundCorrection;
  NonFlatBgdFitType fNonFlatBgdFitType;
  bool fApplyMomResCorrection;

  IncludeResidualsType fIncludeResidualsType;
  ChargedResidualsType fChargedResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;

  bool fFixD0;

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
inline void FitSystematicAnalysis::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fNonFlatBgdFitType = aNonFlatBgdFitType;}
inline void FitSystematicAnalysis::SetApplyMomResCorrection(bool aApply) {fApplyMomResCorrection=aApply;}

inline void FitSystematicAnalysis::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType) {fIncludeResidualsType = aIncludeResidualsType;}
inline void FitSystematicAnalysis::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}
inline void FitSystematicAnalysis::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fResPrimMaxDecayType = aResPrimMaxDecayType;}

inline void FitSystematicAnalysis::SetFixD0(bool aFix) {fFixD0=aFix;}

inline void FitSystematicAnalysis::SetSaveDirectory(TString aDirectory) {fSaveDirectory = aDirectory;}

#endif
