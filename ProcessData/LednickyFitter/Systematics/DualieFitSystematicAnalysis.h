///////////////////////////////////////////////////////////////////////////
// DualieFitSystematicAnalysis.h:                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef DUALIEFITSYSTEMATICANALYSIS_H
#define DUALIEFITSYSTEMATICANALYSIS_H

#include "TSystem.h"

#include "DualieFitGenerator.h"
#include "FitValuesWriter.h"

#include "CanvasPartition.h"
class CanvasPartition;

#include "Types_FitParamValues.h"

class DualieFitSystematicAnalysis {

public:
  DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                   bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);

  DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                   bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);

  DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                   CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                   bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);

  virtual ~DualieFitSystematicAnalysis();
  void SetConjAnalysisType();
  TString GetCutValues(int aIndex);
  void OutputCutValues(int aIndex, ostream &aOut=std::cout);

  double ExtractParamValue(TString aString);
  void AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber);
  void PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut=std::cout);

  void AppendFitInfo(TString &aSaveName);
  void SetRadiusStartValues(DualieFitGenerator* aFitGen);
  void SetLambdaStartValues(DualieFitGenerator* aFitGen);
  void SetScattParamStartValues(DualieFitGenerator* aFitGen);
  DualieFitGenerator* BuildDualieFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, NonFlatBgdFitType aNonFlatBgdFitType);

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
  bool fDualieShareLambda;
  bool fDualieShareRadii;
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
  ClassDef(DualieFitSystematicAnalysis, 1)
#endif
};

inline void DualieFitSystematicAnalysis::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection=aApply;}
inline void DualieFitSystematicAnalysis::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fNonFlatBgdFitType = aNonFlatBgdFitType;}
inline void DualieFitSystematicAnalysis::SetApplyMomResCorrection(bool aApply) {fApplyMomResCorrection=aApply;}

inline void DualieFitSystematicAnalysis::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType) {fIncludeResidualsType = aIncludeResidualsType;}
inline void DualieFitSystematicAnalysis::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}
inline void DualieFitSystematicAnalysis::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fResPrimMaxDecayType = aResPrimMaxDecayType;}

inline void DualieFitSystematicAnalysis::SetFixD0(bool aFix) {fFixD0=aFix;}

inline void DualieFitSystematicAnalysis::SetSaveDirectory(TString aDirectory) {fSaveDirectory = aDirectory;}

#endif
