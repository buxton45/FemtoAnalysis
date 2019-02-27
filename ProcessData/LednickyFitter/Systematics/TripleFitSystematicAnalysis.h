///////////////////////////////////////////////////////////////////////////
// TripleFitSystematicAnalysis.h:                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef TRIPLEFITSYSTEMATICANALYSIS_H
#define TRIPLEFITSYSTEMATICANALYSIS_H

#include "TSystem.h"

#include "TripleFitGenerator.h"
#include "FitValuesWriter.h"

#include "CanvasPartition.h"
class CanvasPartition;

#include "Types_FitParamValues.h"

class TripleFitSystematicAnalysis {

public:
  TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                              TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                              TString aGeneralAnTypeModified,
                              TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                              TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                              CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                              bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);

  TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                              TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                              TString aGeneralAnTypeModified,
                              TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                              CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                              bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);

  TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                              TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                              TString aGeneralAnTypeModified,
                              CentralityType aCentralityType=kMB, FitGeneratorType aGeneratorType=kPairwConj, 
                              bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aDualieShareLambda=true, bool aDualieShareRadii=true);


  virtual ~TripleFitSystematicAnalysis();
  TString GetCutValues(int aIndex);
  void OutputCutValues(int aIndex, ostream &aOut=std::cout);

  double ExtractParamValue(TString aString);
  void AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber);
  void PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut=std::cout, int aNCuts=3);

  void AppendFitInfo(TString &aSaveName);
  void SetRadiusStartValues(TripleFitGenerator* aFitGen);
  void SetLambdaStartValues(TripleFitGenerator* aFitGen);
  void SetScattParamStartValues(TripleFitGenerator* aFitGen);
  TripleFitGenerator* BuildTripleFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes);

  void RunAllFits(bool aSaveImages=false, bool aWriteToTxtFile=false);
  void RunVaryFitRange(bool aSaveImages=false, bool aWriteToTxtFile=false, double aMaxKStar1=0.225, double aMaxKStar2=0.300, double aMaxKStar3=0.375);
  void RunVaryNonFlatBackgroundFit(bool aSaveImages=false, bool aWriteToTxtFile=false);

  //inline
  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitTypes(NonFlatBgdFitType aNonFlatBgdFitType_LamKch, NonFlatBgdFitType aNonFlatBgdFitType_LamK0);
  void SetApplyMomResCorrection(bool aApply);

  void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);

  void SetFixD0(bool aFix);

  void SetSaveDirectory(TString aDirectory);

protected:
  TString fFileLocationBase_LamKch;
  TString fFileLocationBaseMC_LamKch;
  TString fFileLocationBase_LamK0;
  TString fFileLocationBaseMC_LamK0;

  AnalysisType fAnalysisType;
  AnalysisType fConjAnalysisType;

  CentralityType fCentralityType;
  FitGeneratorType fFitGeneratorType;

  bool fShareLambdaParams;
  bool fAllShareSingleLambdaParam;
  bool fDualieShareLambda;
  bool fDualieShareRadii;

  bool fApplyNonFlatBackgroundCorrection;
  NonFlatBgdFitType fNonFlatBgdFitType_LamKch;
  NonFlatBgdFitType fNonFlatBgdFitType_LamK0;
  vector<NonFlatBgdFitType> fNonFlatBgdFitTypes;
  bool fApplyMomResCorrection;

  IncludeResidualsType fIncludeResidualsType;
  ChargedResidualsType fChargedResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;

  bool fFixD0;

  TString fSaveDirectory;

  TString fGeneralAnTypeModified;
  TString fDirNameModifierBase1;
  TString fDirNameModifierBase2;

  vector<double> fModifierValues1;
  vector<double> fModifierValues2;

#ifdef __ROOT__
  ClassDef(TripleFitSystematicAnalysis, 1)
#endif
};

inline void TripleFitSystematicAnalysis::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection=aApply;}
inline void TripleFitSystematicAnalysis::SetNonFlatBgdFitTypes(NonFlatBgdFitType aNonFlatBgdFitType_LamKch, NonFlatBgdFitType aNonFlatBgdFitType_LamK0) 
{
  fNonFlatBgdFitType_LamKch = aNonFlatBgdFitType_LamKch; 
  fNonFlatBgdFitType_LamK0 = aNonFlatBgdFitType_LamK0;
  fNonFlatBgdFitTypes = vector<NonFlatBgdFitType>{fNonFlatBgdFitType_LamK0, fNonFlatBgdFitType_LamK0, 
                                                  fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch};
}
inline void TripleFitSystematicAnalysis::SetApplyMomResCorrection(bool aApply) {fApplyMomResCorrection=aApply;}

inline void TripleFitSystematicAnalysis::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType) {fIncludeResidualsType = aIncludeResidualsType;}
inline void TripleFitSystematicAnalysis::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}
inline void TripleFitSystematicAnalysis::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fResPrimMaxDecayType = aResPrimMaxDecayType;}

inline void TripleFitSystematicAnalysis::SetFixD0(bool aFix) {fFixD0=aFix;}

inline void TripleFitSystematicAnalysis::SetSaveDirectory(TString aDirectory) {fSaveDirectory = aDirectory;}

#endif
