/* DualieFitGenerator.h */
/* Purpose is to be able to combine all cLamcKch analyses, with like centralities sharing
   radii and possibly lambda parameters
       ALWAYS fFitGen1=kLamKchP and fFitGen2=kLamKchM
   and ALWAYS fGeneratorType=kPairwConj
   Name DualieFitGenerator because it combines two FitGenerators into one, and because
   it's badass like a dualie truck (ha!)
*/

#ifndef DUALIEFITGENERATOR_H
#define DUALIEFITGENERATOR_H

#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

class DualieFitGenerator {

public:
  DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);

  DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="", bool aUseStavCf=false);

  DualieFitGenerator(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2);

  virtual ~DualieFitGenerator();

  void CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii=true);  //Sharing means only across a given centrality
  void CreateMasterSharedAn();
  void CreateMinuitParameters(bool aShareLambda, bool aShareRadii=true);

  void InitializeGenerator(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);  //Called withith DoFit
  void DoFit(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);
  void ReturnNecessaryInfoToFitGenerators();  //This isn't needed for the fitting process, just for drawing

  void WriteToMasterFitValuesFile(TString aFileLocation, TString aResultsDate);

/*
  void WriteAllFitParameters(ostream &aOut);
  vector<TString> GetAllFitParametersTStringVector();
  void FindGoodInitialValues(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection);
*/

  //------------------ DRAWING ----------------------------------
  virtual TObjArray* DrawKStarCfs(bool aSaveImage=false, bool aDrawSysErrors=true);

  virtual TObjArray* DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aZoomROP=true);

  virtual TObjArray* DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType=kLinear, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);

  //Begin 2D TObjArray*--------------------------------------
  virtual TObjArray* DrawAllResiduals(bool aSaveImage=false);

  virtual TObjArray* DrawAllResidualsWithTransformMatrices(bool aSaveImage=false);

  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);

  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);
  //End 2D TObjArray*--------------------------------------

  virtual TObjArray* DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aZoomResiduals=false);

  virtual TObjArray* DrawModelKStarCfs(bool aSaveImage=false);


//-------inline
  void SetRadiusStartValue(double aRad, int aIndex=0);
  void SetRadiusStartValues(const vector<double> &aStartValues);
  void SetRadiusLimits(double aMin, double aMax, int aIndex=0);  //set for particular R parameter
  void SetRadiusLimits(const td2dVec &aMinMax2dVec);  //set unique values for each R parameter
  void SetAllRadiiLimits(double aMin, double aMax);

  void SetScattParamStartValue(double aVal, ParameterType aParamType, bool aIsFixed=false);
  void SetScattParamStartValues(double aReF0, double aImF0, double aD0, bool aAreFixed=false);
  void SetScattParamLimits(double aMin, double aMax, ParameterType aParamType);
  void SetScattParamLimits(const td2dVec &aMinMax2dVec);


  void SetLambdaParamStartValue(double aLam, bool aConjPair=false, CentralityType aCentType=kMB, bool aIsFixed=false);
  void SetAllLambdaParamStartValues(const vector<double> &aLams, bool aAreFixed=false);
  void SetLambdaParamLimits(double aMin, double aMax, bool tConjPair=false, CentralityType aCentType=kMB);
  void SetAllLambdaParamLimits(double aMin, double aMax);

  void SetSaveLocationBase(TString aBase, TString aSaveNameModifier="");

  void SetFitType(FitType aFitType);

  void SetKStarMinMaxNorm(double aMin, double aMax);
  void SetMinMaxBgdFit(double aMin, double aMax);

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetUseNewBgdTreatment(bool aUse);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda=0., double aMaxLambda=1.0);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);
  void SetUsemTScalingOfResidualRadii(bool aUse=true, double aPower=-0.5);

  void SetSaveFileType(TString aType);

  void SetFixNormParams(bool aFixNormParams);

  FitGeneratorAndDraw* GetFitGen1();
  FitGeneratorAndDraw* GetFitGen2();

  void SetMasterFileLocation(TString aLocation);
  void SetSystematicsFileLocation(TString aLocation);

protected:

  FitGeneratorAndDraw* fFitGen1;
  FitGeneratorAndDraw* fFitGen2;

  LednickyFitter* fMasterLednickyFitter;
  FitSharedAnalyses* fMasterSharedAn;

  int fNMinuitParams;
  vector<double> fMinuitMinParams;
  vector<double> fMinuitParErrors;

  vector<vector<FitParameter*> > fMasterMinuitFitParametersMatrix;


#ifdef __ROOT__
  ClassDef(DualieFitGenerator, 1)
#endif
};


inline void DualieFitGenerator::SetRadiusStartValue(double aRad, int aIndex) {fFitGen1->SetRadiusStartValue(aRad, aIndex); fFitGen2->SetRadiusStartValue(aRad, aIndex);}
inline void DualieFitGenerator::SetRadiusStartValues(const vector<double> &aStartValues) {fFitGen1->SetRadiusStartValues(aStartValues); fFitGen2->SetRadiusStartValues(aStartValues);}
inline void DualieFitGenerator::SetRadiusLimits(double aMin, double aMax, int aIndex) {fFitGen1->SetRadiusLimits(aMin, aMax, aIndex); fFitGen2->SetRadiusLimits(aMin, aMax, aIndex);}
inline void DualieFitGenerator::SetRadiusLimits(const td2dVec &aMinMax2dVec) {fFitGen1->SetRadiusLimits(aMinMax2dVec); fFitGen2->SetRadiusLimits(aMinMax2dVec);}
inline void DualieFitGenerator::SetAllRadiiLimits(double aMin, double aMax) {fFitGen1->SetAllRadiiLimits(aMin, aMax); fFitGen2->SetAllRadiiLimits(aMin, aMax);}

inline void DualieFitGenerator::SetScattParamStartValue(double aVal, ParameterType aParamType, bool aIsFixed) {fFitGen1->SetScattParamStartValue(aVal, aParamType, aIsFixed); fFitGen2->SetScattParamStartValue(aVal, aParamType, aIsFixed);}
inline void DualieFitGenerator::SetScattParamStartValues(double aReF0, double aImF0, double aD0, bool aAreFixed) {fFitGen1->SetScattParamStartValues(aReF0, aImF0, aD0, aAreFixed); fFitGen2->SetScattParamStartValues(aReF0, aImF0, aD0, aAreFixed);}
inline void DualieFitGenerator::SetScattParamLimits(double aMin, double aMax, ParameterType aParamType) {fFitGen1->SetScattParamLimits(aMin, aMax, aParamType); fFitGen2->SetScattParamLimits(aMin, aMax, aParamType);}
inline void DualieFitGenerator::SetScattParamLimits(const td2dVec &aMinMax2dVec) {fFitGen1->SetScattParamLimits(aMinMax2dVec); fFitGen2->SetScattParamLimits(aMinMax2dVec);}


inline void DualieFitGenerator::SetLambdaParamStartValue(double aLam, bool aConjPair, CentralityType aCentType, bool aIsFixed) {fFitGen1->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed); fFitGen2->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed);}
inline void DualieFitGenerator::SetAllLambdaParamStartValues(const vector<double> &aLams, bool aAreFixed) {fFitGen1->SetAllLambdaParamStartValues(aLams, aAreFixed); fFitGen2->SetAllLambdaParamStartValues(aLams, aAreFixed);}
inline void DualieFitGenerator::SetLambdaParamLimits(double aMin, double aMax, bool aConjPair, CentralityType aCentType) {fFitGen1->SetLambdaParamLimits(aMin, aMax, aConjPair, aCentType); fFitGen2->SetLambdaParamLimits(aMin, aMax, aConjPair, aCentType);}
inline void DualieFitGenerator::SetAllLambdaParamLimits(double aMin, double aMax) {fFitGen1->SetAllLambdaParamLimits(aMin, aMax); fFitGen2->SetAllLambdaParamLimits(aMin, aMax);}

inline void DualieFitGenerator::SetSaveLocationBase(TString aBase, TString aSaveNameModifier) {fFitGen1->SetSaveLocationBase(aBase, aSaveNameModifier); fFitGen2->SetSaveLocationBase(aBase, aSaveNameModifier);}

inline void DualieFitGenerator::SetFitType(FitType aFitType) {fFitGen1->SetFitType(aFitType); fFitGen2->SetFitType(aFitType);}

inline void DualieFitGenerator::SetKStarMinMaxNorm(double aMin, double aMax) {fFitGen1->SetKStarMinMaxNorm(aMin, aMax); fFitGen2->SetKStarMinMaxNorm(aMin, aMax);}
inline void DualieFitGenerator::SetMinMaxBgdFit(double aMin, double aMax) {fFitGen1->SetMinMaxBgdFit(aMin, aMax); fFitGen2->SetMinMaxBgdFit(aMin, aMax);}

inline void DualieFitGenerator::SetApplyNonFlatBackgroundCorrection(bool aApply) {fFitGen1->SetApplyNonFlatBackgroundCorrection(aApply); fFitGen2->SetApplyNonFlatBackgroundCorrection(aApply);}
inline void DualieFitGenerator::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fFitGen1->SetNonFlatBgdFitType(aNonFlatBgdFitType); fFitGen2->SetNonFlatBgdFitType(aNonFlatBgdFitType);}
inline void DualieFitGenerator::SetUseNewBgdTreatment(bool aUse) {fFitGen1->SetUseNewBgdTreatment(aUse); fFitGen2->SetUseNewBgdTreatment(aUse);}
inline void DualieFitGenerator::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fFitGen1->SetApplyMomResCorrection(aApplyMomResCorrection); fFitGen2->SetApplyMomResCorrection(aApplyMomResCorrection);}
inline void DualieFitGenerator::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda, double aMaxLambda) {fFitGen1->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda); fFitGen2->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda);}

inline void DualieFitGenerator::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fFitGen1->SetChargedResidualsType(aChargedResidualsType); fFitGen2->SetChargedResidualsType(aChargedResidualsType);}
inline void DualieFitGenerator::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fFitGen1->SetResPrimMaxDecayType(aResPrimMaxDecayType); fFitGen2->SetResPrimMaxDecayType(aResPrimMaxDecayType);}
inline void DualieFitGenerator::SetUsemTScalingOfResidualRadii(bool aUse, double aPower) {fFitGen1->SetUsemTScalingOfResidualRadii(aUse, aPower); fFitGen2->SetUsemTScalingOfResidualRadii(aUse, aPower);}

inline void DualieFitGenerator::SetSaveFileType(TString aType) {fFitGen1->SetSaveFileType(aType); fFitGen2->SetSaveFileType(aType);}

inline void DualieFitGenerator::SetFixNormParams(bool aFixNormParams) {fFitGen1->SetFixNormParams(aFixNormParams); fFitGen2->SetFixNormParams(aFixNormParams);}


inline FitGeneratorAndDraw* DualieFitGenerator::GetFitGen1() {return fFitGen1;}
inline FitGeneratorAndDraw* DualieFitGenerator::GetFitGen2() {return fFitGen2;}

inline void DualieFitGenerator::SetMasterFileLocation(TString aLocation) {fFitGen1->SetMasterFileLocation(aLocation); fFitGen2->SetMasterFileLocation(aLocation);}
inline void DualieFitGenerator::SetSystematicsFileLocation(TString aLocation) {fFitGen1->SetSystematicsFileLocation(aLocation); fFitGen2->SetSystematicsFileLocation(aLocation);}
#endif


