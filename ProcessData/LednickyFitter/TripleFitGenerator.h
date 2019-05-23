/* TripleFitGenerator.h */
/* Purpose is to be able to combine all LamK analyses, with like centralities sharing
   radii and possibly lambda parameters
       ALWAYS fFitGen1=kLamKchP, fFitGen2=kLamKchM, and fFitGen3=LamK0
   and ALWAYS fGeneratorType=kPairwConj
*/

//TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
//For now, LamK0 will have same sharing properties as LamKch
//   instead of case when fit separate, when I typically have all LamK0 share a single lambda parameter

#ifndef TRIPLEFITGENERATOR_H
#define TRIPLEFITGENERATOR_H

#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

class TripleFitGenerator {

public:
  TripleFitGenerator(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType_LamKch=kTrain, AnalysisRunType aRunType_LamK0=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier_LamKch="", TString aDirNameModifier_LamK0="", bool aUseStavCf=false);

  TripleFitGenerator(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0, CentralityType aCentralityType=kMB, AnalysisRunType aRunType_LamKch=kTrain, AnalysisRunType aRunType_LamK0=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier_LamKch="", TString aDirNameModifier_LamK0="", bool aUseStavCf=false);

  TripleFitGenerator(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2);

  virtual ~TripleFitGenerator();

  void CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii=true);  //Sharing means only across a given centrality
  void CreateMasterSharedAn();
  void CreateMinuitParameters(bool aShareLambda, bool aShareRadii=true);

  void InitializeGenerator(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);  //Called withith DoFit
  void DoFit(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);
  void ReturnNecessaryInfoToFitGenerators();  //This isn't needed for the fitting process, just for drawing

  TCanvas* GenerateContourPlots(int aNPoints, const vector<double> &aParams, const vector<double> &aErrVals={4,1}, TString aSaveNameModifier="", bool aFixAllOthers=false, bool aShareLambda=true, bool aShareRadii=true, double aMaxFitKStar=0.3);  //1=1sigma, 4=2sigma

  void WriteToMasterFitValuesFile(TString aFileLocation_LamKch, TString aFileLocation_LamK0, TString aResultsDate);


  //------------------ DRAWING ----------------------------------
  virtual TObjArray* DrawKStarCfs(bool aSaveImage=false, bool aDrawSysErrors=true);

  virtual TObjArray* DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aZoomROP=true);

  virtual TObjArray* DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);

  //Begin 2D TObjArray*--------------------------------------
  virtual TObjArray* DrawAllResiduals(bool aSaveImage=false);

  virtual TObjArray* DrawAllResidualsWithTransformMatrices(bool aSaveImage=false, bool aDrawv2=false);

  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);

  virtual TObjArray* DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aOutputCheckCorrectedCf=false);
  //End 2D TObjArray*--------------------------------------

  virtual TObjArray* DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aZoomResiduals=false);

  virtual TObjArray* DrawModelKStarCfs(bool aSaveImage=false);

  virtual CanvasPartition* BuildKStarCfswFitsCanvasPartition_CombineConj_AllAn(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);
  virtual TCanvas* DrawKStarCfswFits_CombineConj_AllAn(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage=false, bool aDrawSysErrors=true, bool aZoomROP=true, bool aSuppressFitInfoOutput=false, bool aLabelLines=false);

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

  void SetSaveLocationBase(TString aBase_LamKch, TString aBase_LamK0, TString aSaveNameModifier="");

  void SetFitType(FitType aFitType);

  void SetKStarMinMaxNorm(double aMin, double aMax);
  void SetMinMaxBgdFit(AnalysisType aAnType, double aMin, double aMax);

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitTypes(NonFlatBgdFitType aNonFlatBgdFitType_LamKch, NonFlatBgdFitType aNonFlatBgdFitType_LamK0);
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
  FitGeneratorAndDraw* GetFitGen3();

  void SetMasterFileLocation(TString aLocation_LamKch, TString aLocation_LamK0);
  void SetSystematicsFileLocation(TString aLocation_LamKch, TString aLocation_LamK0);

protected:

  FitGeneratorAndDraw* fFitGen1;
  FitGeneratorAndDraw* fFitGen2;
  FitGeneratorAndDraw* fFitGen3;

  LednickyFitter* fMasterLednickyFitter;
  FitSharedAnalyses* fMasterSharedAn;

  int fNMinuitParams;
  vector<double> fMinuitMinParams;
  vector<double> fMinuitParErrors;

  vector<vector<FitParameter*> > fMasterMinuitFitParametersMatrix;


#ifdef __ROOT__
  ClassDef(TripleFitGenerator, 1)
#endif
};


inline void TripleFitGenerator::SetRadiusStartValue(double aRad, int aIndex) {fFitGen1->SetRadiusStartValue(aRad, aIndex); fFitGen2->SetRadiusStartValue(aRad, aIndex); fFitGen3->SetRadiusStartValue(aRad, aIndex);}
inline void TripleFitGenerator::SetRadiusStartValues(const vector<double> &aStartValues) {fFitGen1->SetRadiusStartValues(aStartValues); fFitGen2->SetRadiusStartValues(aStartValues); fFitGen3->SetRadiusStartValues(aStartValues);}
inline void TripleFitGenerator::SetRadiusLimits(double aMin, double aMax, int aIndex) {fFitGen1->SetRadiusLimits(aMin, aMax, aIndex); fFitGen2->SetRadiusLimits(aMin, aMax, aIndex); fFitGen3->SetRadiusLimits(aMin, aMax, aIndex);}
inline void TripleFitGenerator::SetRadiusLimits(const td2dVec &aMinMax2dVec) {fFitGen1->SetRadiusLimits(aMinMax2dVec); fFitGen2->SetRadiusLimits(aMinMax2dVec); fFitGen3->SetRadiusLimits(aMinMax2dVec);}
inline void TripleFitGenerator::SetAllRadiiLimits(double aMin, double aMax) {fFitGen1->SetAllRadiiLimits(aMin, aMax); fFitGen2->SetAllRadiiLimits(aMin, aMax); fFitGen3->SetAllRadiiLimits(aMin, aMax);}

inline void TripleFitGenerator::SetScattParamStartValue(double aVal, ParameterType aParamType, bool aIsFixed) {fFitGen1->SetScattParamStartValue(aVal, aParamType, aIsFixed); fFitGen2->SetScattParamStartValue(aVal, aParamType, aIsFixed); fFitGen3->SetScattParamStartValue(aVal, aParamType, aIsFixed);}
inline void TripleFitGenerator::SetScattParamStartValues(double aReF0, double aImF0, double aD0, bool aAreFixed) {fFitGen1->SetScattParamStartValues(aReF0, aImF0, aD0, aAreFixed); fFitGen2->SetScattParamStartValues(aReF0, aImF0, aD0, aAreFixed); fFitGen3->SetScattParamStartValues(aReF0, aImF0, aD0, aAreFixed);}
inline void TripleFitGenerator::SetScattParamLimits(double aMin, double aMax, ParameterType aParamType) {fFitGen1->SetScattParamLimits(aMin, aMax, aParamType); fFitGen2->SetScattParamLimits(aMin, aMax, aParamType); fFitGen3->SetScattParamLimits(aMin, aMax, aParamType);}
inline void TripleFitGenerator::SetScattParamLimits(const td2dVec &aMinMax2dVec) {fFitGen1->SetScattParamLimits(aMinMax2dVec); fFitGen2->SetScattParamLimits(aMinMax2dVec); fFitGen3->SetScattParamLimits(aMinMax2dVec);}


inline void TripleFitGenerator::SetLambdaParamStartValue(double aLam, bool aConjPair, CentralityType aCentType, bool aIsFixed) {fFitGen1->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed); fFitGen2->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed); fFitGen3->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed);}
inline void TripleFitGenerator::SetAllLambdaParamStartValues(const vector<double> &aLams, bool aAreFixed) {fFitGen1->SetAllLambdaParamStartValues(aLams, aAreFixed); fFitGen2->SetAllLambdaParamStartValues(aLams, aAreFixed); fFitGen3->SetAllLambdaParamStartValues(aLams, aAreFixed);}
inline void TripleFitGenerator::SetLambdaParamLimits(double aMin, double aMax, bool aConjPair, CentralityType aCentType) {fFitGen1->SetLambdaParamLimits(aMin, aMax, aConjPair, aCentType); fFitGen2->SetLambdaParamLimits(aMin, aMax, aConjPair, aCentType); fFitGen3->SetLambdaParamLimits(aMin, aMax, aConjPair, aCentType);}
inline void TripleFitGenerator::SetAllLambdaParamLimits(double aMin, double aMax) {fFitGen1->SetAllLambdaParamLimits(aMin, aMax); fFitGen2->SetAllLambdaParamLimits(aMin, aMax); fFitGen3->SetAllLambdaParamLimits(aMin, aMax);}

inline void TripleFitGenerator::SetSaveLocationBase(TString aBase_LamKch, TString aBase_LamK0, TString aSaveNameModifier) {fFitGen1->SetSaveLocationBase(aBase_LamKch, aSaveNameModifier); fFitGen2->SetSaveLocationBase(aBase_LamKch, aSaveNameModifier); fFitGen3->SetSaveLocationBase(aBase_LamK0, aSaveNameModifier);}

inline void TripleFitGenerator::SetFitType(FitType aFitType) {fFitGen1->SetFitType(aFitType); fFitGen2->SetFitType(aFitType); fFitGen3->SetFitType(aFitType);}

inline void TripleFitGenerator::SetKStarMinMaxNorm(double aMin, double aMax) {fFitGen1->SetKStarMinMaxNorm(aMin, aMax); fFitGen2->SetKStarMinMaxNorm(aMin, aMax); fFitGen3->SetKStarMinMaxNorm(aMin, aMax);}
inline void TripleFitGenerator::SetMinMaxBgdFit(AnalysisType aAnType, double aMin, double aMax) 
{
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP){fFitGen1->SetMinMaxBgdFit(aMin, aMax); fFitGen2->SetMinMaxBgdFit(aMin, aMax);}
  else if(aAnType==kLamK0 || aAnType==kALamK0) fFitGen3->SetMinMaxBgdFit(aMin, aMax);
  else assert(0);
}

inline void TripleFitGenerator::SetApplyNonFlatBackgroundCorrection(bool aApply) {fFitGen1->SetApplyNonFlatBackgroundCorrection(aApply); fFitGen2->SetApplyNonFlatBackgroundCorrection(aApply); fFitGen3->SetApplyNonFlatBackgroundCorrection(aApply);}
inline void TripleFitGenerator::SetNonFlatBgdFitTypes(NonFlatBgdFitType aNonFlatBgdFitType_LamKch, NonFlatBgdFitType aNonFlatBgdFitType_LamK0) {fFitGen1->SetNonFlatBgdFitType(aNonFlatBgdFitType_LamKch); fFitGen2->SetNonFlatBgdFitType(aNonFlatBgdFitType_LamKch); fFitGen3->SetNonFlatBgdFitType(aNonFlatBgdFitType_LamK0);}
inline void TripleFitGenerator::SetUseNewBgdTreatment(bool aUse) {fFitGen1->SetUseNewBgdTreatment(aUse); fFitGen2->SetUseNewBgdTreatment(aUse); fFitGen3->SetUseNewBgdTreatment(aUse);}
inline void TripleFitGenerator::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fFitGen1->SetApplyMomResCorrection(aApplyMomResCorrection); fFitGen2->SetApplyMomResCorrection(aApplyMomResCorrection); fFitGen3->SetApplyMomResCorrection(aApplyMomResCorrection);}
inline void TripleFitGenerator::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda, double aMaxLambda) {fFitGen1->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda); fFitGen2->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda); fFitGen3->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda);}

inline void TripleFitGenerator::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fFitGen1->SetChargedResidualsType(aChargedResidualsType); fFitGen2->SetChargedResidualsType(aChargedResidualsType); fFitGen3->SetChargedResidualsType(aChargedResidualsType);}
inline void TripleFitGenerator::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fFitGen1->SetResPrimMaxDecayType(aResPrimMaxDecayType); fFitGen2->SetResPrimMaxDecayType(aResPrimMaxDecayType); fFitGen3->SetResPrimMaxDecayType(aResPrimMaxDecayType);}
inline void TripleFitGenerator::SetUsemTScalingOfResidualRadii(bool aUse, double aPower) {fFitGen1->SetUsemTScalingOfResidualRadii(aUse, aPower); fFitGen2->SetUsemTScalingOfResidualRadii(aUse, aPower); fFitGen3->SetUsemTScalingOfResidualRadii(aUse, aPower);}

inline void TripleFitGenerator::SetSaveFileType(TString aType) {fFitGen1->SetSaveFileType(aType); fFitGen2->SetSaveFileType(aType); fFitGen3->SetSaveFileType(aType);}

inline void TripleFitGenerator::SetFixNormParams(bool aFixNormParams) {fFitGen1->SetFixNormParams(aFixNormParams); fFitGen2->SetFixNormParams(aFixNormParams); fFitGen3->SetFixNormParams(aFixNormParams);}


inline FitGeneratorAndDraw* TripleFitGenerator::GetFitGen1() {return fFitGen1;}
inline FitGeneratorAndDraw* TripleFitGenerator::GetFitGen2() {return fFitGen2;}
inline FitGeneratorAndDraw* TripleFitGenerator::GetFitGen3() {return fFitGen3;}

inline void TripleFitGenerator::SetMasterFileLocation(TString aLocation_LamKch, TString aLocation_LamK0) {fFitGen1->SetMasterFileLocation(aLocation_LamKch); fFitGen2->SetMasterFileLocation(aLocation_LamKch); fFitGen3->SetMasterFileLocation(aLocation_LamK0);}
inline void TripleFitGenerator::SetSystematicsFileLocation(TString aLocation_LamKch, TString aLocation_LamK0) {fFitGen1->SetSystematicsFileLocation(aLocation_LamKch); fFitGen2->SetSystematicsFileLocation(aLocation_LamKch); fFitGen3->SetSystematicsFileLocation(aLocation_LamK0);}
#endif


