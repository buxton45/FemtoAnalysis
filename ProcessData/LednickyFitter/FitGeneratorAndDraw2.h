/* FitGeneratorAndDraw2.h */
/* Purpose is to be able to combine all cLamcKch analyses, with like centralities sharing
   radii and possibly lambda parameters
       ALWAYS fFitGen1=kLamKchP and fFitGen2=kLamKchM
   and ALWAYS fGeneratorType=kPairwConj
*/

#ifndef FITGENERATORANDDRAW2_H
#define FITGENERATORANDDRAW2_H

#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

class FitGeneratorAndDraw2 {

public:
  FitGeneratorAndDraw2(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");

  FitGeneratorAndDraw2(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");

  FitGeneratorAndDraw2(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2);

  virtual ~FitGeneratorAndDraw2();

  void CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii=true);  //Sharing means only across a given centrality
  void CreateMasterSharedAn();
  void CreateMinuitParameters(bool aShareLambda, bool aShareRadii=true);

  void InitializeGenerator(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);  //Called withith DoFit
  void DoFit(bool aShareLambda, bool aShareRadii=true, double aMaxFitKStar=0.3);

/*
  void WriteAllFitParameters(ostream &aOut);
  vector<TString> GetAllFitParametersTStringVector();
  void FindGoodInitialValues(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection);
*/

//-------inline
  void SetFitType(FitType aFitType);
  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda=0., double aMaxLambda=1.0);
  void SetAllLambdaParamLimits(double aMin, double aMax);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);
  void SetLambdaParamStartValue(double aLam, bool aConjPair=false, CentralityType aCentType=kMB, bool aIsFixed=false);

protected:

  FitGeneratorAndDraw* fFitGen1;
  FitGeneratorAndDraw* fFitGen2;

  TMinuit* fMasterMinuit;
  LednickyFitter* fMasterLednickyFitter;
  FitSharedAnalyses* fMasterSharedAn;

  int fNMinuitParams;
  vector<double> fMinuitMinParams;
  vector<double> fMinuitParErrors;

  vector<vector<FitParameter*> > fMasterMinuitFitParametersMatrix;


#ifdef __ROOT__
  ClassDef(FitGeneratorAndDraw2, 1)
#endif
};

inline void FitGeneratorAndDraw2::SetFitType(FitType aFitType) {fFitGen1->SetFitType(aFitType); fFitGen2->SetFitType(aFitType);}
inline void FitGeneratorAndDraw2::SetApplyNonFlatBackgroundCorrection(bool aApply) {fFitGen1->SetApplyNonFlatBackgroundCorrection(aApply); fFitGen2->SetApplyNonFlatBackgroundCorrection(aApply);}
inline void FitGeneratorAndDraw2::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fFitGen1->SetNonFlatBgdFitType(aNonFlatBgdFitType); fFitGen2->SetNonFlatBgdFitType(aNonFlatBgdFitType);}
inline void FitGeneratorAndDraw2::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fFitGen1->SetApplyMomResCorrection(aApplyMomResCorrection); fFitGen2->SetApplyMomResCorrection(aApplyMomResCorrection);}
inline void FitGeneratorAndDraw2::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda, double aMaxLambda) {fFitGen1->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda); fFitGen2->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda);}
inline void FitGeneratorAndDraw2::SetAllLambdaParamLimits(double aMin, double aMax) {fFitGen1->SetAllLambdaParamLimits(aMin, aMax); fFitGen2->SetAllLambdaParamLimits(aMin, aMax);}
inline void FitGeneratorAndDraw2::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fFitGen1->SetChargedResidualsType(aChargedResidualsType); fFitGen2->SetChargedResidualsType(aChargedResidualsType);}
inline void FitGeneratorAndDraw2::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fFitGen1->SetResPrimMaxDecayType(aResPrimMaxDecayType); fFitGen2->SetResPrimMaxDecayType(aResPrimMaxDecayType);}
inline void FitGeneratorAndDraw2::SetLambdaParamStartValue(double aLam, bool aConjPair, CentralityType aCentType, bool aIsFixed) {fFitGen1->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed); fFitGen2->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed);}

#endif


