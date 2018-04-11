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
  DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");

  DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType=kMB, AnalysisRunType aRunType=kTrain, int aNPartialAnalysis=2, FitGeneratorType aGeneratorType=kPairwConj, bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, TString aDirNameModifier="");

  DualieFitGenerator(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2);

  virtual ~DualieFitGenerator();

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

inline void DualieFitGenerator::SetFitType(FitType aFitType) {fFitGen1->SetFitType(aFitType); fFitGen2->SetFitType(aFitType);}
inline void DualieFitGenerator::SetApplyNonFlatBackgroundCorrection(bool aApply) {fFitGen1->SetApplyNonFlatBackgroundCorrection(aApply); fFitGen2->SetApplyNonFlatBackgroundCorrection(aApply);}
inline void DualieFitGenerator::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fFitGen1->SetNonFlatBgdFitType(aNonFlatBgdFitType); fFitGen2->SetNonFlatBgdFitType(aNonFlatBgdFitType);}
inline void DualieFitGenerator::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fFitGen1->SetApplyMomResCorrection(aApplyMomResCorrection); fFitGen2->SetApplyMomResCorrection(aApplyMomResCorrection);}
inline void DualieFitGenerator::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType, double aMinLambda, double aMaxLambda) {fFitGen1->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda); fFitGen2->SetIncludeResidualCorrelationsType(aIncludeResidualsType, aMinLambda, aMaxLambda);}
inline void DualieFitGenerator::SetAllLambdaParamLimits(double aMin, double aMax) {fFitGen1->SetAllLambdaParamLimits(aMin, aMax); fFitGen2->SetAllLambdaParamLimits(aMin, aMax);}
inline void DualieFitGenerator::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fFitGen1->SetChargedResidualsType(aChargedResidualsType); fFitGen2->SetChargedResidualsType(aChargedResidualsType);}
inline void DualieFitGenerator::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fFitGen1->SetResPrimMaxDecayType(aResPrimMaxDecayType); fFitGen2->SetResPrimMaxDecayType(aResPrimMaxDecayType);}
inline void DualieFitGenerator::SetLambdaParamStartValue(double aLam, bool aConjPair, CentralityType aCentType, bool aIsFixed) {fFitGen1->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed); fFitGen2->SetLambdaParamStartValue(aLam, aConjPair, aCentType, aIsFixed);}

#endif


