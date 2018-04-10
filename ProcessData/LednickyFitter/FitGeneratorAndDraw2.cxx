/* FitGeneratorAndDraw2.cxx */

#include "FitGeneratorAndDraw2.h"

#ifdef __ROOT__
ClassImp(FitGeneratorAndDraw2)
#endif

//GLOBAL!!!!!!!!!!!!!!!
LednickyFitter *GlobalFitter2 = NULL;

//______________________________________________________________________________
void GlobalFCN2(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalFitter2->CalculateFitFunction(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::FitGeneratorAndDraw2(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
  fFitGen1(nullptr),
  fFitGen2(nullptr),

  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchP, aCentralityTypes, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchM, aCentralityTypes, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);

}


//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::FitGeneratorAndDraw2(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
  fFitGen1(nullptr),
  fFitGen2(nullptr),

  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchP, aCentralityType, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchM, aCentralityType, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);

}

//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::FitGeneratorAndDraw2(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2) :
  fFitGen1(aFitGen1),
  fFitGen2(aFitGen2),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{

}

//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::~FitGeneratorAndDraw2()
{
  cout << "FitGeneratorAndDraw2 object is being deleted!!!!!" << endl;
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw2::CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii)
{
  //TODO For now, not worrying about complications of "new" background treatment
  //     Anyway, I don't like this "new" method, and it will probably be abandoned
  assert(!fFitGen1->GetSharedAn()->UsingNewBgdTreatment());
  assert(!fFitGen2->GetSharedAn()->UsingNewBgdTreatment());

  fFitGen1->SetAllParameters();
  fFitGen2->SetAllParameters();

  assert(fFitGen1->GetSharedAn()->GetNFitParamsPerAnalysis() == fFitGen2->GetSharedAn()->GetNFitParamsPerAnalysis());
  int tNFitParamsPerAnalysis = fFitGen1->GetSharedAn()->GetNFitParamsPerAnalysis();

  assert(fFitGen1->GetSharedAn()->GetNFitPairAnalysis() == fFitGen2->GetSharedAn()->GetNFitPairAnalysis());
  int tNFitPairAnalysis = fFitGen1->GetSharedAn()->GetNFitPairAnalysis();

  for(int iPairAn=0; iPairAn<tNFitPairAnalysis; iPairAn++)
  {
    for(int iPar=0; iPar<tNFitParamsPerAnalysis; iPar++)
    {
      ParameterType tParamType = static_cast<ParameterType>(iPar);
      if((tParamType==kLambda && aShareLambda) || (tParamType==kRadius && aShareRadii)) 
      {
        fFitGen2->GetSharedAn()->GetFitPairAnalysis(iPairAn)->SetFitParameterShallow(fFitGen1->GetSharedAn()->GetFitPairAnalysis(iPairAn)->GetFitParameter(tParamType));
      }
    }
  }


  for(int iPar=0; iPar<tNFitParamsPerAnalysis; iPar++)
  {
    vector<FitParameter*> tTempParamVec(0);

    ParameterType tParamType = static_cast<ParameterType>(iPar);
    vector<FitParameter*> tTempParamVec1 = fFitGen1->GetSharedAn()->GetDistinctParamsOfCommonType(tParamType);
    vector<FitParameter*> tTempParamVec2 = fFitGen2->GetSharedAn()->GetDistinctParamsOfCommonType(tParamType);
    assert(tTempParamVec1.size()==tTempParamVec2.size());

    for(unsigned int i=0; i<tTempParamVec1.size(); i++) tTempParamVec.push_back(tTempParamVec1[i]);

    if(tParamType==kLambda)  //NOTE: Cannot simply combine into if(tParamType==kLambda && !aShareLambda)
    {
      if(!aShareLambda)
      {
        for(unsigned int iLam=0; iLam<tTempParamVec2.size(); iLam++) tTempParamVec.push_back(tTempParamVec2[iLam]);
      }
    }
    else if(tParamType==kRadius)  //NOTE: Cannot simply combine into if(tParamType==kRadius && !aShareRadii)
    {
      if(!aShareRadii)
      {
        for(unsigned int iR=0; iR<tTempParamVec2.size(); iR++) tTempParamVec.push_back(tTempParamVec2[iR]);
      }
    }
    else for(unsigned int i=0; i<tTempParamVec2.size(); i++) tTempParamVec.push_back(tTempParamVec2[i]);

    fMasterMinuitFitParametersMatrix.push_back(tTempParamVec);
  }
  //--------------------------------

}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw2::CreateMasterSharedAn()
{
  vector<FitPairAnalysis*> tTempFitPairAnColl(0);
  for(unsigned int i=0; i<fFitGen1->GetSharedAn()->GetFitPairAnalysisCollection().size(); i++) tTempFitPairAnColl.push_back(fFitGen1->GetSharedAn()->GetFitPairAnalysisCollection()[i]);
  for(unsigned int i=0; i<fFitGen2->GetSharedAn()->GetFitPairAnalysisCollection().size(); i++) tTempFitPairAnColl.push_back(fFitGen2->GetSharedAn()->GetFitPairAnalysisCollection()[i]);

  fMasterSharedAn = new FitSharedAnalyses(tTempFitPairAnColl);
  fMasterSharedAn->SetFitType(fFitGen1->GetSharedAn()->GetFitType());
  fMasterSharedAn->SetApplyNonFlatBackgroundCorrection(fFitGen1->GetSharedAn()->GetApplyNonFlatBackgroundCorrection());
  fMasterSharedAn->SetNonFlatBgdFitType(fFitGen1->GetSharedAn()->GetNonFlatBgdFitType());
  fMasterSharedAn->SetUseNewBgdTreatment(fFitGen1->GetSharedAn()->UsingNewBgdTreatment());
  fMasterSharedAn->SetFixNormParams(fFitGen1->GetSharedAn()->GetFixNormParams());

}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw2::CreateMinuitParameters(bool aShareLambda, bool aShareRadii)
{
  //call AFTER all parameters have been shared!!!!!

  CreateMinuitParametersMatrix(aShareLambda, aShareRadii);
  CreateMasterSharedAn();

  fNMinuitParams = 0;  //for some reason, this makes fNMinuitParams = 0 outside of this function?

  for(unsigned int iPar=0; iPar < fMasterMinuitFitParametersMatrix.size(); iPar++)
  {
    vector<FitParameter*> tempVec = fMasterMinuitFitParametersMatrix[iPar];
    for(unsigned int itemp=0; itemp < tempVec.size(); itemp++)
    {
      fMasterSharedAn->CreateMinuitParameter(tempVec[itemp]);
    }
  }

  //--Create all of the normalization parameters
  double tNormStartValue = 1.;
  for(int iAnaly=0; iAnaly<fMasterSharedAn->GetFitPairAnalysisCollection().size(); iAnaly++)
  {
    for(int iPartAn=0; iPartAn<fMasterSharedAn->GetFitPairAnalysis(iAnaly)->GetNFitPartialAnalysis(); iPartAn++)
    {
      tNormStartValue = 1.;
      fMasterSharedAn->GetFitPairAnalysis(iAnaly)->GetFitNormParameter(iPartAn)->SetStartValue(tNormStartValue);
//      if(fFixNormParams || fUseNewBgdTreatment) fMasterSharedAn->GetFitPairAnalysis(iAnaly)->GetFitNormParameter(iPartAn)->SetFixed(true);

      fMasterSharedAn->CreateMinuitParameter(fMasterSharedAn->GetFitPairAnalysis(iAnaly)->GetFitNormParameter(iPartAn));
    }
  }
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw2::InitializeGenerator(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
//  if(fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll)
//  {
//   for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++) SetRadiusLimits(1.,12.,iCent);
//  }


  CreateMinuitParameters(aShareLambda, aShareRadii);

  fMasterLednickyFitter = new LednickyFitter(fMasterSharedAn, aMaxFitKStar);
  fMasterLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN2);
  fMasterLednickyFitter->SetApplyMomResCorrection(true);
  fMasterLednickyFitter->SetApplyNonFlatBackgroundCorrection(true);
  fMasterLednickyFitter->SetNonFlatBgdFitType(kLinear);
  fMasterLednickyFitter->SetIncludeResidualCorrelationsType(kInclude3Residuals);
  fMasterLednickyFitter->SetChargedResidualsType(kUseXiDataAndCoulombOnlyInterp);
  fMasterLednickyFitter->SetResPrimMaxDecayType(k4fm);
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw2::DoFit(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter2 = fMasterLednickyFitter;
  fMasterLednickyFitter->DoFit();
}







