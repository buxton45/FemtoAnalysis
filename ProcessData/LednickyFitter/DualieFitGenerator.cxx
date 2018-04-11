/* DualieFitGenerator.cxx */

#include "DualieFitGenerator.h"

#ifdef __ROOT__
ClassImp(DualieFitGenerator)
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
DualieFitGenerator::DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
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
DualieFitGenerator::DualieFitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
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
DualieFitGenerator::DualieFitGenerator(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2) :
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
DualieFitGenerator::~DualieFitGenerator()
{
  cout << "DualieFitGenerator object is being deleted!!!!!" << endl;
}


//________________________________________________________________________________________________________________
void DualieFitGenerator::CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii)
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
void DualieFitGenerator::CreateMasterSharedAn()
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
void DualieFitGenerator::CreateMinuitParameters(bool aShareLambda, bool aShareRadii)
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
void DualieFitGenerator::InitializeGenerator(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
//  if(fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll)
//  {
//   for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++) SetRadiusLimits(1.,12.,iCent);
//  }


  CreateMinuitParameters(aShareLambda, aShareRadii);

  fMasterLednickyFitter = new LednickyFitter(fMasterSharedAn, aMaxFitKStar);
  fMasterLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN2);

  fMasterLednickyFitter->SetApplyMomResCorrection(fFitGen1->fApplyMomResCorrection);
  fMasterLednickyFitter->SetApplyNonFlatBackgroundCorrection(fFitGen1->fApplyNonFlatBackgroundCorrection);
  fMasterLednickyFitter->SetNonFlatBgdFitType(fFitGen1->fNonFlatBgdFitType);
  fMasterLednickyFitter->SetIncludeResidualCorrelationsType(fFitGen1->fIncludeResidualsType);
  fMasterLednickyFitter->SetChargedResidualsType(fFitGen1->fChargedResidualsType);
  fMasterLednickyFitter->SetResPrimMaxDecayType(fFitGen1->fResPrimMaxDecayType);
}

//________________________________________________________________________________________________________________
void DualieFitGenerator::DoFit(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter2 = fMasterLednickyFitter;
  fMasterLednickyFitter->DoFit();

  ReturnNecessaryInfoToFitGenerators();
}


//________________________________________________________________________________________________________________
void DualieFitGenerator::ReturnNecessaryInfoToFitGenerators()
{
  //This isn't needed for the fitting process, just for drawing
  //Currently, it seems to only thing needed is a somewhat functioning LednickyFitter member
  //  and this fLednickyFitter only needs to know fChi2, fNDF and fKStarBinCenters
  //Making shallow copies, as I so frequently and dangerously I like to do, should work here

  fFitGen1->fLednickyFitter = fMasterLednickyFitter;
  fFitGen2->fLednickyFitter = fMasterLednickyFitter;
}


//******************************************* DRAWING ************************************************************

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawKStarCfs(bool aSaveImage, bool aDrawSysErrors)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfs(aSaveImage, aDrawSysErrors));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfs(aSaveImage, aDrawSysErrors));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFits_PartAn(aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFits_PartAn(aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFits(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFits(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawAllResiduals(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllResiduals(aSaveImage));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllResiduals(aSaveImage));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawAllResidualsWithTransformMatrices(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllResidualsWithTransformMatrices(aSaveImage));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllResidualsWithTransformMatrices(aSaveImage));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllSingleKStarCfwFitAndResiduals_PartAn(aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllSingleKStarCfwFitAndResiduals_PartAn(aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllSingleKStarCfwFitAndResiduals(aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllSingleKStarCfwFitAndResiduals(aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aZoomResiduals)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFitsAndResiduals(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aZoomResiduals));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFitsAndResiduals(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aZoomResiduals));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
TObjArray* DualieFitGenerator::DrawModelKStarCfs(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawModelKStarCfs(aSaveImage));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawModelKStarCfs(aSaveImage));
  return tReturnArray;
}





















