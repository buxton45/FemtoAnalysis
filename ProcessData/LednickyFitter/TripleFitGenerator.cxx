/* TripleFitGenerator.cxx */

#include "TripleFitGenerator.h"

#ifdef __ROOT__
ClassImp(TripleFitGenerator)
#endif

//GLOBAL!!!!!!!!!!!!!!!
LednickyFitter *GlobalFitter3 = NULL;

//______________________________________________________________________________
void GlobalFCN3(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalFitter3->CalculateFitFunction(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
TripleFitGenerator::TripleFitGenerator(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType_LamKch, AnalysisRunType aRunType_LamK0, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier_LamKch, TString aDirNameModifier_LamK0, bool aUseStavCf) :
  fFitGen1(nullptr),
  fFitGen2(nullptr),
  fFitGen3(nullptr),

  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, kLamKchP, aCentralityTypes, aRunType_LamKch, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamKch, aUseStavCf);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, kLamKchM, aCentralityTypes, aRunType_LamKch, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamKch, aUseStavCf);
  fFitGen3 = new FitGeneratorAndDraw(aFileLocationBase_LamK0, aFileLocationBaseMC_LamK0, kLamK0, aCentralityTypes, aRunType_LamK0, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamK0, aUseStavCf);

}


//________________________________________________________________________________________________________________
TripleFitGenerator::TripleFitGenerator(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0, CentralityType aCentralityType, AnalysisRunType aRunType_LamKch, AnalysisRunType aRunType_LamK0, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier_LamKch, TString aDirNameModifier_LamK0, bool aUseStavCf) :
  fFitGen1(nullptr),
  fFitGen2(nullptr),
  fFitGen3(nullptr),

  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, kLamKchP, aCentralityType, aRunType_LamKch, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamKch, aUseStavCf);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, kLamKchM, aCentralityType, aRunType_LamKch, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamKch, aUseStavCf);
  fFitGen3 = new FitGeneratorAndDraw(aFileLocationBase_LamK0, aFileLocationBaseMC_LamK0, kLamK0, aCentralityType, aRunType_LamK0, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier_LamK0, aUseStavCf);

}

//________________________________________________________________________________________________________________
TripleFitGenerator::TripleFitGenerator(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2) :
  fFitGen1(aFitGen1),
  fFitGen2(aFitGen2),
  fFitGen3(aFitGen2),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{

}

//________________________________________________________________________________________________________________
TripleFitGenerator::~TripleFitGenerator()
{
  cout << "TripleFitGenerator object is being deleted!!!!!" << endl;
}


//________________________________________________________________________________________________________________
void TripleFitGenerator::CreateMinuitParametersMatrix(bool aShareLambda, bool aShareRadii)
{
  //For now, not worrying about complications of "new" background treatment
  //     Anyway, I don't like this "new" method, and it will probably be abandoned
  assert(!fFitGen1->GetSharedAn()->UsingNewBgdTreatment());
  assert(!fFitGen2->GetSharedAn()->UsingNewBgdTreatment());
  assert(!fFitGen3->GetSharedAn()->UsingNewBgdTreatment());

  //TODO For now, LamK0 will have same sharing properties as LamKch
  // instead of case when fit separate, when I typically have all LamK0 share a single lambda parameter

  fFitGen1->SetAllParameters();
  fFitGen2->SetAllParameters();
  fFitGen3->SetAllParameters();

  assert(fFitGen1->GetSharedAn()->GetNFitParamsPerAnalysis() == fFitGen2->GetSharedAn()->GetNFitParamsPerAnalysis());
  assert(fFitGen1->GetSharedAn()->GetNFitParamsPerAnalysis() == fFitGen3->GetSharedAn()->GetNFitParamsPerAnalysis());
  int tNFitParamsPerAnalysis = fFitGen1->GetSharedAn()->GetNFitParamsPerAnalysis();

  assert(fFitGen1->GetSharedAn()->GetNFitPairAnalysis() == fFitGen2->GetSharedAn()->GetNFitPairAnalysis());
  assert(fFitGen1->GetSharedAn()->GetNFitPairAnalysis() == fFitGen3->GetSharedAn()->GetNFitPairAnalysis());
  int tNFitPairAnalysis = fFitGen1->GetSharedAn()->GetNFitPairAnalysis();

  for(int iPairAn=0; iPairAn<tNFitPairAnalysis; iPairAn++)
  {
    for(int iPar=0; iPar<tNFitParamsPerAnalysis; iPar++)
    {
      ParameterType tParamType = static_cast<ParameterType>(iPar);
      if((tParamType==kLambda && aShareLambda) || (tParamType==kRadius && aShareRadii)) 
      {
        fFitGen2->GetSharedAn()->GetFitPairAnalysis(iPairAn)->SetFitParameterShallow(fFitGen1->GetSharedAn()->GetFitPairAnalysis(iPairAn)->GetFitParameter(tParamType));
        fFitGen3->GetSharedAn()->GetFitPairAnalysis(iPairAn)->SetFitParameterShallow(fFitGen1->GetSharedAn()->GetFitPairAnalysis(iPairAn)->GetFitParameter(tParamType));
      }
    }
  }


  for(int iPar=0; iPar<tNFitParamsPerAnalysis; iPar++)
  {
    vector<FitParameter*> tTempParamVec(0);

    ParameterType tParamType = static_cast<ParameterType>(iPar);
    vector<FitParameter*> tTempParamVec1 = fFitGen1->GetSharedAn()->GetDistinctParamsOfCommonType(tParamType);
    vector<FitParameter*> tTempParamVec2 = fFitGen2->GetSharedAn()->GetDistinctParamsOfCommonType(tParamType);
    vector<FitParameter*> tTempParamVec3 = fFitGen3->GetSharedAn()->GetDistinctParamsOfCommonType(tParamType);
    assert(tTempParamVec1.size()==tTempParamVec2.size());
    assert(tTempParamVec1.size()==tTempParamVec3.size());

    for(unsigned int i=0; i<tTempParamVec1.size(); i++) tTempParamVec.push_back(tTempParamVec1[i]);

    if(tParamType==kLambda)  //NOTE: Cannot simply combine into if(tParamType==kLambda && !aShareLambda)
    {
      if(!aShareLambda)
      {
        for(unsigned int iLam=0; iLam<tTempParamVec2.size(); iLam++) tTempParamVec.push_back(tTempParamVec2[iLam]);
        for(unsigned int iLam=0; iLam<tTempParamVec3.size(); iLam++) tTempParamVec.push_back(tTempParamVec3[iLam]);
      }
    }
    else if(tParamType==kRadius)  //NOTE: Cannot simply combine into if(tParamType==kRadius && !aShareRadii)
    {
      if(!aShareRadii)
      {
        for(unsigned int iR=0; iR<tTempParamVec2.size(); iR++) tTempParamVec.push_back(tTempParamVec2[iR]);
        for(unsigned int iR=0; iR<tTempParamVec3.size(); iR++) tTempParamVec.push_back(tTempParamVec3[iR]);
      }
    }
    else
    {
      for(unsigned int i=0; i<tTempParamVec2.size(); i++) tTempParamVec.push_back(tTempParamVec2[i]);
      for(unsigned int i=0; i<tTempParamVec3.size(); i++) tTempParamVec.push_back(tTempParamVec3[i]);
    }

    fMasterMinuitFitParametersMatrix.push_back(tTempParamVec);
  }
  //--------------------------------

}


//________________________________________________________________________________________________________________
void TripleFitGenerator::CreateMasterSharedAn()
{
  vector<FitPairAnalysis*> tTempFitPairAnColl(0);
  for(unsigned int i=0; i<fFitGen1->GetSharedAn()->GetFitPairAnalysisCollection().size(); i++) tTempFitPairAnColl.push_back(fFitGen1->GetSharedAn()->GetFitPairAnalysisCollection()[i]);
  for(unsigned int i=0; i<fFitGen2->GetSharedAn()->GetFitPairAnalysisCollection().size(); i++) tTempFitPairAnColl.push_back(fFitGen2->GetSharedAn()->GetFitPairAnalysisCollection()[i]);
  for(unsigned int i=0; i<fFitGen3->GetSharedAn()->GetFitPairAnalysisCollection().size(); i++) tTempFitPairAnColl.push_back(fFitGen3->GetSharedAn()->GetFitPairAnalysisCollection()[i]);

  fMasterSharedAn = new FitSharedAnalyses(tTempFitPairAnColl);
  fMasterSharedAn->SetFitType(fFitGen1->GetSharedAn()->GetFitType());
  fMasterSharedAn->SetApplyNonFlatBackgroundCorrection(fFitGen1->GetSharedAn()->GetApplyNonFlatBackgroundCorrection());

  fMasterSharedAn->SetNonFlatBgdFitType(kLamKchP, fFitGen1->GetSharedAn()->GetNonFlatBgdFitType());
  fMasterSharedAn->SetNonFlatBgdFitType(kLamK0, fFitGen3->GetSharedAn()->GetNonFlatBgdFitType());

  fMasterSharedAn->SetUseNewBgdTreatment(fFitGen1->GetSharedAn()->UsingNewBgdTreatment());
  fMasterSharedAn->SetFixNormParams(fFitGen1->GetSharedAn()->GetFixNormParams());

}

//________________________________________________________________________________________________________________
void TripleFitGenerator::CreateMinuitParameters(bool aShareLambda, bool aShareRadii)
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
void TripleFitGenerator::InitializeGenerator(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  if(fFitGen1->fIncludeResidualsType != kIncludeNoResiduals && fFitGen1->fChargedResidualsType != kUseXiDataForAll)
  {
   for(unsigned int iCent=0; iCent<fFitGen1->fCentralityTypes.size(); iCent++) SetRadiusLimits(1.,12.,iCent);
  }


  CreateMinuitParameters(aShareLambda, aShareRadii);

  fMasterLednickyFitter = new LednickyFitter(fMasterSharedAn, aMaxFitKStar);
  fMasterLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN3);

  fMasterLednickyFitter->SetApplyMomResCorrection(fFitGen1->fApplyMomResCorrection);
  fMasterLednickyFitter->SetApplyNonFlatBackgroundCorrection(fFitGen1->fApplyNonFlatBackgroundCorrection);

  fMasterLednickyFitter->SetNonFlatBgdFitType(kLamKchP, fFitGen1->fNonFlatBgdFitType);
  fMasterLednickyFitter->SetNonFlatBgdFitType(kLamK0, fFitGen3->fNonFlatBgdFitType);

  fMasterLednickyFitter->SetIncludeResidualCorrelationsType(fFitGen1->fIncludeResidualsType);
  fMasterLednickyFitter->SetChargedResidualsType(fFitGen1->fChargedResidualsType);
  fMasterLednickyFitter->SetResPrimMaxDecayType(fFitGen1->fResPrimMaxDecayType);
}

//________________________________________________________________________________________________________________
void TripleFitGenerator::DoFit(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter3 = fMasterLednickyFitter;
  fMasterLednickyFitter->DoFit();

  ReturnNecessaryInfoToFitGenerators();
}


//________________________________________________________________________________________________________________
void TripleFitGenerator::ReturnNecessaryInfoToFitGenerators()
{
  //This isn't needed for the fitting process, just for drawing
  //Currently, it seems to only thing needed is a somewhat functioning LednickyFitter member
  //  and this fLednickyFitter only needs to know fChi2, fNDF and fKStarBinCenters
  //Making shallow copies, as I so frequently and dangerously I like to do, should work here

  fFitGen1->fLednickyFitter = fMasterLednickyFitter;
  fFitGen2->fLednickyFitter = fMasterLednickyFitter;
  fFitGen3->fLednickyFitter = fMasterLednickyFitter;
}


//******************************************* DRAWING ************************************************************

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawKStarCfs(bool aSaveImage, bool aDrawSysErrors)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfs(aSaveImage, aDrawSysErrors));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfs(aSaveImage, aDrawSysErrors));
  tReturnArray->Add((TCanvas*)fFitGen3->DrawKStarCfs(aSaveImage, aDrawSysErrors));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aZoomROP)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFits_PartAn(aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchP], aSaveImage, aZoomROP));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFits_PartAn(aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchM], aSaveImage, aZoomROP));
  tReturnArray->Add((TCanvas*)fFitGen3->DrawKStarCfswFits_PartAn(aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamK0], aSaveImage, aZoomROP));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFits(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchP], aSaveImage, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFits(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchM], aSaveImage, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines));
  tReturnArray->Add((TCanvas*)fFitGen3->DrawKStarCfswFits(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamK0], aSaveImage, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawAllResiduals(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllResiduals(aSaveImage));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllResiduals(aSaveImage));
  tReturnArray->Add((TObjArray*)fFitGen3->DrawAllResiduals(aSaveImage));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawAllResidualsWithTransformMatrices(bool aSaveImage, bool aDrawv2)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllResidualsWithTransformMatrices(aSaveImage, aDrawv2));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllResidualsWithTransformMatrices(aSaveImage, aDrawv2));
  tReturnArray->Add((TObjArray*)fFitGen3->DrawAllResidualsWithTransformMatrices(aSaveImage, aDrawv2));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllSingleKStarCfwFitAndResiduals_PartAn(aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchP], aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllSingleKStarCfwFitAndResiduals_PartAn(aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchM], aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen3->DrawAllSingleKStarCfwFitAndResiduals_PartAn(aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamK0], aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TObjArray*)fFitGen1->DrawAllSingleKStarCfwFitAndResiduals(aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchP], aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen2->DrawAllSingleKStarCfwFitAndResiduals(aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchM], aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  tReturnArray->Add((TObjArray*)fFitGen3->DrawAllSingleKStarCfwFitAndResiduals(aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamK0], aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aZoomResiduals)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawKStarCfswFitsAndResiduals(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchP], aSaveImage, aDrawSysErrors, aZoomROP, aZoomResiduals));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawKStarCfswFitsAndResiduals(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamKchM], aSaveImage, aDrawSysErrors, aZoomROP, aZoomResiduals));
  tReturnArray->Add((TCanvas*)fFitGen3->DrawKStarCfswFitsAndResiduals(aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[kLamK0], aSaveImage, aDrawSysErrors, aZoomROP, aZoomResiduals));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
TObjArray* TripleFitGenerator::DrawModelKStarCfs(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)fFitGen1->DrawModelKStarCfs(aSaveImage));
  tReturnArray->Add((TCanvas*)fFitGen2->DrawModelKStarCfs(aSaveImage));
  tReturnArray->Add((TCanvas*)fFitGen3->DrawModelKStarCfs(aSaveImage));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
void TripleFitGenerator::WriteToMasterFitValuesFile(TString aFileLocation_LamKch, TString aFileLocation_LamK0, TString aResultsDate)
{
  fFitGen1->WriteToMasterFitValuesFile(aFileLocation_LamKch, aResultsDate);
  fFitGen2->WriteToMasterFitValuesFile(aFileLocation_LamKch, aResultsDate);
  fFitGen3->WriteToMasterFitValuesFile(aFileLocation_LamK0, aResultsDate);
}

















