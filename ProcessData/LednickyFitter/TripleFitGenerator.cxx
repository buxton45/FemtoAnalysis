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
void TripleFitGenerator::DoFit(bool aShareLambda, bool aShareRadii, double aMaxFitKStar, bool aOutputCorrCoeffFile)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter3 = fMasterLednickyFitter;
  
  //Set directory for aOutputCorrCoeffFile to be saved
  if(aOutputCorrCoeffFile)
  {
    fFitGen1->ExistsSaveLocationBase();
    fMasterLednickyFitter->SetSaveLocationBase(fFitGen1->GetSaveLocationBase());
  }  
  
  fMasterLednickyFitter->DoFit(aOutputCorrCoeffFile);

  ReturnNecessaryInfoToFitGenerators();
}

//________________________________________________________________________________________________________________
void TripleFitGenerator::DoFitOnce(bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter3 = fMasterLednickyFitter;
  fMasterLednickyFitter->DoFitOnce();

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

//________________________________________________________________________________________________________________
TCanvas* TripleFitGenerator::GenerateContourPlots(int aNPoints, const vector<double> &aParams, const vector<double> &aErrVals, TString aSaveNameModifier, bool aFixAllOthers, bool aShareLambda, bool aShareRadii, double aMaxFitKStar)
{
  InitializeGenerator(aShareLambda, aShareRadii, aMaxFitKStar);
  GlobalFitter3 = fMasterLednickyFitter;
  TCanvas* tReturnCan = fMasterLednickyFitter->GenerateContourPlots(aNPoints, aParams, aErrVals, aSaveNameModifier, aFixAllOthers);
  return tReturnCan;
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



//________________________________________________________________________________________________________________
CanvasPartition* TripleFitGenerator::BuildKStarCfswFitsCanvasPartition_CombineConj_AllAn(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines, bool aDrawErrBands)
{
  assert(fFitGen1->GetNAnalyses() == 6);
  assert(fFitGen2->GetNAnalyses() == 6);
  assert(fFitGen3->GetNAnalyses() == 6);

  TString tCanvasName = aCanvasBaseName;
  if(!aZoomROP) tCanvasName += TString("UnZoomed");
  if(aLabelLines) tCanvasName += TString("_LabelLines");
  if(aDrawErrBands) tCanvasName += TString("_wErrBands");

  int tNx=3, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  //double tYLow = 0.86;
  //double tYHigh = 1.07;
  double tYLow = 0.83;
  double tYHigh = 1.09;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }
  if(aDrawErrBands)
  {
    tYLow = 0.76;
    tYHigh = 1.15;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.11,0.0025,0.10,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100, 1500);
  tCanPart->SetAllTicks(1,1);

  int tAnalysisNumberA=0, tAnalysisNumberB=0;
  for(int j=0; j<tNy; j++)
  {
    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = tAnalysisNumberA+1;

    assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
    assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
    CentralityType tCentType = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    //-------------------------------------------------------------------------------------------------------------------------------
    AnalysisType tAnType1   = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
    AnalysisType tConjType1 = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

    AnalysisType tAnType2   = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
    AnalysisType tConjType2 = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

    AnalysisType tAnType3   = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
    AnalysisType tConjType3 = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

    //------------------------------

    fFitGen1->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, 0, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, aDrawErrBands);
    fFitGen2->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, 1, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, aDrawErrBands);
    fFitGen3->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, 2, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, aDrawErrBands);

    //------------------------------
    double tTextSize = 50;
    if(aZoomROP) tTextSize = 56;

    double tTextPosY = 0.75;
    if(aZoomROP) tTextPosY = 0.75;

    TString tSysTypeText1 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tConjType1]);
    TPaveText* tSysType1 = tCanPart->SetupTPaveText(tSysTypeText1, 0, j, 0.25, tTextPosY, 0.25, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tSysType1, 0, j);

    TString tCentTypeText1 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
    TPaveText* tCentType1 = tCanPart->SetupTPaveText(tCentTypeText1, 0, j, 0.75, tTextPosY, 0.10, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tCentType1, 0, j);

    //-----

    TString tSysTypeText2 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType2], cAnalysisRootTags[tConjType2]);
    TPaveText* tSysType2 = tCanPart->SetupTPaveText(tSysTypeText2, 1, j, 0.25, tTextPosY, 0.25, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tSysType2, 1, j);

    TString tCentTypeText2 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
    TPaveText* tCentType2 = tCanPart->SetupTPaveText(tCentTypeText2, 1, j, 0.75, tTextPosY, 0.10, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tCentType2, 1, j);

    //-----

    TString tSysTypeText3 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType3], cAnalysisRootTags[tConjType3]);
    TPaveText* tSysType3 = tCanPart->SetupTPaveText(tSysTypeText3, 2, j, 0.25, (tTextPosY-0.025), 0.25, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tSysType3, 2, j);

    TString tCentTypeText3 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
    TPaveText* tCentType3 = tCanPart->SetupTPaveText(tCentTypeText3, 2, j, 0.75, (tTextPosY-0.025), 0.10, 0.15, 43, tTextSize, 21, true);
    tCanPart->AddPadPaveText(tCentType3, 2, j);

    //------------------------------

    TString tMasterFileLocation1      = fFitGen1->GetMasterFileLocation();
    TString tSystematicsFileLocation1 = fFitGen1->GetSystematicsFileLocation();
    td1dVec tSysErrors1;
    if(tMasterFileLocation1.IsNull() || tSystematicsFileLocation1.IsNull()) 
    {
      if(aDrawSysErrors)
      {
        cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
      tSysErrors1 = fFitGen1->GetSystErrs(fFitGen1->fIncludeResidualsType, tAnType1, tCentType);
    }
    else tSysErrors1 = fFitGen1->GetSystErrs(tMasterFileLocation1, tSystematicsFileLocation1, fFitGen1->GetSaveNameModifier(), tAnType1, tCentType);

    TString tMasterFileLocation2      = fFitGen2->GetMasterFileLocation();
    TString tSystematicsFileLocation2 = fFitGen2->GetSystematicsFileLocation();
    td1dVec tSysErrors2;
    if(tMasterFileLocation2.IsNull() || tSystematicsFileLocation2.IsNull()) 
    {
      if(aDrawSysErrors)
      {
        cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
      tSysErrors2 = fFitGen2->GetSystErrs(fFitGen2->fIncludeResidualsType, tAnType2, tCentType);
    }
    else tSysErrors2 = fFitGen2->GetSystErrs(tMasterFileLocation2, tSystematicsFileLocation2, fFitGen2->GetSaveNameModifier(), tAnType2, tCentType);

    TString tMasterFileLocation3      = fFitGen3->GetMasterFileLocation();
    TString tSystematicsFileLocation3 = fFitGen3->GetSystematicsFileLocation();
    td1dVec tSysErrors3;
    if(tMasterFileLocation3.IsNull() || tSystematicsFileLocation3.IsNull()) 
    {
      if(aDrawSysErrors)
      {
        cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
      tSysErrors3 = fFitGen3->GetSystErrs(fFitGen3->fIncludeResidualsType, tAnType3, tCentType);
    }
    else tSysErrors3 = fFitGen3->GetSystErrs(tMasterFileLocation3, tSystematicsFileLocation3, fFitGen3->GetSaveNameModifier(), tAnType3, tCentType);

    //------------------------------

    bool bDrawAll = false;
    if(j==0) bDrawAll = true;
    if(!aSuppressFitInfoOutput) fFitGen1->CreateParamFinalValuesText(tAnType1, tCanPart, 0, j, (TF1*)fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors1, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    if(!aSuppressFitInfoOutput) fFitGen2->CreateParamFinalValuesText(tAnType2, tCanPart, 1, j, (TF1*)fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors2, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    if(!aSuppressFitInfoOutput) fFitGen3->CreateParamFinalValuesText(tAnType3, tCanPart, 2, j, (TF1*)fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors3, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    //------------------------------

    if(j==(tNy-1))
    {
/*
      TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
      TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,2,j,0.20,0.125,0.725,0.15,43,17);
      tCanPart->AddPadPaveText(tSysInfo,2,j);
*/
/*
    TString tTextSysInfo1 = TString("ALICE");
    TPaveText* tSysInfo1 = tCanPart->SetupTPaveText(tTextSysInfo1, 1, j, 0.0, 0.10, 1., 0.15, 43, 25, 22, true);
    tCanPart->AddPadPaveText(tSysInfo1,1, j);
*/
/*
    TString tTextSysInfo2 = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    TPaveText* tSysInfo2 = tCanPart->SetupTPaveText(tTextSysInfo2, 1, j, 0.0, 0.10, 1., 0.15, 43, 45, 22, true);
    tCanPart->AddPadPaveText(tSysInfo2,1, j);
    */
    TString tTextSysInfo = TString("ALICE");
    TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo, 0, j, 0.0, 0.10, 1., 0.15, 43, 57, 22, true);
    tCanPart->AddPadPaveText(tSysInfo,0, j);
   
    TString tTextSysInfo2 = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    TPaveText* tSysInfo2 = tCanPart->SetupTPaveText(tTextSysInfo2, 1, j, 0.0, 0.10, 1., 0.15, 43, 57, 22, true);
    tCanPart->AddPadPaveText(tSysInfo2,1, j);        
    }

    //----------
    double tLabelScaleX = 3.5;
    double tLabelScaleY = 3.5;
    ((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetXaxis()->SetLabelSize(tLabelScaleX*((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetXaxis()->GetLabelSize());
    ((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetYaxis()->SetLabelSize(tLabelScaleY*((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetYaxis()->GetLabelSize());

    ((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetXaxis()->SetLabelSize(tLabelScaleX*((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetXaxis()->GetLabelSize());
    ((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetYaxis()->SetLabelSize(tLabelScaleY*((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetYaxis()->GetLabelSize());

    ((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetXaxis()->SetLabelSize(tLabelScaleX*((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetXaxis()->GetLabelSize());
    ((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetYaxis()->SetLabelSize(tLabelScaleY*((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetYaxis()->GetLabelSize());

    //----------
    double tLabelOffsetScaleX = 2.0;
    double tLabelOffsetScaleY = 2.0;
    ((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetXaxis()->SetLabelOffset(tLabelOffsetScaleX*((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetXaxis()->GetLabelOffset());    
    ((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetYaxis()->SetLabelOffset(tLabelOffsetScaleY*((TH1*)tCanPart->GetGraphsInPad(0,j)->At(0))->GetYaxis()->GetLabelOffset());

    ((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetXaxis()->SetLabelOffset(tLabelOffsetScaleX*((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetXaxis()->GetLabelOffset());
    ((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetYaxis()->SetLabelOffset(tLabelOffsetScaleY*((TH1*)tCanPart->GetGraphsInPad(1,j)->At(0))->GetYaxis()->GetLabelOffset());

    ((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetXaxis()->SetLabelOffset(tLabelOffsetScaleX*((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetXaxis()->GetLabelOffset());
    ((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetYaxis()->SetLabelOffset(tLabelOffsetScaleY*((TH1*)tCanPart->GetGraphsInPad(2,j)->At(0))->GetYaxis()->GetLabelOffset());
  }

  //------------------------------ Change marker size
  //Since pads are much larger now, increase the width of the lines and size of markers
  //However, if I blindly increase the line width of all, the error bar lines become too fat, so I must do so only for TF1* objects
  //Obnoxiously, I cast everything as TH1 (don't remember why, but probably there was a reason), 
  //  so cannot simply check the typeid and I need to check the drawing options instead!
  double tMarkerSize = 1.75;
  double tLineWidth = 2.0;
  for(int i=0; i<tNx; i++)
  {
    for(int j=0; j<tNy; j++)
    {
      vector<TString> tGraphsDrawOptionsInPad = tCanPart->GetGraphsDrawOptionsInPad(i,j);
      for(int iGr=0; iGr<tCanPart->GetGraphsInPad(i, j)->GetEntries(); iGr++)
      {

        if(tGraphsDrawOptionsInPad[iGr].Contains("ex0"))  tLineWidth = 1.0;
        else                                              tLineWidth = 2.0;
        ((TH1*)tCanPart->GetGraphsInPad(i, j)->At(iGr))->SetLineWidth(tLineWidth);
        ((TH1*)tCanPart->GetGraphsInPad(i, j)->At(iGr))->SetMarkerSize(tMarkerSize);
      }
    }
  }


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 75, 0.825, 0.005); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 75, 0.05, 0.875); 


  if(aLabelLines)
  {
    assert(aSuppressFitInfoOutput);  //Not enough room for everyone!
//    fFitGen1->AddColoredLinesLabels(tCanPart, 0, 0, aZoomROP);
    fFitGen1->AddColoredLinesLabelsAndDatav2(tCanPart, 0, 0, aZoomROP);
  }

  return tCanPart;
}

//________________________________________________________________________________________________________________
TCanvas* TripleFitGenerator::DrawKStarCfswFits_CombineConj_AllAn(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines, bool aDrawErrBands)
{
  TString tCanvasBaseName = "canKStarCfwFits_CombineConj_AllAn";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition_CombineConj_AllAn(tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines, aDrawErrBands);

  if(aSaveImage)
  {
    fFitGen1->ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fFitGen1->GetSaveLocationBase()+tCanPart->GetCanvas()->GetName()+TString::Format(".%s", fFitGen1->GetSaveFileType().Data()));
  }

  return tCanPart->GetCanvas();
}






//________________________________________________________________________________________________________________
CanvasPartition* TripleFitGenerator::BuildKStarCfswFitsCanvasPartition_CombineConj(TString aCanvasBaseName, CentralityType aCentType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  assert(fFitGen1->GetNAnalyses() == 6);
  assert(fFitGen2->GetNAnalyses() == 6);
  assert(fFitGen3->GetNAnalyses() == 6);

  TString tCanvasName = aCanvasBaseName;
  if(!aZoomROP) tCanvasName += TString("UnZoomed");
  if(aLabelLines) tCanvasName += TString("_LabelLines");

  int tNx=3, tNy=1;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.075,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100,500);

  assert(aCentType <= k3050);
  int tAnalysisNumberA = 2*(int)aCentType;
  int tAnalysisNumberB = tAnalysisNumberA+1;


  assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
  assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
  CentralityType tCentType = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
  //-------------------------------------------------------------------------------------------------------------------------------
  AnalysisType tAnType1   = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType1 = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  AnalysisType tAnType2   = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType2 = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  AnalysisType tAnType3   = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType3 = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  //------------------------------
  int tNx1 =      0, tNy1 =      0;
  int tNx2 =      1, tNy2 =      0;
  int tNx3 =      2, tNy3 =      0;

  fFitGen1->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx1, tNy1, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, false);
  fFitGen2->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx2, tNy2, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, false);
  fFitGen3->BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx3, tNy3, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes[tAnType1], aDrawSysErrors, aZoomROP, false);

  //------------------------------ Change marker size
  for(int i=0; i<tNx; i++)
  {
    for(int j=0; j<tNy; j++)
    {
      for(int iGr=0; iGr<tCanPart->GetGraphsInPad(i, j)->GetEntries(); iGr++)
      {
        ((TH1*)tCanPart->GetGraphsInPad(i, j)->At(iGr))->SetMarkerSize(1.5);
      }
    }
  }


  //------------------------------
  double tTextSize = 60;
  if(aZoomROP) tTextSize = 60;

  double tTextPosY = 0.775;
  if(aZoomROP) tTextPosY = 0.775;

  TString tSysTypeText1 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tConjType1]);
  TPaveText* tSysType1 = tCanPart->SetupTPaveText(tSysTypeText1, tNx1, tNy1, 0.20, tTextPosY, 0.25, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tSysType1, tNx1, tNy1);

  TString tCentTypeText1 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
  TPaveText* tCentType1 = tCanPart->SetupTPaveText(tCentTypeText1, tNx1, tNy1, 0.775, tTextPosY, 0.10, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tCentType1, tNx1, tNy1);

    //-----

  TString tSysTypeText2 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType2], cAnalysisRootTags[tConjType2]);
  TPaveText* tSysType2 = tCanPart->SetupTPaveText(tSysTypeText2, tNx2, tNy2, 0.20, tTextPosY, 0.25, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tSysType2, tNx2, tNy2);

  TString tCentTypeText2 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
  TPaveText* tCentType2 = tCanPart->SetupTPaveText(tCentTypeText2, tNx2, tNy2, 0.775, tTextPosY, 0.10, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tCentType2, tNx2, tNy2);

    //-----

  TString tSysTypeText3 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s", cAnalysisRootTags[tAnType3], cAnalysisRootTags[tConjType3]);
  TPaveText* tSysType3 = tCanPart->SetupTPaveText(tSysTypeText3, tNx3, tNy3, 0.20, (tTextPosY-0.025), 0.25, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tSysType3, tNx3, tNy3);

  TString tCentTypeText3 = TString::Format("%s", cPrettyCentralityTags[tCentType]);
  TPaveText* tCentType3 = tCanPart->SetupTPaveText(tCentTypeText3, tNx3, tNy3, 0.775, (tTextPosY-0.025), 0.10, 0.15, 43, tTextSize, 21, true);
  tCanPart->AddPadPaveText(tCentType3, tNx3, tNy3);

    //------------------------------




  //------------------------------

  TString tMasterFileLocation1      = fFitGen1->GetMasterFileLocation();
  TString tSystematicsFileLocation1 = fFitGen1->GetSystematicsFileLocation();
  td1dVec tSysErrors1;
  if(tMasterFileLocation1.IsNull() || tSystematicsFileLocation1.IsNull()) 
  {
    if(aDrawSysErrors)
    {
      cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
    tSysErrors1 = fFitGen1->GetSystErrs(fFitGen1->fIncludeResidualsType, tAnType1, tCentType);
  }
  else tSysErrors1 = fFitGen1->GetSystErrs(tMasterFileLocation1, tSystematicsFileLocation1, fFitGen1->GetSaveNameModifier(), tAnType1, tCentType);

  TString tMasterFileLocation2      = fFitGen2->GetMasterFileLocation();
  TString tSystematicsFileLocation2 = fFitGen2->GetSystematicsFileLocation();
  td1dVec tSysErrors2;
  if(tMasterFileLocation2.IsNull() || tSystematicsFileLocation2.IsNull()) 
  {
    if(aDrawSysErrors)
    {
      cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
    tSysErrors2 = fFitGen2->GetSystErrs(fFitGen2->fIncludeResidualsType, tAnType2, tCentType);
  }
  else tSysErrors2 = fFitGen2->GetSystErrs(tMasterFileLocation2, tSystematicsFileLocation2, fFitGen2->GetSaveNameModifier(), tAnType2, tCentType);

  TString tMasterFileLocation3      = fFitGen3->GetMasterFileLocation();
  TString tSystematicsFileLocation3 = fFitGen3->GetSystematicsFileLocation();
  td1dVec tSysErrors3;
  if(tMasterFileLocation3.IsNull() || tSystematicsFileLocation3.IsNull()) 
  {
    if(aDrawSysErrors)
    {
      cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
    tSysErrors3 = fFitGen3->GetSystErrs(fFitGen3->fIncludeResidualsType, tAnType3, tCentType);
  }
  else tSysErrors3 = fFitGen3->GetSystErrs(tMasterFileLocation3, tSystematicsFileLocation3, fFitGen3->GetSaveNameModifier(), tAnType3, tCentType);

  //------------------------------

    bool bDrawAll = true;
    if(!aSuppressFitInfoOutput) fFitGen1->CreateParamFinalValuesText(tAnType1, tCanPart, tNx1, tNy1, (TF1*)fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors1, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    if(!aSuppressFitInfoOutput) fFitGen2->CreateParamFinalValuesText(tAnType2, tCanPart, tNx2, tNy2, (TF1*)fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors2, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    if(!aSuppressFitInfoOutput) fFitGen3->CreateParamFinalValuesText(tAnType3, tCanPart, tNx3, tNy3, (TF1*)fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors3, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);

    //------------------------------

/*
  TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo, tNx3, tNy3, 0.20, 0.125, 0.725, 0.15, 43, 35);
  tCanPart->AddPadPaveText(tSysInfo,tNx3, tNy3);
*/

  TString tTextSysInfo1 = TString("ALICE Preliminary");
  TPaveText* tSysInfo1 = tCanPart->SetupTPaveText(tTextSysInfo1, tNx3, tNy3, 0.0, 0.10, 1., 0.15, 43, 50, 22, true);
  tCanPart->AddPadPaveText(tSysInfo1,tNx2, tNy2);

  TString tTextSysInfo2 = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo2 = tCanPart->SetupTPaveText(tTextSysInfo2, tNx3, tNy3, 0.0, 0.10, 1., 0.15, 43, 50, 22, true);
  tCanPart->AddPadPaveText(tSysInfo2,tNx3, tNy3);

  //------------
  ((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetXaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->GetLabelSize());

  ((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetXaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->GetLabelSize());

  ((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetXaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->SetLabelSize(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->GetLabelSize());

  //------------

  ((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->SetLabelOffset(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->GetLabelOffset());

  ((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->SetLabelOffset(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->GetLabelOffset());

  ((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->SetLabelOffset(2.0*((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->GetLabelOffset());


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 55, 0.010);
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 55, 0.025, 0.70); 
 


  if(aLabelLines)
  {
    assert(aSuppressFitInfoOutput);  //Not enough room for everyone!
    fFitGen1->AddColoredLinesLabels(tCanPart, tNx1, tNy1, aZoomROP, 0.75);
  }


  return tCanPart;
}


//________________________________________________________________________________________________________________
CanvasPartition* TripleFitGenerator::BuildKStarCfsCanvasPartition_CombineConj(TString aCanvasBaseName, CentralityType aCentType, bool aDrawSysErrors, bool aZoomROP)
{
  assert(fFitGen1->GetNAnalyses() == 6);
  assert(fFitGen2->GetNAnalyses() == 6);
  assert(fFitGen3->GetNAnalyses() == 6);

  TString tCanvasName = aCanvasBaseName;
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

  int tNx=3, tNy=1;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.075,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100,500);

  assert(aCentType <= k3050);
  int tAnalysisNumberA = 2*(int)aCentType;
  int tAnalysisNumberB = tAnalysisNumberA+1;


  assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
  assert(fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType());
  CentralityType tCentType = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
  //-------------------------------------------------------------------------------------------------------------------------------
  AnalysisType tAnType1   = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType1 = fFitGen1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  AnalysisType tAnType2   = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType2 = fFitGen2->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  AnalysisType tAnType3   = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
  AnalysisType tConjType3 = fFitGen3->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

  //------------------------------
  int tNx1 =      0, tNy1 =      0;
  int tNx2 =      1, tNy2 =      0;
  int tNx3 =      2, tNy3 =      0;

  fFitGen1->BuildKStarCfsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx1, tNy1, aDrawSysErrors, aZoomROP);
  fFitGen2->BuildKStarCfsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx2, tNy2, aDrawSysErrors, aZoomROP);
  fFitGen3->BuildKStarCfsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, tNx3, tNy3, aDrawSysErrors, aZoomROP);

  //------------------------------ Change marker size
  for(int i=0; i<tNx; i++)
  {
    for(int j=0; j<tNy; j++)
    {
      for(int iGr=0; iGr<tCanPart->GetGraphsInPad(i, j)->GetEntries(); iGr++)
      {
        ((TH1*)tCanPart->GetGraphsInPad(i, j)->At(iGr))->SetMarkerSize(1.5);
      }
    }
  }


  //------------------------------

  TString tCombinedText1 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s  %s", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tConjType1], cPrettyCentralityTags[tCentType]);
  TPaveText* tCombined1 = tCanPart->SetupTPaveText(tCombinedText1, tNx1, tNy1, 0.60, 0.835, 0.15, 0.10, 43, 40);;
  tCanPart->AddPadPaveText(tCombined1, tNx1, tNy1);

  TString tCombinedText2 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s  %s", cAnalysisRootTags[tAnType2], cAnalysisRootTags[tConjType2], cPrettyCentralityTags[tCentType]);
  TPaveText* tCombined2 = tCanPart->SetupTPaveText(tCombinedText2, tNx2, tNy2, 0.60, 0.835, 0.15, 0.10, 43, 40);;
  tCanPart->AddPadPaveText(tCombined2, tNx2, tNy2);

  TString tCombinedText3 = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s  %s", cAnalysisRootTags[tAnType3], cAnalysisRootTags[tConjType3], cPrettyCentralityTags[tCentType]);
  TPaveText* tCombined3 = tCanPart->SetupTPaveText(tCombinedText3, tNx3, tNy3, 0.60, 0.835, 0.15, 0.10, 43, 40);;
  tCanPart->AddPadPaveText(tCombined3, tNx3, tNy3);

    //------------------------------

/*
  TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo, tNx3, tNy3, 0.20, 0.125, 0.725, 0.15, 43, 35);
  tCanPart->AddPadPaveText(tSysInfo,tNx3, tNy3);
*/

  TString tTextSysInfo1 = TString("ALICE Preliminary");
  TPaveText* tSysInfo1 = tCanPart->SetupTPaveText(tTextSysInfo1, tNx3, tNy3, 0.15, 0.225, 0.725, 0.15, 43, 35, 12);
  tCanPart->AddPadPaveText(tSysInfo1,tNx3, tNy3);

  TString tTextSysInfo2 = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo2 = tCanPart->SetupTPaveText(tTextSysInfo2, tNx3, tNy3, 0.15, 0.075, 0.725, 0.15, 43, 35, 12);
  tCanPart->AddPadPaveText(tSysInfo2,tNx3, tNy3);

  ((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetXaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx1, tNy1)->At(0))->GetYaxis()->GetLabelSize());

  ((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetXaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx2, tNy2)->At(0))->GetYaxis()->GetLabelSize());

  ((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetXaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetXaxis()->GetLabelSize());
  ((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->SetLabelSize(1.75*((TH1*)tCanPart->GetGraphsInPad(tNx3, tNy3)->At(0))->GetYaxis()->GetLabelSize());



  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 55, 0.010);
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 55, 0.025, 0.70); 

  return tCanPart;
}


//________________________________________________________________________________________________________________
TCanvas* TripleFitGenerator::DrawKStarCfswFits_CombineConj(CentralityType aCentType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  TString tCanvasBaseName = TString::Format("canKStarCfwFits_CombineConj%s", cCentralityTags[aCentType]);
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition_CombineConj(tCanvasBaseName, aCentType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitTypes, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines);

  if(aSaveImage)
  {
    fFitGen1->ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fFitGen1->GetSaveLocationBase()+tCanPart->GetCanvas()->GetName()+TString::Format(".%s", fFitGen1->GetSaveFileType().Data()));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
TCanvas* TripleFitGenerator::DrawKStarCfs_CombineConj(CentralityType aCentType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP)
{
  TString tCanvasBaseName = TString::Format("canKStarCf_CombineConj%s", cCentralityTags[aCentType]);
  CanvasPartition* tCanPart = BuildKStarCfsCanvasPartition_CombineConj(tCanvasBaseName, aCentType, aDrawSysErrors, aZoomROP);

  if(aSaveImage)
  {
    fFitGen1->ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fFitGen1->GetSaveLocationBase()+tCanPart->GetCanvas()->GetName()+TString::Format(".%s", fFitGen1->GetSaveFileType().Data()));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
td2dVec TripleFitGenerator::GetParamsMinMaxForErrBands_FINALRESULTSONLY(AnalysisType aAnType)
{
  assert(aAnType==kLamKchP || aAnType==kLamKchM || aAnType==kLamK0);

  td1dVec tMinParams = fMasterLednickyFitter->GetMinParams();
  assert(tMinParams.size()==51);
  
  td3dVec tParamsMinMax_LamKchP = fFitGen1->GetParamsMinMaxForErrBands_FINALRESULTSONLY();
  td3dVec tParamsMinMax_LamKchM = fFitGen2->GetParamsMinMaxForErrBands_FINALRESULTSONLY();
  td3dVec tParamsMinMax_LamK0   = fFitGen3->GetParamsMinMaxForErrBands_FINALRESULTSONLY();
  
  td2dVec tReturnVec(tMinParams.size());
  
  //Lambda params, take from LamKchP0010, 1030, and 3050
  tReturnVec[0] = tParamsMinMax_LamKchP[0][0];
  tReturnVec[1] = tParamsMinMax_LamKchP[1][0];
  tReturnVec[2] = tParamsMinMax_LamKchP[2][0];

  //Radius params, take from LamKchP0010, 1030, and 3050  
  tReturnVec[3] = tParamsMinMax_LamKchP[0][1];
  tReturnVec[4] = tParamsMinMax_LamKchP[1][1];
  tReturnVec[5] = tParamsMinMax_LamKchP[2][1];
  
  //Ref0, taken from LamKchP0010, LamKchM0010, and LamK00010
  //Imf0, taken from LamKchP0010, LamKchM0010, and LamK00010
  //d0, taken from LamKchP0010, LamKchM0010, and LamK00010
  for(unsigned int i=6; i<15; i++)
  {
    tReturnVec[i] = vector<double>{tMinParams[i]};
  }
  if(aAnType==kLamKchP)
  {
    tReturnVec[6] = tParamsMinMax_LamKchP[0][2];
    tReturnVec[9] =  tParamsMinMax_LamKchP[0][3];
    tReturnVec[12] = tParamsMinMax_LamKchP[0][4];

  }
  else if(aAnType==kLamKchM)
  {
    tReturnVec[7] = tParamsMinMax_LamKchM[0][2];
    tReturnVec[10] = tParamsMinMax_LamKchM[0][3];
    tReturnVec[13] = tParamsMinMax_LamKchM[0][4];
  }
  else if(aAnType==kLamK0)
  {
    tReturnVec[8] = tParamsMinMax_LamK0[0][2];
    tReturnVec[11] = tParamsMinMax_LamK0[0][3];
    tReturnVec[14] = tParamsMinMax_LamK0[0][4];  
  }
  else assert(0);
  

  
  for(unsigned int i=15; i<tMinParams.size(); i++)
  {
    tReturnVec[i] = vector<double>{tMinParams[i]};
  }
  return tReturnVec;
}



//________________________________________________________________________________________________________________
td2dVec TripleFitGenerator::FormAllPossibleCombos(td2dVec &aVec)
{
  td2dVec tReturnVec(0);

  vector<vector<double>::iterator> it;
  for(unsigned int i=0; i<aVec.size(); i++)
  {
    it.push_back(aVec[i].begin());
  }
  unsigned int K = aVec.size();
  while (it[0] != aVec[0].end()) {
  // process the pointed-to elements

  td1dVec tNewCombo(0);
  for(unsigned int i=0; i<aVec.size(); i++) 
  {
    tNewCombo.push_back(*it[i]);
  }
  tReturnVec.push_back(tNewCombo);
  
  
  // the following increments the "odometer" by 1
  ++it[K-1];
  for (int i = K-1; (i > 0) && (it[i] == aVec[i].end()); --i) {
    it[i] = aVec[i].begin();
    ++it[i-1];
    }
  }
  return tReturnVec;
}


//________________________________________________________________________________________________________________
void TripleFitGenerator::BuildFitErrBands_FINALRESULTSONLY(AnalysisType aAnType)
{
  cout << "aAnType = "  << aAnType << endl;
  td2dVec tParamsMinMax = GetParamsMinMaxForErrBands_FINALRESULTSONLY(aAnType);
  
  for(unsigned int i=0; i<tParamsMinMax.size(); i++)
  {
    cout << endl << endl << "i = " << i << endl;
    for(unsigned int j=0; j<tParamsMinMax[i].size(); j++) cout << tParamsMinMax[i][j] << endl;
  }
  
  td2dVec tAllParamsCombos = FormAllPossibleCombos(tParamsMinMax);
  cout << "tAllParamsCombos.size() = " << tAllParamsCombos.size() << endl;
  
  int tNpar=51;
  double tChi2 = 0.;
  
  for(unsigned int i=0; i<tAllParamsCombos.size(); i++) 
  {
    assert(tAllParamsCombos[i].size()==tNpar);
    fMasterLednickyFitter->CalculateFitFunction_forErrBands(tNpar, tChi2, tAllParamsCombos[i].data());
  }
}


//________________________________________________________________________________________________________________
void TripleFitGenerator::BuildFitErrBands_FINALRESULTSONLY()
{
  BuildFitErrBands_FINALRESULTSONLY(kLamKchP);
  BuildFitErrBands_FINALRESULTSONLY(kLamKchM);
  BuildFitErrBands_FINALRESULTSONLY(kLamK0);
}


