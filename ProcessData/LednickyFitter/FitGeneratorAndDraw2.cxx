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
  fMasterMinuit(nullptr),
  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchP, aCentralityTypes, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchM, aCentralityTypes, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);

  fMasterMinuit = new TMinuit(50);
}


//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::FitGeneratorAndDraw2(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
  fFitGen1(nullptr),
  fFitGen2(nullptr),
  fMasterMinuit(nullptr),
  fMasterLednickyFitter(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fFitGen1 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchP, aCentralityType, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);
  fFitGen2 = new FitGeneratorAndDraw(aFileLocationBase, aFileLocationBaseMC, kLamKchM, aCentralityType, aRunType, aNPartialAnalysis, kPairwConj, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);

  fMasterMinuit = new TMinuit(50);
}

//________________________________________________________________________________________________________________
FitGeneratorAndDraw2::FitGeneratorAndDraw2(FitGeneratorAndDraw* aFitGen1, FitGeneratorAndDraw* aFitGen2) :
  fFitGen1(aFitGen1),
  fFitGen2(aFitGen2),
  fMasterMinuit(nullptr),
  fMasterSharedAn(nullptr),

  fNMinuitParams(0),
  fMinuitMinParams(0),
  fMinuitParErrors(0),

  fMasterMinuitFitParametersMatrix(0)
{
  fMasterMinuit = new TMinuit(50);
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

  fFitGen1->GetSharedAn()->CreateMinuitParametersMatrix();
  fFitGen2->GetSharedAn()->CreateMinuitParametersMatrix();

  vector<vector<FitParameter*> > tMinuitParamMatrix1 = fFitGen1->GetSharedAn()->GetMinuitFitParametersMatrix();
  vector<vector<FitParameter*> > tMinuitParamMatrix2 = fFitGen2->GetSharedAn()->GetMinuitFitParametersMatrix();

  assert(tMinuitParamMatrix1.size()==tMinuitParamMatrix2.size());
  fMasterMinuitFitParametersMatrix.clear();
  vector<FitParameter*> tTempParamVec;
  for(unsigned int iPar=0; iPar<tMinuitParamMatrix1.size(); iPar++)
  {
    tTempParamVec.clear();
    for(unsigned int i=0; i<tMinuitParamMatrix1[iPar].size(); i++) tTempParamVec.push_back(tMinuitParamMatrix1[iPar][i]);



    if(tMinuitParamMatrix1[iPar][0]->GetType()==kLambda)
    {
      if(aShareLambda)
      {
        for(unsigned int iLam=1; iLam<tMinuitParamMatrix1[iPar].size(); iLam++)
        {
          tMinuitParamMatrix2[iPar][iLam]=tMinuitParamMatrix1[iPar][iLam];
        }
      }
      else
      {
        for(unsigned int i=0; i<tMinuitParamMatrix2[iPar].size(); i++) tTempParamVec.push_back(tMinuitParamMatrix2[iPar][i]);
      }

    }
    else if(tMinuitParamMatrix1[iPar][0]->GetType()==kRadius)
    {
      if(aShareRadii)
      {
        for(unsigned int iR=1; iR<tMinuitParamMatrix1[iPar].size(); iR++)
        {
          tMinuitParamMatrix2[iPar][iR]=tMinuitParamMatrix1[iPar][iR];
        }
      }
      else
      {
        for(unsigned int i=0; i<tMinuitParamMatrix2[iPar].size(); i++) tTempParamVec.push_back(tMinuitParamMatrix2[iPar][i]);
      }
    }
    else for(unsigned int i=0; i<tMinuitParamMatrix2[iPar].size(); i++) tTempParamVec.push_back(tMinuitParamMatrix2[iPar][i]);

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
      FitSharedAnalyses::CreateMinuitParameter(fMasterMinuit, fNMinuitParams,tempVec[itemp]);
      fNMinuitParams++;
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

      FitSharedAnalyses::CreateMinuitParameter(fMasterMinuit, fNMinuitParams, fMasterSharedAn->GetFitPairAnalysis(iAnaly)->GetFitNormParameter(iPartAn));
      fNMinuitParams++;
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
  fMasterLednickyFitter->SetNonFlatBgdFitType(kPolynomial);
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







