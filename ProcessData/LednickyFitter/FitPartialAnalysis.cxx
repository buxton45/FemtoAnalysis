///////////////////////////////////////////////////////////////////////////
// FitPartialAnalysis:                                                   //
///////////////////////////////////////////////////////////////////////////

#include "FitPartialAnalysis.h"

#ifdef __ROOT__
ClassImp(FitPartialAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType) : 
  fFileLocation(aFileLocation),
  fFileLocationMC(0),
  fAnalysisName(aAnalysisName),
  fDirectoryName(0),

  fAnalysisType(aAnalysisType),
  fBFieldType(aBFieldType),
  fCentralityType(aCentralityType),
  fFitPartialAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfLite(0),
  fKStarCf(0),
  fKStarCfNum(0),
  fKStarCfDen(0),

  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),

  fKStarNumScale(0),
  fKStarDenScale(0),

  fNFitParams(5),  //should be initialized here to the correct number of parameters, excluding fNorm
  fLambda(0),
  fRadius(0),
  fRef0(0),
  fImf0(0),
  fd0(0),
  fRef02(0),
  fImf02(0),
  fd02(0),
  fFitParameters(fNFitParams),
  fNorm(0),

  fRejectOmega(false),

  fModelKStarTrueVsRecMixed(0),
  fModelKStarCfFake(0),
  fModelKStarCfFakeIdeal(0)


{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  BuildKStarCf(fKStarMinNorm,fKStarMaxNorm);

  SetParticleTypes();

  //-----Initiate parameters and load into vector
  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    fNFitParams = 8;
    fFitParameters.resize(fNFitParams);

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd0->SetStartValue(0.);
//    fd0->SetFixed(true);

    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
//    fNorm->SetFixed(true);

//TODO give these their own unique start values
    fRef02 = new FitParameter(kRef02, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf02 = new FitParameter(kImf02, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd02 = new FitParameter(kd02, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd02->SetStartValue(0.);

/*
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, 1.46, true);
    fImf0 = new FitParameter(kImf0, 0.24, true);
    fd0 = new FitParameter(kd0, 0., true);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
    fNorm->SetFixed(true);
*/

    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
    fFitParameters[kRef02] = fRef02;
    fFitParameters[kImf02] = fImf02;
    fFitParameters[kd02] = fd02;

  }
  else
  {
/*
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda]);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius]);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0]);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0]);
    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0]);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm]);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
    //fNorm->SetFixed(true);
*/

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.5);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,10.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,10.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,10.);
    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,50.);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm]);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);


    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
}

  if( (fAnalysisType == kLamKchM) || (fAnalysisType == kALamKchP) /*|| (fAnalysisType == kLamKchMwConjugate)*/ ) {fRejectOmega = true;}

}


//________________________________________________________________________________________________________________
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aFileLocationMC, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType) : 
  fFileLocation(aFileLocation),
  fFileLocationMC(aFileLocationMC),
  fAnalysisName(aAnalysisName),
  fDirectoryName(0),

  fAnalysisType(aAnalysisType),
  fBFieldType(aBFieldType),
  fCentralityType(aCentralityType),
  fFitPartialAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfLite(0),
  fKStarCf(0),
  fKStarCfNum(0),
  fKStarCfDen(0),

  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),

  fKStarNumScale(0),
  fKStarDenScale(0),

  fNFitParams(5),  //should be initialized here to the correct number of parameters, excluding fNorm
  fLambda(0),
  fRadius(0),
  fRef0(0),
  fImf0(0),
  fd0(0),
  fRef02(0),
  fImf02(0),
  fd02(0),
  fFitParameters(fNFitParams),
  fNorm(0),

  fRejectOmega(false),

  fModelKStarTrueVsRecMixed(0),
  fModelKStarCfFake(0),
  fModelKStarCfFakeIdeal(0)


{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  BuildKStarCf(fKStarMinNorm,fKStarMaxNorm);

  SetParticleTypes();

  //-----Initiate parameters and load into vector
  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    fNFitParams = 8;
    fFitParameters.resize(fNFitParams);

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd0->SetStartValue(0.);
//    fd0->SetFixed(true);

    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
    fNorm->SetFixed(true);

//TODO give these their own unique start values
    fRef02 = new FitParameter(kRef02, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf02 = new FitParameter(kImf02, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd02 = new FitParameter(kd02, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd02->SetStartValue(0.);

/*
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, 1.46, true);
    fImf0 = new FitParameter(kImf0, 0.24, true);
    fd0 = new FitParameter(kd0, 0., true);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
    fNorm->SetFixed(true);
*/

    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
    fFitParameters[kRef02] = fRef02;
    fFitParameters[kImf02] = fImf02;
    fFitParameters[kd02] = fd02;

  }
  else
  {
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda]);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius]);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0]);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0]);
    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0]);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm]);
    fNorm->SetStartValue(fKStarNumScale/fKStarDenScale);
    //fNorm->SetFixed(true);


    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
}

  if( (fAnalysisType == kLamKchM) || (fAnalysisType == kALamKchP) /*|| (fAnalysisType == kLamKchMwConjugate)*/ ) {fRejectOmega = true;}

//---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***
  TObjArray* tDirMC = ConnectAnalysisDirectory(fFileLocationMC,fDirectoryName);
  TString tTempName = cModelKStarTrueVsRecMixedBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tTempNameNew = tTempName + TString(cCentralityTags[fCentralityType]);
  TH2* tTempHisto = (TH2*)tDirMC->FindObject(tTempName);
  //-----make sure tHisto is retrieved
  if(!tTempHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << tTempName << endl;}
  assert(tTempHisto);
  //----------------------------------

  fModelKStarTrueVsRecMixed = (TH2*)tTempHisto->Clone(tTempNameNew);
  fModelKStarTrueVsRecMixed->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!fModelKStarTrueVsRecMixed->GetSumw2N()) {fModelKStarTrueVsRecMixed->Sumw2();}

  delete tTempHisto;

  tDirMC->Delete();
  delete tDirMC;
//---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***


}


//________________________________________________________________________________________________________________
FitPartialAnalysis::~FitPartialAnalysis()
{
  cout << "FitPartialAnalysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
TObjArray* FitPartialAnalysis::ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TList *tFemtolist = (TList*)tFile->Get("femtolist");
    tFemtolist->SetOwner();

  TObjArray *ReturnArray = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    ReturnArray->SetOwner();


  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtolist);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tFemtolist->Delete();
  delete tFemtolist;

  tFile->Close();
  delete tFile;


  return ReturnArray;
}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetParticleTypes()
{
  if(fAnalysisType == kLamK0) {fParticleTypes[0] = kLam; fParticleTypes[1] = kK0;}
  else if(fAnalysisType == kALamK0) {fParticleTypes[0] = kALam; fParticleTypes[1] = kK0;}

  else if(fAnalysisType == kLamKchP) {fParticleTypes[0] = kLam; fParticleTypes[1] = kKchP;}
  else if(fAnalysisType == kALamKchP) {fParticleTypes[0] = kALam; fParticleTypes[1] = kKchP;}
  else if(fAnalysisType == kLamKchM) {fParticleTypes[0] = kLam; fParticleTypes[1] = kKchM;}
  else if(fAnalysisType == kALamKchM) {fParticleTypes[0] = kALam; fParticleTypes[1] = kKchM;}

  else if(fAnalysisType == kXiKchP) {fParticleTypes[0] = kXi; fParticleTypes[1] = kKchP;}
  else if(fAnalysisType == kAXiKchP) {fParticleTypes[0] = kAXi; fParticleTypes[1] = kKchP;}
  else if(fAnalysisType == kXiKchM) {fParticleTypes[0] = kXi; fParticleTypes[1] = kKchM;}
  else if(fAnalysisType == kAXiKchM) {fParticleTypes[0] = kAXi; fParticleTypes[1] = kKchM;}

  else if(fAnalysisType == kLamLam) {fParticleTypes[0] = kLam; fParticleTypes[1] = kLam;}
  else if(fAnalysisType == kALamALam) {fParticleTypes[0] = kALam; fParticleTypes[1] = kALam;}
  else if(fAnalysisType == kLamALam) {fParticleTypes[0] = kLam; fParticleTypes[1] = kALam;}

  else if(fAnalysisType == kLamPiP) {fParticleTypes[0] = kLam; fParticleTypes[1] = kPiP;}
  else if(fAnalysisType == kALamPiP) {fParticleTypes[0] = kALam; fParticleTypes[1] = kPiP;}
  else if(fAnalysisType == kLamPiM) {fParticleTypes[0] = kLam; fParticleTypes[1] = kPiM;}
  else if(fAnalysisType == kALamPiM) {fParticleTypes[0] = kALam; fParticleTypes[1] = kPiM;}

  else{ cout << "ERROR IN FitPartialAnalysis::SetParticleTypes:  Invalid fAnalysisType!!!!!!!" << endl << endl;}
}


//________________________________________________________________________________________________________________
TH1* FitPartialAnalysis::Get1dHisto(TString aHistoName, TString aNewName)
{
  TObjArray* tDir = ConnectAnalysisDirectory(fFileLocation,fDirectoryName);
    tDir->SetOwner();

  TH1 *tHisto = (TH1*)tDir->FindObject(aHistoName);
    tHisto->SetDirectory(0);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1 *ReturnHisto = (TH1*)tHisto->Clone(aNewName);
    ReturnHisto->SetDirectory(0);

  delete tHisto;

  tDir->Delete();
  delete tDir;

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH1*)ReturnHisto;
}

//________________________________________________________________________________________________________________
TH1* FitPartialAnalysis::Get1dHisto(TString aFileLocation, TString aHistoName, TString aNewName)
{
  TObjArray* tDir = ConnectAnalysisDirectory(aFileLocation,fDirectoryName);
    tDir->SetOwner();

  TH1 *tHisto = (TH1*)tDir->FindObject(aHistoName);
    tHisto->SetDirectory(0);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1 *ReturnHisto = (TH1*)tHisto->Clone(aNewName);
    ReturnHisto->SetDirectory(0);

  delete tHisto;

  tDir->Delete();
  delete tDir;

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH1*)ReturnHisto;
}




//________________________________________________________________________________________________________________
TH2* FitPartialAnalysis::Get2dHisto(TString aHistoName, TString aNewName)
{
  TObjArray* tDir = ConnectAnalysisDirectory(fFileLocation,fDirectoryName);

  TH2 *tHisto = (TH2*)tDir->FindObject(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH2 *ReturnHisto = (TH2*)tHisto->Clone(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH2*)ReturnHisto;
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::BuildKStarCf(double aMinNorm, double aMaxNorm)
{
  fKStarMinNorm = aMinNorm;
  fKStarMaxNorm = aMaxNorm;

  TString tNumName = cKStarCfBaseTagNum + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cKStarCfBaseTagDen + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "KStarCf_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  fKStarCfLite = new CfLite(tCfName,tCfName,tNum,tDen,fKStarMinNorm,fKStarMaxNorm);

  fKStarCf = fKStarCfLite->Cf();
  fKStarCfNum = fKStarCfLite->Num();
  fKStarCfDen = fKStarCfLite->Den();

  fKStarNumScale = fKStarCfLite->GetNumScale();
  fKStarDenScale = fKStarCfLite->GetDenScale();

}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::RebinKStarCf(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fKStarMinNorm = aMinNorm;
  fKStarMaxNorm = aMaxNorm;

  fKStarCfLite->Rebin(aRebinFactor,fKStarMinNorm,fKStarMaxNorm);

  fKStarCf = fKStarCfLite->Cf();
  fKStarCfNum = fKStarCfLite->Num();
  fKStarCfDen = fKStarCfLite->Den();

  fKStarNumScale = fKStarCfLite->GetNumScale();
  fKStarDenScale = fKStarCfLite->GetDenScale();

}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetFitParameter(FitParameter* aParam)
{
  //Created a shallow copy, which I think is what I want
  fFitParameters[aParam->GetType()] = aParam;
}

//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCfFake(double aMinNorm, double aMaxNorm, int aRebin)
{
  TString tNumName = cModelKStarCfNumFakeBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFake_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  fModelKStarCfFake = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
  if(aRebin != 1) fModelKStarCfFake->Rebin(aRebin);

  return fModelKStarCfFake;
}


//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCfFakeIdeal(double aMinNorm, double aMaxNorm, int aRebin)
{
  TString tNumName = cModelKStarCfNumFakeIdealBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenIdealBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFakeIdeal_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  fModelKStarCfFakeIdeal = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
  if(aRebin != 1) fModelKStarCfFakeIdeal->Rebin(aRebin);

  return fModelKStarCfFakeIdeal;
}
