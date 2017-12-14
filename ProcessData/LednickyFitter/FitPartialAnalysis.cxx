///////////////////////////////////////////////////////////////////////////
// FitPartialAnalysis:                                                   //
///////////////////////////////////////////////////////////////////////////

#include "FitPartialAnalysis.h"

#ifdef __ROOT__
ClassImp(FitPartialAnalysis)
#endif


//GLOBAL!!!!!!!!!!!!!!!
BackgroundFitter *GlobalBgdFitter = NULL;

//______________________________________________________________________________
void GlobalBgdFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalBgdFitter->CalculateBgdFitFunction(npar,f,par);
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType, TString aDirNameModifier, bool aIncludeSingletAndTriplet) : 
  fAnalysisRunType(aRunType),
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

  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),

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
  fModelKStarCfFakeIdeal(0),

  fNonFlatBackground(0),
  fCorrectedFitVec(0)


{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  if(!aDirNameModifier.IsNull()) fDirectoryName += aDirNameModifier;

  BuildKStarCf(fKStarMinNorm,fKStarMaxNorm);

  SetParticleTypes();

  //-----Initiate parameters and load into vector
  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    if(aIncludeSingletAndTriplet) fNFitParams = 8;
    fFitParameters.resize(fNFitParams);

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd0->SetStartValue(0.);
//    fd0->SetFixed(true);

    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
//    fNorm->SetFixed(true);

//TODO give these their own unique start values
    if(aIncludeSingletAndTriplet)
    {
      fRef02 = new FitParameter(kRef02, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
      fImf02 = new FitParameter(kImf02, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

      fd02 = new FitParameter(kd02, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
      fd02->SetStartValue(0.);
    }

/*
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, 1.46, true);
    fImf0 = new FitParameter(kImf0, 0.24, true);
    fd0 = new FitParameter(kd0, 0., true);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
    fNorm->SetFixed(true);
*/

    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
    if(aIncludeSingletAndTriplet)
    {
      fFitParameters[kRef02] = fRef02;
      fFitParameters[kImf02] = fImf02;
      fFitParameters[kd02] = fd02;
    }

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
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
    //fNorm->SetFixed(true);
*/

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.5);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,10.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,10.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,10.);
    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,50.);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm]);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());


    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
}

  if( (fAnalysisType == kLamKchM) || (fAnalysisType == kALamKchP) /*|| (fAnalysisType == kLamKchMwConjugate)*/ ) {fRejectOmega = true;}

}


//________________________________________________________________________________________________________________
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aFileLocationMC, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType, TString aDirNameModifier, bool aIncludeSingletAndTriplet) : 
  fAnalysisRunType(aRunType),
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

  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),

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
  fModelKStarCfFakeIdeal(0),

  fNonFlatBackground(0),
  fCorrectedFitVec(0)


{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  if(!aDirNameModifier.IsNull()) fDirectoryName += aDirNameModifier;

  BuildKStarCf(fKStarMinNorm,fKStarMaxNorm);

  SetParticleTypes();

  //-----Initiate parameters and load into vector
  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    if(aIncludeSingletAndTriplet) fNFitParams = 8;
    fFitParameters.resize(fNFitParams);

    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
    fd0->SetStartValue(0.);
//    fd0->SetFixed(true);

    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
//    fNorm->SetFixed(true);

//TODO give these their own unique start values
    if(aIncludeSingletAndTriplet)
    {
      fRef02 = new FitParameter(kRef02, cStartValues[fAnalysisType][fCentralityType][kRef0],false,0.,0.,1.);
      fImf02 = new FitParameter(kImf02, cStartValues[fAnalysisType][fCentralityType][kImf0],false,0.,0.,1.);

      fd02 = new FitParameter(kd02, cStartValues[fAnalysisType][fCentralityType][kd0],false,0.,0.,5.);
      fd02->SetStartValue(0.);
    }
/*
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda],false,0.,0.,0.4);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius],false,0.,0.,5.);
    fRef0 = new FitParameter(kRef0, 1.46, true);
    fImf0 = new FitParameter(kImf0, 0.24, true);
    fd0 = new FitParameter(kd0, 0., true);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm],false,0.,0.,0.1);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
    fNorm->SetFixed(true);
*/

    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
    if(aIncludeSingletAndTriplet)
    {
      fFitParameters[kRef02] = fRef02;
      fFitParameters[kImf02] = fImf02;
      fFitParameters[kd02] = fd02;
    }
  }
  else
  {
    fLambda = new FitParameter(kLambda, cStartValues[fAnalysisType][fCentralityType][kLambda]/*,false,0.,0.,0.4*/);
    fRadius = new FitParameter(kRadius, cStartValues[fAnalysisType][fCentralityType][kRadius]/*,false,0.,0.,5.*/);
    fRef0 = new FitParameter(kRef0, cStartValues[fAnalysisType][fCentralityType][kRef0]/*,false,0.,0.,1.*/);
    fImf0 = new FitParameter(kImf0, cStartValues[fAnalysisType][fCentralityType][kImf0]/*,false,0.,0.,1.*/);
    fd0 = new FitParameter(kd0, cStartValues[fAnalysisType][fCentralityType][kd0]/*,false,0.,0.,5.*/);
    fNorm = new FitParameter(kNorm, cStartValues[fAnalysisType][fCentralityType][kNorm]);
    fNorm->SetStartValue(fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale());
    //fNorm->SetFixed(true);


    fFitParameters[kLambda] = fLambda;
    fFitParameters[kRadius] = fRadius;
    fFitParameters[kRef0] = fRef0;
    fFitParameters[kImf0] = fImf0;
    fFitParameters[kd0] = fd0;
}

  if( (fAnalysisType == kLamKchM) || (fAnalysisType == kALamKchP) /*|| (fAnalysisType == kLamKchMwConjugate)*/ ) {fRejectOmega = true;}

//---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***---***
//  I only have MC data for 0-10% centrality.  So other centralities must share the data
  TString tDirectoryNameMC = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[k0010]);
  if(!aDirNameModifier.IsNull()) tDirectoryNameMC += aDirNameModifier;

  TString tTempName = cModelKStarTrueVsRecMixedBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tTempNameNew = tTempName + TString(cCentralityTags[k0010]);
  TH2* tTempHisto = Get2dHisto(fFileLocationMC,tDirectoryNameMC,tTempName,tTempNameNew);

  //-----make sure tHisto is retrieved
  if(!tTempHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << tTempName << endl;}
  assert(tTempHisto);
  //----------------------------------

  fModelKStarTrueVsRecMixed = (TH2*)tTempHisto->Clone(tTempNameNew);
  fModelKStarTrueVsRecMixed->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!fModelKStarTrueVsRecMixed->GetSumw2N()) {fModelKStarTrueVsRecMixed->Sumw2();}

  delete tTempHisto;
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
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys)
  {
    tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
    if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
    else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
    else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
    else
    {
      cout << "ERROR in FitPartialAnalysis::ConnectAnalysisDirectory!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << "Invalid aDirectoryName for fAnalysisRunType=kTrain||kTrainSys:  aDirectoryName = " << aDirectoryName << endl;
      assert(0);
    }

    if(fAnalysisRunType==kTrainSys) tFemtoListName += TString("_Systematics");
    tFemtoListName += TString("_femtolist");
    tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
    aDirectoryName.ReplaceAll("0010","010");
  }
  else
  {
    tFemtoListName = "femtolist";
    tFemtolist = (TList*)tFile->Get(tFemtoListName);
  }

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

  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) 
  {
    tDirFile->Close();
    delete tDirFile;
  }

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
TH2* FitPartialAnalysis::Get2dHisto(TString aFileLocation, TString aDirectoryName, TString aHistoName, TString aNewName)
{
  TObjArray* tDir = ConnectAnalysisDirectory(aFileLocation,aDirectoryName);

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
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]) + TString(cBFieldTags[fBFieldType]);
  fKStarCfLite = new CfLite(tCfName,tCfName,tNum,tDen,fKStarMinNorm,fKStarMaxNorm);
}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::RebinKStarCf(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fKStarMinNorm = aMinNorm;
  fKStarMaxNorm = aMaxNorm;

  fKStarCfLite->Rebin(aRebinFactor,fKStarMinNorm,fKStarMaxNorm);
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::FitNonFlatBackground(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType, 
                                              double aMinBgdFit, double aMaxBgdFit, double aKStarMinNorm, double aKStarMaxNorm)
{
  BackgroundFitter* tBgdFitter = new BackgroundFitter(aNum, aDen, aCf, aBgdFitType, aFitType, aMinBgdFit, aMaxBgdFit, aKStarMinNorm, aKStarMaxNorm);
  tBgdFitter->GetMinuitObject()->SetFCN(GlobalBgdFCN);
  GlobalBgdFitter = tBgdFitter;

  TF1* tNonFlatBackground = tBgdFitter->FitNonFlatBackground();

  delete tBgdFitter;
  return tNonFlatBackground;
}

//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::FitNonFlatBackground(TH1* aCf, NonFlatBgdFitType aBgdFitType, 
                                              double aMinBgdFit, double aMaxBgdFit, double aKStarMinNorm, double aKStarMaxNorm)
{
  TH1* tDummyNum=nullptr;
  TH1* tDummyDen=nullptr;

  return FitNonFlatBackground(tDummyNum, tDummyDen, aCf, aBgdFitType, kChi2, aMinBgdFit, aMaxBgdFit, aKStarMinNorm, aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::GetNonFlatBackground(NonFlatBgdFitType aBgdFitType, FitType aFitType)
{
  if(fNonFlatBackground) return fNonFlatBackground;

  if(aFitType==kChi2PML)
  {
    fNonFlatBackground = FitNonFlatBackground(fKStarCfLite->Num(), fKStarCfLite->Den(), fKStarCfLite->Cf(), aBgdFitType, aFitType, fMinBgdFit, fMaxBgdFit, fKStarMinNorm, fKStarMaxNorm);
  }
  else if(aFitType==kChi2)
  {
    fNonFlatBackground = FitNonFlatBackground(fKStarCfLite->Cf(), aBgdFitType, fMinBgdFit, fMaxBgdFit, fKStarMinNorm, fKStarMaxNorm);
  }
  else assert(0);

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetFitParameter(FitParameter* aParam)
{
  //Created a shallow copy, which I think is what I want
  fFitParameters[aParam->GetType()] = aParam;
}

//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCf(double aMinNorm, double aMaxNorm, int aRebin)
{
  TString tNumName = cKStarCfBaseTagNum + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName = cKStarCfBaseTagDen + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCf_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  CfLite* tReturnCfLite = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
  if(aRebin != 1) tReturnCfLite->Rebin(aRebin);

  return tReturnCfLite;
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
