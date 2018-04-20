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
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType, TString aDirNameModifier, bool aUseNumRotPar2InsteadOfDen, bool aIncludeSingletAndTriplet) : 
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

  fKStarCfLite(nullptr),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),
  fMaxBgdBuild(2.0),
  fNormalizeBgdFitToCf(false),

  fNFitParams(5),  //should be initialized here to the correct number of parameters, excluding fNorm
  fLambda(nullptr),
  fRadius(nullptr),
  fRef0(nullptr),
  fImf0(nullptr),
  fd0(nullptr),
  fRef02(nullptr),
  fImf02(nullptr),
  fd02(nullptr),
  fFitParameters(fNFitParams),
  fNorm(nullptr),
  fBgdParameters(0),

  fRejectOmega(false),

  fModelKStarTrueVsRecMixed(nullptr),
  fModelKStarCfFake(nullptr),
  fModelKStarCfFakeIdeal(nullptr),

  fPrimaryFit(nullptr),
  fNonFlatBackground(nullptr),
  fThermNonFlatBgd(nullptr),
  fCorrectedFitVec(0),

  fUseNumRotPar2InsteadOfDen(aUseNumRotPar2InsteadOfDen)
{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  if(!aDirNameModifier.IsNull()) fDirectoryName += aDirNameModifier;

  double tKStarMinNorm = 0.32, tKStarMaxNorm = 0.40;
  BuildKStarCf(tKStarMinNorm, tKStarMaxNorm);

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

  //-----Set owner information
  for(unsigned int i=0; i<fFitParameters.size(); i++) fFitParameters[i]->SetOwnerInfo(fAnalysisType, fCentralityType, fBFieldType);
  fNorm->SetOwnerInfo(fAnalysisType, fCentralityType, fBFieldType);
}


//________________________________________________________________________________________________________________
FitPartialAnalysis::FitPartialAnalysis(TString aFileLocation, TString aFileLocationMC, TString aAnalysisName, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, AnalysisRunType aRunType, TString aDirNameModifier, bool aUseNumRotPar2InsteadOfDen, bool aIncludeSingletAndTriplet) : 
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

  fKStarCfLite(nullptr),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),
  fMaxBgdBuild(2.0),
  fNormalizeBgdFitToCf(false),

  fNFitParams(5),  //should be initialized here to the correct number of parameters, excluding fNorm
  fLambda(nullptr),
  fRadius(nullptr),
  fRef0(nullptr),
  fImf0(nullptr),
  fd0(nullptr),
  fRef02(nullptr),
  fImf02(nullptr),
  fd02(nullptr),
  fFitParameters(fNFitParams),
  fNorm(nullptr),
  fBgdParameters(0),

  fRejectOmega(false),

  fModelKStarTrueVsRecMixed(nullptr),
  fModelKStarCfFake(nullptr),
  fModelKStarCfFakeIdeal(nullptr),

  fPrimaryFit(nullptr),
  fNonFlatBackground(nullptr),
  fThermNonFlatBgd(nullptr),
  fCorrectedFitVec(0),

  fUseNumRotPar2InsteadOfDen(aUseNumRotPar2InsteadOfDen)
{

  fDirectoryName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  if(!aDirNameModifier.IsNull()) fDirectoryName += aDirNameModifier;

  double tKStarMinNorm = 0.32, tKStarMaxNorm = 0.40;
  BuildKStarCf(tKStarMinNorm, tKStarMaxNorm);

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

  //-----Set owner information
  for(unsigned int i=0; i<fFitParameters.size(); i++) fFitParameters[i]->SetOwnerInfo(fAnalysisType, fCentralityType, fBFieldType);
  fNorm->SetOwnerInfo(fAnalysisType, fCentralityType, fBFieldType);
}


//________________________________________________________________________________________________________________
FitPartialAnalysis::~FitPartialAnalysis()
{
  cout << "FitPartialAnalysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}

//________________________________________________________________________
double FitPartialAnalysis::GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double FitPartialAnalysis::GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double FitPartialAnalysis::LednickyEq(double *x, double *par)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first

  std::complex<double> f0 (par[2],par[3]);
  double Alpha = 0.; // alpha = 0 for non-identical
  double z = 2.*(x[0]/hbarc)*par[1];  //z = 2k*R, to be fed to GetLednickyF1(2)

  double C_QuantumStat = Alpha*exp(-z*z);  // will be zero for my analysis

  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*par[4]*(x[0]/hbarc)*(x[0]/hbarc) - ImI*(x[0]/hbarc),-1);

  double C_FSI = (1+Alpha)*( 0.5*norm(ScattAmp)/(par[1]*par[1])*(1.-1./(2*sqrt(TMath::Pi()))*(par[4]/par[1])) + 2.*real(ScattAmp)/(par[1]*sqrt(TMath::Pi()))*GetLednickyF1(z) - (imag(ScattAmp)/par[1])*GetLednickyF2(z));

  double Cf = 1. + par[0]*(C_QuantumStat + C_FSI);
  //Cf *= par[5];

  return Cf;
}

//________________________________________________________________________
double FitPartialAnalysis::LednickyEqWithNorm(double *x, double *par)
{

  double tUnNormCf = LednickyEq(x, par);
  double tNormCf = par[5]*tUnNormCf;
  return tNormCf;
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
void FitPartialAnalysis::BuildKStarCf(double aKStarMinNorm, double aKStarMaxNorm)
{
  TString tNumName = cKStarCfBaseTagNum + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName;
  if(!fUseNumRotPar2InsteadOfDen) tDenName = cKStarCfBaseTagDen;
  else tDenName = cKStarCfBaseTagNumRotatePar2;
  tDenName += TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "KStarCf_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]) + TString(cBFieldTags[fBFieldType]);
  fKStarCfLite = new CfLite(tCfName, tCfName, tNum, tDen, aKStarMinNorm, aKStarMaxNorm);
}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::RebinKStarCf(int aRebinFactor, double aKStarMinNorm, double aKStarMaxNorm)
{
  fKStarCfLite->Rebin(aRebinFactor, aKStarMinNorm, aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::CreateFitFunction(bool aApplyNorm, IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, double aChi2, int aNDF, 
                                           double aKStarMin, double aKStarMax, TString aBaseName)
{
  TString tName = TString::Format("%s_%s%s%s", aBaseName.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType], cBFieldTags[fBFieldType]);
  if(aApplyNorm) tName += TString("_WithNorm");
  assert(fFitParameters.size()==5);  //unless I go back to singlet and triplet, this should be equal to 5 (not counting normalization parameter)
  assert(fFitParameters.size()==fNFitParams);

  fPrimaryFit = new TF1(tName, LednickyEqWithNorm, aKStarMin, aKStarMax, fNFitParams+1);
  double tParamValue, tParamError;
  for(int iPar=0; iPar<fNFitParams; iPar++)
  {
    ParameterType tParamType = fFitParameters[iPar]->GetType();
    tParamValue = fFitParameters[iPar]->GetFitValue();
    tParamError = fFitParameters[iPar]->GetFitValueError();
    if(tParamType==kLambda && aIncResType != kIncludeNoResiduals)
    {
      tParamValue *= cAnalysisLambdaFactorsArr[aIncResType][aResPrimMaxDecayType][fAnalysisType];
      tParamError *= cAnalysisLambdaFactorsArr[aIncResType][aResPrimMaxDecayType][fAnalysisType];
    }
    fPrimaryFit->SetParameter(iPar,tParamValue);
    fPrimaryFit->SetParError(iPar,tParamError);
  }

  if(aApplyNorm)
  {
    fPrimaryFit->SetParameter(5, fNorm->GetFitValue());
    fPrimaryFit->SetParError(5, fNorm->GetFitValueError());
  }
  else
  {
    fPrimaryFit->SetParameter(5, 1.);
    fPrimaryFit->SetParError(5, 0.);
  }

  fPrimaryFit->SetChisquare(aChi2);
  fPrimaryFit->SetNDF(aNDF);

  fPrimaryFit->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");
//  fKStarCfLite->Cf()->GetListOfFunctions()->Add(fPrimaryFit);
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::FitNonFlatBackground(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf, 
                                              double aMinBgdFit, double aMaxBgdFit, double aMaxBgdBuild, double aKStarMinNorm, double aKStarMaxNorm)
{
  BackgroundFitter* tBgdFitter = new BackgroundFitter(aNum, aDen, aCf, aBgdFitType, aFitType, aNormalizeFitToCf, aMinBgdFit, aMaxBgdFit, aMaxBgdBuild, aKStarMinNorm, aKStarMaxNorm);
  tBgdFitter->GetMinuitObject()->SetFCN(GlobalBgdFCN);
  GlobalBgdFitter = tBgdFitter;

  TF1* tNonFlatBackground = tBgdFitter->FitNonFlatBackground();

  delete tBgdFitter;
  return tNonFlatBackground;
}

//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::FitNonFlatBackground(TH1* aCf, NonFlatBgdFitType aBgdFitType, 
                                              double aMinBgdFit, double aMaxBgdFit, double aMaxBgdBuild, double aKStarMinNorm, double aKStarMaxNorm)
{
  TH1* tDummyNum=nullptr;
  TH1* tDummyDen=nullptr;

  return FitNonFlatBackground(tDummyNum, tDummyDen, aCf, aBgdFitType, kChi2, false, aMinBgdFit, aMaxBgdFit, aMaxBgdBuild, aKStarMinNorm, aKStarMaxNorm);
}

//________________________________________________________________________________________________________________
TH1* FitPartialAnalysis::GetThermNonFlatBackground(bool aCombineConj, bool aCombineLamKchPM, ThermEventsType aThermEventsType)
{
  if(fThermNonFlatBgd) return fThermNonFlatBgd->GetThermCf();

  //------------------------------------
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  TString tCfDescriptor = "Full";
//  TString tCfDescriptor = "PrimaryOnly";

  int aRebin = 1;

  fThermNonFlatBgd = new ThermCf(tFileName, tCfDescriptor, fAnalysisType, fCentralityType, aCombineConj, aThermEventsType, 
                                 aRebin, fKStarCfLite->GetMinNorm(), fKStarCfLite->GetMaxNorm());
  fThermNonFlatBgd->SetCombineLamKchPM(aCombineLamKchPM);

  return fThermNonFlatBgd->GetThermCf();
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::DivideCfByThermBgd(bool aCombineConj, bool aCombineLamKchPM, ThermEventsType aThermEventsType)
{
  TH1* tThermBgd = GetThermNonFlatBackground(aCombineConj, aCombineLamKchPM, aThermEventsType);
  fKStarCfLite->DivideCfByThermBgd(tThermBgd);
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::GetThermNonFlatBackgroundFit()
{
  if(fNonFlatBackground) return fNonFlatBackground;

  TString tFitName = TString::Format("ThermNonFlatBgd_%s%s", cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);
  fNonFlatBackground = new TF1(tFitName, BackgroundFitter::NormalizedFitFunctionPolynomial, 0., fMaxBgdBuild, 8);

  fNonFlatBackground->FixParameter(0, cThermBgdParamValues[fAnalysisType][fCentralityType][0]);
  fNonFlatBackground->FixParameter(1, cThermBgdParamValues[fAnalysisType][fCentralityType][1]);
  fNonFlatBackground->FixParameter(2, cThermBgdParamValues[fAnalysisType][fCentralityType][2]);
  fNonFlatBackground->FixParameter(3, cThermBgdParamValues[fAnalysisType][fCentralityType][3]);
  fNonFlatBackground->FixParameter(4, cThermBgdParamValues[fAnalysisType][fCentralityType][4]);
  fNonFlatBackground->FixParameter(5, cThermBgdParamValues[fAnalysisType][fCentralityType][5]);
  fNonFlatBackground->FixParameter(6, cThermBgdParamValues[fAnalysisType][fCentralityType][6]);
  fNonFlatBackground->SetParameter(7, 1.);

  TH1* tCf = fKStarCfLite->Cf();
  tCf->Fit(tFitName, "0q", "", fMinBgdFit, fMaxBgdFit);

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::GetNonFlatBackground(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf)
{
  if(fNonFlatBackground && aNormalizeFitToCf==fNormalizeBgdFitToCf) return fNonFlatBackground;
  if(aBgdFitType==kPolynomial) return GetThermNonFlatBackgroundFit();

  fNormalizeBgdFitToCf = aNormalizeFitToCf;
  if(aFitType==kChi2PML)
  {
    fNonFlatBackground = FitNonFlatBackground(fKStarCfLite->Num(), fKStarCfLite->Den(), fKStarCfLite->Cf(), aBgdFitType, aFitType, aNormalizeFitToCf, 
                                              fMinBgdFit, fMaxBgdFit, fMaxBgdBuild, fKStarCfLite->GetMinNorm(), fKStarCfLite->GetMaxNorm());
  }
  else if(aFitType==kChi2)
  {
    assert(!aNormalizeFitToCf);  //It is already normalized by fitting to Cf
    fNonFlatBackground = FitNonFlatBackground(fKStarCfLite->Cf(), aBgdFitType, fMinBgdFit, fMaxBgdFit, fMaxBgdBuild, fKStarCfLite->GetMinNorm(), fKStarCfLite->GetMaxNorm());
  }
  else assert(0);

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* FitPartialAnalysis::GetNewNonFlatBackground(NonFlatBgdFitType aBgdFitType)
{
  TString tFitName = TString::Format("NonFlatBackgroundFit%s_%s%s%s", cNonFlatBgdFitTypeTags[aBgdFitType], 
                                                                    cAnalysisBaseTags[fAnalysisType],
                                                                    cCentralityTags[fCentralityType],
                                                                    cBFieldTags[fBFieldType]);
  switch(aBgdFitType) {
  case kLinear:
    //2 parameters
    //par[0]*x[0] + par[1]
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::FitFunctionLinear, 0., fMaxBgdBuild, 2);
    break;

  case kQuadratic:
    //3 parameters
    //par[0]*x[0]*x[0] + par[1]*x[0] + par[2]
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::FitFunctionQuadratic, 0., fMaxBgdBuild, 3);
    break;

  case kGaussian:
    //4 parameters (although, likely par[1] fixed to zero
    //par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::FitFunctionGaussian, 0., fMaxBgdBuild, 4);
    break;

  case kPolynomial:
    //7 parameters
    //par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + ...
    // + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::FitFunctionPolynomial, 0., fMaxBgdBuild, 7);
    break;

  default:
    cout << "FitPartialAnalysis::GetNewNonFlatBackground: Invalid NonFlatBgdFitType = " << aBgdFitType << " selected" << endl;
    assert(0);
  }

  for(unsigned int i=0; i<fBgdParameters.size(); i++) fNonFlatBackground->SetParameter(i, fBgdParameters[i]->GetFitValue());

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::InitializeBackgroundParams(NonFlatBgdFitType aNonFlatBgdType)
{
  fBgdParameters.clear();
//  double tScale = fKStarCfLite->GetNumScale()/fKStarCfLite->GetDenScale();
  double tScale = 1.;

  //FitParameter::FitParameter(ParameterType aParamType, double aStartValue, bool aIsFixed, double aLowerParamBound, double aUpperParamBound, double aStepSize);
  switch(aNonFlatBgdType) {
  case kLinear:
    //2 parameters
    //par[0]*x[0] + par[1]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));       //par[0]
    fBgdParameters.push_back(new FitParameter(kBgd, tScale, false, 0., 0., 0.001));      //par[1]
    break;

  case kQuadratic:
    //3 parameters
    //par[0]*x[0]*x[0] + par[1]*x[0] + par[2]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));       //par[0]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));       //par[1]
    fBgdParameters.push_back(new FitParameter(kBgd, tScale, false, 0., 0., 0.01));       //par[2]
    break;

  case kGaussian:
    //4 parameters (although, likely par[1] fixed to zero
    //par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.0001, false, 0., 0., 0.0001));       //par[0]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     true,  0., 0., 0.1));          //par[1]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.5,    false, 0., 0., 0.01));         //par[2]
    fBgdParameters.push_back(new FitParameter(kBgd, tScale, false, 0., 0., 0.1));          //par[3]
    break;

  case kPolynomial:
    //7 parameters
    //par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + ...
    // + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
    fBgdParameters.push_back(new FitParameter(kBgd, tScale, false, 0., 0., 0.01));         //par[0]
//    fBgdParameters.push_back(new FitParameter(kBgd, 1.0100, false, 1., 1.05, 0.01));         //par[0]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[1]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[2]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[3]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[4]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[5]
    fBgdParameters.push_back(new FitParameter(kBgd, 0.,     false, 0., 0., 0.01));         //par[6]
    break;

  default:
    cout << "FitPartialAnalysis::InitializeBackgroundParams: Invalid NonFlatBgdFitType = " << aNonFlatBgdType << " selected" << endl;
    assert(0);
  }

  //Default naming in FitParameter is to name them cParameterNames[fParamType] = Bgd, in this case
  //So, I want to give them each a unique name
  for(unsigned int i=0; i<fBgdParameters.size(); i++) fBgdParameters[i]->SetName(TString::Format("%s_%i", cParameterNames[kBgd], i));

  //-----Set owner information
  for(unsigned int i=0; i<fBgdParameters.size(); i++) fBgdParameters[i]->SetOwnerInfo(fAnalysisType, fCentralityType, fBFieldType);
}


//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetBgdParametersSharedLocal(bool aIsShared, vector<int> &aSharedAnalyses)
{
  for(unsigned int i=0; i<fBgdParameters.size(); i++)
  {
    fBgdParameters[i]->SetSharedLocal(aIsShared, aSharedAnalyses);
  }
}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetBgdParametersSharedGlobal(bool aIsShared, vector<int> &aSharedAnalyses)
{
  for(unsigned int i=0; i<fBgdParameters.size(); i++)
  {
    fBgdParameters[i]->SetSharedGlobal(aIsShared, aSharedAnalyses);
  }
}



//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetFitParameterShallow(FitParameter* aParam)
{
  //Created a shallow copy, which I think is what I want
  fFitParameters[aParam->GetType()] = aParam;
}

//________________________________________________________________________________________________________________
void FitPartialAnalysis::SetBgdParametersShallow(vector<FitParameter*> &aBgdParameters)
{
  assert(fBgdParameters.size() == aBgdParameters.size());
  for(unsigned int i=0; i<fBgdParameters.size(); i++) fBgdParameters[i] = aBgdParameters[i];
}


//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCf(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  TString tNumName = cKStarCfBaseTagNum + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName;
  if(!fUseNumRotPar2InsteadOfDen) tDenName = cKStarCfBaseTagDen;
  else tDenName = cKStarCfBaseTagNumRotatePar2;
  tDenName += TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCf_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  CfLite* tReturnCfLite = new CfLite(tCfName,tCfName,tNum,tDen,aKStarMinNorm,aKStarMaxNorm);
  if(aRebin != 1) tReturnCfLite->Rebin(aRebin);

  return tReturnCfLite;
}

//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCfFake(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  TString tNumName = cModelKStarCfNumFakeBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFake_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  fModelKStarCfFake = new CfLite(tCfName,tCfName,tNum,tDen,aKStarMinNorm,aKStarMaxNorm);
  if(aRebin != 1) fModelKStarCfFake->Rebin(aRebin);

  return fModelKStarCfFake;
}


//________________________________________________________________________________________________________________
CfLite* FitPartialAnalysis::GetModelKStarCfFakeIdeal(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  TString tNumName = cModelKStarCfNumFakeIdealBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewNumName = tNumName + TString(cBFieldTags[fBFieldType]);
  TH1* tNum = Get1dHisto(fFileLocationMC,tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenIdealBaseTag + TString(cAnalysisBaseTags[fAnalysisType]);
  TString tNewDenName = tDenName + TString(cBFieldTags[fBFieldType]);
  TH1* tDen = Get1dHisto(fFileLocationMC,tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFakeIdeal_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cBFieldTags[fBFieldType]);
  fModelKStarCfFakeIdeal = new CfLite(tCfName,tCfName,tNum,tDen,aKStarMinNorm,aKStarMaxNorm);
  if(aRebin != 1) fModelKStarCfFakeIdeal->Rebin(aRebin);

  return fModelKStarCfFakeIdeal;
}
