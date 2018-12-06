///////////////////////////////////////////////////////////////////////////
// PartialAnalysis:                                                      //
///////////////////////////////////////////////////////////////////////////


#include "PartialAnalysis.h"

#ifdef __ROOT__
ClassImp(PartialAnalysis)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
PartialAnalysis::PartialAnalysis(TString aFileLocation, TString aAnalysisName, AnalysisType aAnalysisType, BFieldType aBFieldType, CentralityType aCentralityType, AnalysisRunType aRunType, TString aDirNameModifier) :
  fAnalysisRunType(aRunType),
  fFileLocation(aFileLocation),
  fAnalysisName(aAnalysisName),
  fDirectoryName(0),
  fDir(0),

  fAnalysisType(aAnalysisType),
  fAnalysisBaseTag(cAnalysisBaseTags[fAnalysisType]),

  fBFieldType(aBFieldType),
  fBFieldTag(cBFieldTags[fBFieldType]),

  fCentralityType(aCentralityType),
  fCentralityTag(cCentralityTags[fCentralityType]),

  fParticleTypes(2),
  fDaughterParticleTypes(0),

  fNEventsPass(0),
  fNEventsFail(0),
  fNPart1Pass(0),
  fNPart1Fail(0),
  fNPart2Pass(0),
  fNPart2Fail(0),
  fNKStarNumEntries(0),

  fKStarCf(0),
  fKStarCfMCTrue(0),

  fModelKStarCfTrue(0),
  fModelKStarCfTrueIdeal(0),
  fModelKStarCfFake(0),
  fModelKStarCfFakeIdeal(0),
  fModelKStarCfFakeIdealSmeared(0),

  fModelKStarCfTrueUnitWeights(0),
  fModelKStarCfTrueIdealUnitWeights(0),

//  fModelKStarTrueVsRec(0),
  fModelKStarTrueVsRecSame(0),
  fModelKStarTrueVsRecRotSame(0),
  fModelKStarTrueVsRecMixed(0),
  fModelKStarTrueVsRecRotMixed(0),

  fDaughterPairTypes(0),
  fAvgSepCfs(kMaxNDaughterPairTypes),
  fSepCfs(kMaxNDaughterPairTypes),  //make size equal to total number of pair types so SepCfs can be stored in consistent matter
                                    //ex, for LamKchP analysis, {[0],[1],[2],[3]} = empty, {[4],[5]} = full, {[6]} = empty
  fAvgSepCowSailCfs(kMaxNDaughterPairTypes),

  fKStar2dCfKStarOut(0),
  fKStar1dCfKStarOutPos(0),
  fKStar1dCfKStarOutNeg(0),
  fKStar1dKStarOutPosNegRatio(0),

  fKStar2dCfKStarSide(0),
  fKStar1dCfKStarSidePos(0),
  fKStar1dCfKStarSideNeg(0),
  fKStar1dKStarSidePosNegRatio(0),

  fKStar2dCfKStarLong(0),
  fKStar1dCfKStarLongPos(0),
  fKStar1dCfKStarLongNeg(0),
  fKStar1dKStarLongPosNegRatio(0),

  fPart1MassFail(0)


{

  fDirectoryName = fAnalysisBaseTag + fCentralityTag;
  if(!aDirNameModifier.IsNull()) fDirectoryName += aDirNameModifier;
  fDir = ConnectAnalysisDirectory(fFileLocation,fDirectoryName);

  BuildKStarCf(/*double aMinNorm=0.32, double aMaxNorm=0.4*/);


  if( (fAnalysisType==kLamK0) || (fAnalysisType==kALamK0) || (fAnalysisType==kLamLam) || (fAnalysisType==kALamALam) || (fAnalysisType==kLamALam) ) 
  {

    fDaughterPairTypes.push_back(kPosPos);
    fDaughterPairTypes.push_back(kPosNeg);
    fDaughterPairTypes.push_back(kNegPos);
    fDaughterPairTypes.push_back(kNegNeg);

  }

  else if( (fAnalysisType==kLamKchP) || (fAnalysisType==kALamKchP) || (fAnalysisType==kLamKchM) || (fAnalysisType==kALamKchM) || (fAnalysisType==kLamPiP) || (fAnalysisType==kALamPiP) || (fAnalysisType==kLamPiM) || (fAnalysisType==kALamPiM) ) 
  {

    fDaughterPairTypes.push_back(kTrackPos);
    fDaughterPairTypes.push_back(kTrackNeg);

  }

  else if( (fAnalysisType==kXiKchP) || (fAnalysisType==kAXiKchP) || (fAnalysisType==kXiKchM) || (fAnalysisType==kAXiKchM) )
  {

    fDaughterPairTypes.push_back(kTrackPos);
    fDaughterPairTypes.push_back(kTrackNeg);
    fDaughterPairTypes.push_back(kTrackBac);

  }


  SetParticleTypes();
  SetDaughterParticleTypes();

  if(fAnalysisRunType != kTrainSys)  //TrainSys analyses DO NOT include FAIL cut monitors
  {
    SetNEventsPassFail();
    SetNPart1PassFail();
    SetNPart2PassFail();
    SetPart1MassFail();
  }

  SetNKStarNumEntries();


}







//________________________________________________________________________________________________________________
PartialAnalysis::~PartialAnalysis()
{
  cout << "PartialAnalysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}



//________________________________________________________________________________________________________________
TObjArray* PartialAnalysis::ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName)
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
    else if(aDirectoryName.Contains("XiK0")) tFemtoListName = "cXiK0";
    else
    {
      cout << "ERROR in Analysis::ConnectAnalysisDirectory!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << "Invalid aDirectoryName for fAnalysisRunType==kTrain||kTrainSys:  aDirectoryName = " << aDirectoryName << endl;
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
void PartialAnalysis::SetParticleTypes()
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

  else if(fAnalysisType == kXiK0) {fParticleTypes[0] = kXi; fParticleTypes[1] = kK0;}
  else if(fAnalysisType == kAXiK0) {fParticleTypes[0] = kAXi; fParticleTypes[1] = kK0;}

  else if(fAnalysisType == kLamLam) {fParticleTypes[0] = kLam; fParticleTypes[1] = kLam;}
  else if(fAnalysisType == kALamALam) {fParticleTypes[0] = kALam; fParticleTypes[1] = kALam;}
  else if(fAnalysisType == kLamALam) {fParticleTypes[0] = kLam; fParticleTypes[1] = kALam;}

  else if(fAnalysisType == kLamPiP) {fParticleTypes[0] = kLam; fParticleTypes[1] = kPiP;}
  else if(fAnalysisType == kALamPiP) {fParticleTypes[0] = kALam; fParticleTypes[1] = kPiP;}
  else if(fAnalysisType == kLamPiM) {fParticleTypes[0] = kLam; fParticleTypes[1] = kPiM;}
  else if(fAnalysisType == kALamPiM) {fParticleTypes[0] = kALam; fParticleTypes[1] = kPiM;}

  else{ cout << "ERROR IN SetParticleTypes:  Invalid fAnalysisType!!!!!!!" << endl << endl;}
}


//________________________________________________________________________________________________________________
void PartialAnalysis::SetDaughterParticleTypes()
{
  fDaughterParticleTypes.clear();

  if( (fAnalysisType==kLamK0) || (fAnalysisType==kALamK0) || (fAnalysisType==kLamLam) || (fAnalysisType==kALamALam) || (fAnalysisType==kLamALam) ) 
  {
    fDaughterParticleTypes.resize(2);

    vector<ParticleType> tPart1Daughters(2);
    vector<ParticleType> tPart2Daughters(2);

    if(fAnalysisType==kLamK0)
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;

      tPart2Daughters[0] = kPiP;
      tPart2Daughters[1] = kPiM;
    }
    else if(fAnalysisType==kALamK0)
    {
      tPart1Daughters[0] = kPiP;
      tPart1Daughters[1] = kAntiProton;

      tPart2Daughters[0] = kPiP;
      tPart2Daughters[1] = kPiM;
    }
    else if(fAnalysisType==kLamLam)
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;

      tPart2Daughters[0] = kProton;
      tPart2Daughters[1] = kPiM;
    }
    else if(fAnalysisType==kALamALam)
    {
      tPart1Daughters[0] = kPiP;
      tPart1Daughters[1] = kAntiProton;

      tPart2Daughters[0] = kPiP;
      tPart2Daughters[1] = kAntiProton;
    }
    else if(fAnalysisType==kLamALam)
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;

      tPart2Daughters[0] = kPiP;
      tPart2Daughters[1] = kAntiProton;
    }

    fDaughterParticleTypes[0] = tPart1Daughters;
    fDaughterParticleTypes[1] = tPart2Daughters;

  }



  else if( (fAnalysisType==kLamKchP) || (fAnalysisType==kALamKchP) || (fAnalysisType==kLamKchM) || (fAnalysisType==kALamKchM) || (fAnalysisType==kLamPiP) || (fAnalysisType==kALamPiP) || (fAnalysisType==kLamPiM) || (fAnalysisType==kALamPiM) ) 
  {
    fDaughterParticleTypes.resize(1);

    vector<ParticleType> tPart1Daughters(2);

    if( (fAnalysisType==kLamKchP) || (fAnalysisType==kLamKchM) || (fAnalysisType==kLamPiP) || (fAnalysisType==kLamPiM) )
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;
    }

    else if( (fAnalysisType==kALamKchP) || (fAnalysisType==kALamKchM) || (fAnalysisType==kALamPiP) || (fAnalysisType==kALamPiM) )
    {
      tPart1Daughters[0] = kPiP;
      tPart1Daughters[1] = kAntiProton;
    }

    fDaughterParticleTypes[0] = tPart1Daughters;

  }

  else if( (fAnalysisType==kXiKchP) || (fAnalysisType==kAXiKchP) || (fAnalysisType==kXiKchM) || (fAnalysisType==kAXiKchM) )
  {
    fDaughterParticleTypes.resize(1);
    vector<ParticleType> tPart1Daughters(3);

    if( (fAnalysisType==kXiKchP) || (fAnalysisType==kXiKchM) )
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;
      tPart1Daughters[2] = kPiM;
    }
    else if( (fAnalysisType==kAXiKchP) || (fAnalysisType==kAXiKchM) )
    {
      tPart1Daughters[0] = kPiP;
      tPart1Daughters[1] = kAntiProton;
      tPart1Daughters[2] = kPiP;
    }
    fDaughterParticleTypes[0] = tPart1Daughters;
  }

  else if( (fAnalysisType==kXiK0) || (fAnalysisType==kAXiK0) )
  {
    fDaughterParticleTypes.resize(2);
    vector<ParticleType> tPart1Daughters(3);
    vector<ParticleType> tPart2Daughters(2);

    if(fAnalysisType==kXiK0)
    {
      tPart1Daughters[0] = kProton;
      tPart1Daughters[1] = kPiM;
      tPart1Daughters[2] = kPiM;
    }
    else if(fAnalysisType==kAXiK0)
    {
      tPart1Daughters[0] = kPiP;
      tPart1Daughters[1] = kAntiProton;
      tPart1Daughters[2] = kPiP;
    }

    tPart2Daughters[0] = kPiP;
    tPart2Daughters[1] = kPiM;

    fDaughterParticleTypes[0] = tPart1Daughters;
    fDaughterParticleTypes[1] = tPart2Daughters;
  }

  else{ cout << "ERROR IN SetDaughterParticleTypes:  Invalid fAnalysisType!!!!!!!" << endl << endl;}


}



//________________________________________________________________________________________________________________
TH1* PartialAnalysis::Get1dHisto(TString aHistoName, TString aNewName)
{
  TH1 *tHisto = (TH1*)fDir->FindObject(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1 *ReturnHisto = (TH1*)tHisto->Clone(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  return (TH1*)ReturnHisto;
}



//________________________________________________________________________________________________________________
TH2* PartialAnalysis::Get2dHisto(TString aHistoName, TString aNewName)
{
  TH2 *tHisto = (TH2*)fDir->FindObject(aHistoName);

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
void PartialAnalysis::BuildKStarCf(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cKStarCfBaseTagNum + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cKStarCfBaseTagDen + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "KStarCf_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fKStarCf = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);

}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildKStarCfMCTrue(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cKStarCfMCTrueBaseTagNum + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cKStarCfMCTrueBaseTagDen + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "KStarCfMCTrue_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fKStarCfMCTrue = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);

}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfTrue(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumTrueBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfTrue_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfTrue = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfTrueIdeal(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumTrueIdealBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenIdealBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfTrueIdeal_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfTrueIdeal = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfFake(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumFakeBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFake_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfFake = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfFakeIdeal(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumFakeIdealBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenIdealBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfFakeIdeal_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfFakeIdeal = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfFakeIdealSmeared(TH2* aMomResMatrix, double aMinNorm, double aMaxNorm, int aRebinFactor)
{
  TString tNumName = cModelKStarCfNumFakeIdealBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);
  tNum->Rebin(aRebinFactor);

  TString tDenName = cModelKStarCfDenIdealBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);
  tDen->Rebin(aRebinFactor);

cout << "tNum->GetBinWidth(1) = " << tNum->GetBinWidth(1) << endl;
cout << "tDen->GetBinWidth(1) = " << tDen->GetBinWidth(1) << endl;
cout << "aMomResMatrix->GetXaxis()->GetBinWidth(1) = " << aMomResMatrix->GetXaxis()->GetBinWidth(1) << endl;
cout << "aMomResMatrix->GetYaxis()->GetBinWidth(1) = " << aMomResMatrix->GetYaxis()->GetBinWidth(1) << endl;

  assert(tNum->GetNbinsX() == tDen->GetNbinsX());
  for(int j=1; j <= tNum->GetNbinsX(); j++)
  {
    double tValueNum = 0.;
    double tValueDen = 0.;
    assert(tNum->GetBinCenter(j) == aMomResMatrix->GetYaxis()->GetBinCenter(j));
    assert(tDen->GetBinCenter(j) == aMomResMatrix->GetYaxis()->GetBinCenter(j));

    for(int i=1; i <= aMomResMatrix->GetNbinsX(); i++)
    {
      assert(tNum->GetBinCenter(j) == aMomResMatrix->GetXaxis()->GetBinCenter(j));
      assert(tDen->GetBinCenter(j) == aMomResMatrix->GetXaxis()->GetBinCenter(j));

      assert(tNum->GetBinContent(i) > 0. && tDen->GetBinContent(i) > 0.);
      tValueNum += tNum->GetBinContent(i)*aMomResMatrix->GetBinContent(i,j);
      tValueDen += tDen->GetBinContent(i)*aMomResMatrix->GetBinContent(i,j);
    }
    tValueNum /= aMomResMatrix->Integral(1,aMomResMatrix->GetNbinsX(),j,j);
    tValueDen /= aMomResMatrix->Integral(1,aMomResMatrix->GetNbinsX(),j,j);

    tNum->SetBinContent(j,tValueNum);
    tDen->SetBinContent(j,tValueDen);
  }



  TString tCfBaseName = "ModelKStarCfFakeIdealSmeared_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfFakeIdealSmeared = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
  //fModelKStarCfFakeIdealSmeared->Rebin(aRebinFactor);
}

//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfTrueUnitWeights(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumTrueUnitWeightsBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfTrueUnitWeights_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfTrueUnitWeights = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarCfTrueIdealUnitWeights(double aMinNorm, double aMaxNorm)
{
  TString tNumName = cModelKStarCfNumTrueIdealUnitWeightsBaseTag + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cModelKStarCfDenIdealBaseTag + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  TString tCfBaseName = "ModelKStarCfTrueIdealUnitWeights_";
  TString tCfName = tCfBaseName + fAnalysisBaseTag + fBFieldTag;
  fModelKStarCfTrueIdealUnitWeights = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildModelKStarTrueVsRec(KStarTrueVsRecType aType)
{
  TString tName, tNewName;

  switch(aType) {
  case kSame:
    tName = cModelKStarTrueVsRecSameBaseTag;
      tName += fAnalysisBaseTag;
    tNewName = tName + fBFieldTag;
    fModelKStarTrueVsRecSame = Get2dHisto(tName,tNewName);
    break;

  case kRotSame:
    tName = cModelKStarTrueVsRecRotSameBaseTag;
      tName += fAnalysisBaseTag;
    tNewName = tName + fBFieldTag;
    fModelKStarTrueVsRecRotSame = Get2dHisto(tName,tNewName);
    break;

  case kMixed:
    tName = cModelKStarTrueVsRecMixedBaseTag;
      tName += fAnalysisBaseTag;
    tNewName = tName + fBFieldTag;
    fModelKStarTrueVsRecMixed = Get2dHisto(tName,tNewName);
    break;

  case kRotMixed:
    tName = cModelKStarTrueVsRecRotMixedBaseTag;
      tName += fAnalysisBaseTag;
    tNewName = tName + fBFieldTag;
    fModelKStarTrueVsRecRotMixed = Get2dHisto(tName,tNewName);
    break;


  default:
    cout << "ERROR: PartialAnalysis::BuildModelKStarTrueVsRec:  Invalide KStarTrueVsRecType aType = "
         << aType << " selected" << endl;
    assert(0);
    break;
  }
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAllModelKStarTrueVsRec()
{
  BuildModelKStarTrueVsRec(kSame);
  BuildModelKStarTrueVsRec(kRotSame);
  BuildModelKStarTrueVsRec(kMixed);
  BuildModelKStarTrueVsRec(kRotMixed);
}

//________________________________________________________________________________________________________________
TH2* PartialAnalysis::GetModelKStarTrueVsRec(KStarTrueVsRecType aType)
{
  switch(aType) {
  case kSame:
    if(!fModelKStarTrueVsRecSame) BuildModelKStarTrueVsRec(aType);
    return fModelKStarTrueVsRecSame;
    break;

  case kRotSame:
    if(!fModelKStarTrueVsRecRotSame) BuildModelKStarTrueVsRec(aType);
    return fModelKStarTrueVsRecRotSame;
    break;

  case kMixed:
    if(!fModelKStarTrueVsRecMixed) BuildModelKStarTrueVsRec(aType);
    return fModelKStarTrueVsRecMixed;
    break;

  case kRotMixed:
    if(!fModelKStarTrueVsRecRotMixed) BuildModelKStarTrueVsRec(aType);
    return fModelKStarTrueVsRecRotMixed;
    break;


  default:
    cout << "ERROR: PartialAnalysis::GetModelKStarTrueVsRec:  Invalide KStarTrueVsRecType aType = "
         << aType << " selected" << endl;
    assert(0);
  }

  return 0;
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAvgSepCf(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm)
{
  TString tNumName = cAvgSepCfBaseTagsNum[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH1* tNum = Get1dHisto(tNumName,tNewNumName);

  TString tDenName = cAvgSepCfBaseTagsDen[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH1* tDen = Get1dHisto(tDenName,tNewDenName);

  //Note DOES NOT WORK:  TString tCfName = "AvgSepCf" + cDaughterPairTags[aDaughterPairType];
  //Solution (1): TString tTemp = "AvgSepCf";
  //              tCfName = tTemp + cDaughterPairTags[aDaughterPairType];
  //
  //Solution (2): TString tCfName = "AvgSepCf" + TString(cDaughterPairTags[aDaughterPairType]);

  TString tCfBaseName = "AvgSepCf";
  TString tCfName = tCfBaseName + cDaughterPairTags[aDaughterPairType] + "_" + fAnalysisBaseTag + fBFieldTag;
  CfLite* tCfLite = new CfLite(tCfName,tCfName,tNum,tDen,aMinNorm,aMaxNorm);

  fAvgSepCfs[aDaughterPairType] = tCfLite;
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAllAvgSepCfs(double aMinNorm, double aMaxNorm)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    BuildAvgSepCf(fDaughterPairTypes[i],aMinNorm,aMaxNorm);
  }

}


//________________________________________________________________________________________________________________
CfLite* PartialAnalysis::GetAvgSepCf(DaughterPairType aDaughterPairType)
{
  assert(fAvgSepCfs[aDaughterPairType]);
  return fAvgSepCfs[aDaughterPairType];
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildSepCfs(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm)
{
  TString tNumName = cSepCfsBaseTagsNum[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH2* tNum = Get2dHisto(tNumName,tNewNumName);

  TString tDenName = cSepCfsBaseTagsDen[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH2* tDen = Get2dHisto(tDenName,tNewDenName);

  int tNbinsX = 8;
  vector<vector<int> > tProjectionBins;
  vector<int> tTempVec;

  for(int iBin=0; iBin < tNbinsX; iBin++)
  {
    tTempVec.clear();
    tTempVec.push_back(iBin+1);  //iBin+1 because bin==0 is underflow bin
    tTempVec.push_back(iBin+1);

    tProjectionBins.push_back(tTempVec);
  }

  TString tCfBaseName = "SepCfs";
  TString tCfName = tCfBaseName + cDaughterPairTags[aDaughterPairType] + "_" + fAnalysisBaseTag + fBFieldTag + "_";
  Cf2dLite* tCf2dLite = new Cf2dLite(tCfName,tNum,tDen,kYaxis,tProjectionBins,aMinNorm,aMaxNorm);

  fSepCfs[aDaughterPairType] = tCf2dLite;
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAllSepCfs(double aMinNorm, double aMaxNorm)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    BuildSepCfs(fDaughterPairTypes[i],aMinNorm,aMaxNorm);
  }

}


//________________________________________________________________________________________________________________
Cf2dLite* PartialAnalysis::GetSepCfs(DaughterPairType aDaughterPairType)
{
  assert(fSepCfs[aDaughterPairType]);
  return fSepCfs[aDaughterPairType];
}



//________________________________________________________________________________________________________________
void PartialAnalysis::BuildKStar2dCfs(double aMinNorm, double aMaxNorm)
{
  TString tNumNameCommon = cKStar2dCfBaseTagNum + fAnalysisBaseTag;

  TString tNumNameKStarOut = tNumNameCommon + "KStarOut";
  TString tNumNameKStarSide = tNumNameCommon + "KStarSide";
  TString tNumNameKStarLong = tNumNameCommon + "KStarLong";

  TString tNewNumNameKStarOut = tNumNameKStarOut + fBFieldTag;
  TString tNewNumNameKStarSide = tNumNameKStarSide + fBFieldTag;
  TString tNewNumNameKStarLong = tNumNameKStarLong + fBFieldTag;

  TH2* tNumKStarOut = Get2dHisto(tNumNameKStarOut,tNewNumNameKStarOut);
  TH2* tNumKStarSide = Get2dHisto(tNumNameKStarSide,tNewNumNameKStarSide);
  TH2* tNumKStarLong = Get2dHisto(tNumNameKStarLong,tNewNumNameKStarLong);

  //----------

  TString tDenNameCommon = cKStar2dCfBaseTagDen + fAnalysisBaseTag;

  TString tDenNameKStarOut = tDenNameCommon + "KStarOut";
  TString tDenNameKStarSide = tDenNameCommon + "KStarSide";
  TString tDenNameKStarLong = tDenNameCommon + "KStarLong";

  TString tNewDenNameKStarOut = tDenNameKStarOut + fBFieldTag;
  TString tNewDenNameKStarSide = tDenNameKStarSide + fBFieldTag;
  TString tNewDenNameKStarLong = tDenNameKStarLong + fBFieldTag;

  TH2* tDenKStarOut = Get2dHisto(tDenNameKStarOut,tNewDenNameKStarOut);
  TH2* tDenKStarSide = Get2dHisto(tDenNameKStarSide,tNewDenNameKStarSide);
  TH2* tDenKStarLong = Get2dHisto(tDenNameKStarLong,tNewDenNameKStarLong);

  //----------

  TString tCfBaseNameCommon = "KStar2dCf";

  TString tCfBaseNameKStarOut = tCfBaseNameCommon + "KStarOut";
  TString tCfBaseNameKStarSide = tCfBaseNameCommon + "KStarSide";
  TString tCfBaseNameKStarLong = tCfBaseNameCommon + "KStarLong";

  TString tCfDaughtersBaseNameKStarOut = tCfBaseNameKStarOut + fAnalysisBaseTag + fBFieldTag + "_";
  TString tCfDaughtersBaseNameKStarSide = tCfBaseNameKStarSide + fAnalysisBaseTag + fBFieldTag + "_";
  TString tCfDaughtersBaseNameKStarLong = tCfBaseNameKStarLong + fAnalysisBaseTag + fBFieldTag + "_";
  //for now, I will have NumKStar2dCfKStarOutLamK0_Bp1_0, DenKStar2dCfKStarOutLamK0_Bp1_0, KStar2dCfKStarOutLamK0_Bp1_0 & NumKStar2dCfKStarOutLamK0_Bp1_1, DenKStar2dCfKStarOutLamK0_Bp1_1, KStar2dCfKStarOutLamK0_Bp1_1, etc

  vector<vector<int> > tProjectionBins;
  vector<int> tTempVec;
  for(int i=0; i<2; i++)
  {
    tTempVec.clear();
    tTempVec.push_back(i+1);
    tTempVec.push_back(i+1);

    tProjectionBins.push_back(tTempVec);
  }

  //-----
  fKStar2dCfKStarOut = new Cf2dLite(tCfDaughtersBaseNameKStarOut,tNumKStarOut,tDenKStarOut,kXaxis,tProjectionBins,aMinNorm,aMaxNorm);

  fKStar1dCfKStarOutNeg = fKStar2dCfKStarOut->GetDaughterCf(0);
  fKStar1dCfKStarOutPos = fKStar2dCfKStarOut->GetDaughterCf(1);

  fKStar1dKStarOutPosNegRatio = (TH1*)fKStar1dCfKStarOutPos->Cf()->Clone("KStar1dKStarOutPosNegRatio");;
  fKStar1dKStarOutPosNegRatio->Divide(fKStar1dCfKStarOutNeg->Cf());

  //-----
  fKStar2dCfKStarSide = new Cf2dLite(tCfDaughtersBaseNameKStarSide,tNumKStarSide,tDenKStarSide,kXaxis,tProjectionBins,aMinNorm,aMaxNorm);

  fKStar1dCfKStarSideNeg = fKStar2dCfKStarSide->GetDaughterCf(0);
  fKStar1dCfKStarSidePos = fKStar2dCfKStarSide->GetDaughterCf(1);

  fKStar1dKStarSidePosNegRatio = (TH1*)fKStar1dCfKStarSidePos->Cf()->Clone("KStar1dKStarSidePosNegRatio");;
  fKStar1dKStarSidePosNegRatio->Divide(fKStar1dCfKStarSideNeg->Cf());

  //-----
  fKStar2dCfKStarLong = new Cf2dLite(tCfDaughtersBaseNameKStarLong,tNumKStarLong,tDenKStarLong,kXaxis,tProjectionBins,aMinNorm,aMaxNorm);

  fKStar1dCfKStarLongNeg = fKStar2dCfKStarLong->GetDaughterCf(0);
  fKStar1dCfKStarLongPos = fKStar2dCfKStarLong->GetDaughterCf(1);

  fKStar1dKStarLongPosNegRatio = (TH1*)fKStar1dCfKStarLongPos->Cf()->Clone("KStar1dKStarLongPosNegRatio");;
  fKStar1dKStarLongPosNegRatio->Divide(fKStar1dCfKStarLongNeg->Cf());

}



//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAvgSepCowSailCfs(DaughterPairType aDaughterPairType, double aMinNorm, double aMaxNorm)
{
  TString tNumName = cAvgSepCfCowboysAndSailorsBaseTagsNum[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewNumName = tNumName + fBFieldTag;
  TH2* tNum = Get2dHisto(tNumName,tNewNumName);

  TString tDenName = cAvgSepCfCowboysAndSailorsBaseTagsDen[aDaughterPairType] + fAnalysisBaseTag;
  TString tNewDenName = tDenName + fBFieldTag;
  TH2* tDen = Get2dHisto(tDenName,tNewDenName);

  vector<vector<int> > tProjectionBins(2,vector<int>(2));
  tProjectionBins[0][0] = 1;
  tProjectionBins[0][1] = 20;
  tProjectionBins[1][0] = 21;
  tProjectionBins[1][1] = 40;

  TString tCfBaseName = "AvgSepCowSailCfs";
  TString tCfName = tCfBaseName + cDaughterPairTags[aDaughterPairType] + "_" + fAnalysisBaseTag + fBFieldTag + "_";
  Cf2dLite* tCf2dLite = new Cf2dLite(tCfName,tNum,tDen,kYaxis,tProjectionBins,aMinNorm,aMaxNorm);

  fAvgSepCowSailCfs[aDaughterPairType] = tCf2dLite;
}


//________________________________________________________________________________________________________________
void PartialAnalysis::BuildAllAvgSepCowSailCfs(double aMinNorm, double aMaxNorm)
{
  int tNCfs = fDaughterPairTypes.size();

  for(int i=0; i<tNCfs; i++)
  {
    BuildAvgSepCowSailCfs(fDaughterPairTypes[i],aMinNorm,aMaxNorm);
  }

}

//________________________________________________________________________________________________________________
Cf2dLite* PartialAnalysis::GetAvgSepCowSailCfs(DaughterPairType aDaughterPairType)
{
  assert(fAvgSepCowSailCfs[aDaughterPairType]);
  return fAvgSepCowSailCfs[aDaughterPairType];
}


//________________________________________________________________________________________________________________
TH1* PartialAnalysis::GetPurityHisto(ParticleType aParticleType)
{
  assert((aParticleType==kLam) || (aParticleType==kALam) || (aParticleType==kK0) || (aParticleType==kXi) || (aParticleType==kAXi) );

  TString tHistoName;

  if(aParticleType==kLam) {tHistoName = TString(cLambdaPurityTag);}
  else if(aParticleType==kALam) {tHistoName = TString(cAntiLambdaPurityTag);}
  else if(aParticleType==kK0) {tHistoName = TString(cK0ShortPurityTag);}
  else if(aParticleType==kXi) {tHistoName = TString(cXiPurityTag);}
  else if(aParticleType==kAXi) {tHistoName = TString(cAXiPurityTag);}

  TString tNewName = tHistoName + fAnalysisBaseTag + fBFieldTag + fCentralityTag;

  TH1* tPurityHisto = Get1dHisto(tHistoName,tNewName);

  return tPurityHisto;
}



//________________________________________________________________________________________________________________
void PartialAnalysis::SetNEventsPassFail()
{
  TH1* tNormEvMult_EvPass = Get1dHisto("NormEvMult_EvPass","NormEvMult_EvPass");
  TH1* tNormEvMult_EvFail = Get1dHisto("NormEvMult_EvFail","NormEvMult_EvFail");

  fNEventsPass = tNormEvMult_EvPass->GetEntries();
  fNEventsFail = tNormEvMult_EvFail->GetEntries();
}

//________________________________________________________________________________________________________________
void PartialAnalysis::SetNPart1PassFail()
{
  TString tPassName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Pass";
  TString tFailName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[0]]) + "_Fail";

  TH1* tPart1PassHisto = Get1dHisto(tPassName,tPassName);
  TH1* tPart1FailHisto = Get1dHisto(tFailName,tFailName);

  fNPart1Pass = tPart1PassHisto->GetEntries();
  fNPart1Fail = tPart1FailHisto->GetEntries();

}


//________________________________________________________________________________________________________________
void PartialAnalysis::SetNPart2PassFail()
{
  TString tPassName;
  TString tFailName;

  if( (fParticleTypes[1]==kLam) || (fParticleTypes[1]==kALam) || (fParticleTypes[1]==kK0) )
  {
    tPassName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[1]]) + "_Pass";
    tFailName = "LambdaMass_" + TString(cParticleTags[fParticleTypes[1]]) + "_Fail";

    TH1* tPart2PassHisto = Get1dHisto(tPassName,tPassName);
    TH1* tPart2FailHisto = Get1dHisto(tFailName,tFailName);

    fNPart2Pass = tPart2PassHisto->GetEntries();
    fNPart2Fail = tPart2FailHisto->GetEntries();
  }

  else if( (fParticleTypes[1]==kKchP) || (fParticleTypes[1]==kKchM) || (fParticleTypes[1]==kPiP) || (fParticleTypes[1]==kPiM) )
  {
    tPassName = "YPt_" + TString(cParticleTags[fParticleTypes[1]]) + "_Pass";
    tFailName = "YPt_" + TString(cParticleTags[fParticleTypes[1]]) + "_Fail";

    TH2* tPart2PassHisto = Get2dHisto(tPassName,tPassName);
    TH2* tPart2FailHisto = Get2dHisto(tFailName,tFailName);

    fNPart2Pass = tPart2PassHisto->GetEntries();
    fNPart2Fail = tPart2FailHisto->GetEntries();
  }

  else {cout << "ERROR IN SetNPart2PassFail!!!!!" << endl << "Unexpected fParticlesTypes[1]!!!!!!!!!" << endl;}


}



//________________________________________________________________________________________________________________
void PartialAnalysis::SetNKStarNumEntries()
{
  fNKStarNumEntries = fKStarCf->Num()->GetEntries();
}


//________________________________________________________________________________________________________________
void PartialAnalysis::SetPart1MassFail()
{
  TString tHistoName;

  if(fParticleTypes[0]==kLam)
  {
    tHistoName = "LambdaMass_";
  }

  else if(fParticleTypes[0]==kALam)
  {
    tHistoName = "AntiLambdaMass_";
  }

  else if(fParticleTypes[0]==kXi)
  {
    tHistoName = "fXiMass_";
  }

  else if(fParticleTypes[0]==kAXi)
  {
    tHistoName = "fXiMass_";
  }

  tHistoName += TString(cParticleTags[fParticleTypes[0]]);
  tHistoName += "_Fail";

  TString tNewHistoName = tHistoName + fBFieldTag;

  fPart1MassFail = Get1dHisto(tHistoName,tNewHistoName);

}

//________________________________________________________________________________________________________________
TH1* PartialAnalysis::GetMCKchPurityHisto(bool aBeforePairCut)
{
  TString tHistoName;
  TString tNewHistoName;

  if(aBeforePairCut)
  {
    tHistoName = "PId_" + TString(cParticleTags[fParticleTypes[1]]) + "_Pass";
    tNewHistoName = "PId_" + TString(cParticleTags[fParticleTypes[1]]) + "_BeforePairCut" + fBFieldTag;
  }
  else
  {
    tHistoName = "PIdSecondPass";
    tNewHistoName = "PId_" + TString(cParticleTags[fParticleTypes[1]]) + "_AfterPairCut" + fBFieldTag;
  }

  TH1* tReturnHisto = Get1dHisto(tHistoName,tNewHistoName);

  return tReturnHisto;

}



//________________________________________________________________________________________________________________
CfLite* PartialAnalysis::GetSHCfLite(int al, int am, bool aRealComponent, double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<TString> tReImVec{"Im", "Re"};
  TString tCommonBase = TString::Format("%sYlm%d%dDirectYlmCf_%s", tReImVec[aRealComponent].Data(), al, am, fAnalysisBaseTag.Data());
  TString tNumName = TString::Format("Num", tCommonBase.Data());
  TString tDenName = TString::Format("Den", tCommonBase.Data());

  TH1* tNum = Get1dHisto(tNumName, TString::Format("%s%s", tNumName.Data(), fBFieldTag.Data()));
  TH1* tDen = Get1dHisto(tDenName, TString::Format("%s%s", tDenName.Data(), fBFieldTag.Data()));

  TString tCfName = TString::Format("%sYlmCfLite%d%d_%s%s%s", tReImVec[aRealComponent].Data(), al, am, fAnalysisBaseTag.Data(), fCentralityTag.Data(), fBFieldTag.Data());
  CfLite* tReturnCfLite = new CfLite(tCfName, tCfName, tNum, tDen, aMinNorm, aMaxNorm);
  if(aRebin != 1) tReturnCfLite->Rebin(aRebin);
  return tReturnCfLite;
}



//________________________________________________________________________________________________________________
TH1* PartialAnalysis::GetSHCf(int al, int am, bool aRealComponent, double aMinNorm, double aMaxNorm, int aRebin)
{
  CfLite* tCfLite = GetSHCfLite(al, am, aRealComponent, aMinNorm, aMaxNorm, aRebin);
  return tCfLite->Cf();
}







