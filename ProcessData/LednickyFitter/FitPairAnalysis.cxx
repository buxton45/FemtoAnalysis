///////////////////////////////////////////////////////////////////////////
// FitPairAnalysis:                                                      //
///////////////////////////////////////////////////////////////////////////

#include "FitPairAnalysis.h"

#ifdef __ROOT__
ClassImp(FitPairAnalysis)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aAnalysisName, vector<FitPartialAnalysis*> &aFitPartialAnalysisCollection) :
  fAnalysisRunType(kTrain),
  fAnalysisName(aAnalysisName),
  fFitPartialAnalysisCollection(aFitPartialAnalysisCollection),
  fNFitPartialAnalysis(fFitPartialAnalysisCollection.size()),

  fAnalysisType(fFitPartialAnalysisCollection[0]->GetAnalysisType()),
  fCentralityType(fFitPartialAnalysisCollection[0]->GetCentralityType()),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(0),
  fKStarCf(0),
  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),
  fFit(0),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),

  fModelKStarTrueVsRecMixed(0),
  fModelKStarHeavyCfFake(0),
  fModelKStarHeavyCfFakeIdeal(0),
  fModelCfFakeIdealCfFakeRatio(0),
  fTransformMatrices(0)

{

  //set fFitPartialAnalysisNumber in each FitPartialAnalysis object
  for(int i=0; i<fNFitPartialAnalysis; i++) {fFitPartialAnalysisCollection[i]->SetFitPartialAnalysisNumber(i);}

  //make sure partial analyses in collection have same pair type (AnalysisType) and centrality (CentralityType)
  for(int i=1; i<fNFitPartialAnalysis; i++)
  {
    assert(fFitPartialAnalysisCollection[i-1]->GetAnalysisType() == fFitPartialAnalysisCollection[i]->GetAnalysisType());
    assert(fFitPartialAnalysisCollection[i-1]->GetCentralityType() == fFitPartialAnalysisCollection[i]->GetCentralityType());
  }

  //Don't need to make sure they have same particle types, because analysis types are same
  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  BuildKStarCfHeavy(fKStarMinNorm,fKStarMaxNorm);

  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters();

}



//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNFitPartialAnalysis, TString aDirNameModifier) :
  fAnalysisRunType(aRunType),
  fAnalysisName(0),
  fFitPartialAnalysisCollection(0),
  fNFitPartialAnalysis(aNFitPartialAnalysis),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(0),
  fKStarCf(0),
  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),
  fFit(0),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),

  fModelKStarTrueVsRecMixed(0),
  fModelKStarHeavyCfFake(0),
  fModelKStarHeavyCfFakeIdeal(0),
  fModelCfFakeIdealCfFakeRatio(0),
  fTransformMatrices(0)

{
  fAnalysisName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  int iStart;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) iStart=0;
  else iStart = 2;

  for(int i=iStart; i<fNFitPartialAnalysis+iStart; i++)
  {
    BFieldType tBFieldType = static_cast<BFieldType>(i);

    TString tFileLocation = aFileLocationBase + cBFieldTags[tBFieldType];
    tFileLocation += ".root";

    TString tFitPartialAnalysisName = fAnalysisName + cBFieldTags[tBFieldType];

    FitPartialAnalysis* tFitPartialAnalysis = new FitPartialAnalysis(tFileLocation, tFitPartialAnalysisName, fAnalysisType, fCentralityType, tBFieldType, fAnalysisRunType, aDirNameModifier);

    fFitPartialAnalysisCollection.push_back(tFitPartialAnalysis);
  } 

  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  BuildKStarCfHeavy(fKStarMinNorm,fKStarMaxNorm);

  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters();

}

//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNFitPartialAnalysis, TString aDirNameModifier) :
  fAnalysisRunType(aRunType),
  fAnalysisName(0),
  fFitPartialAnalysisCollection(0),
  fNFitPartialAnalysis(aNFitPartialAnalysis),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(0),
  fKStarCf(0),
  fKStarMinNorm(0.32),
  fKStarMaxNorm(0.40),
  fFit(0),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),

  fModelKStarTrueVsRecMixed(0),
  fModelKStarHeavyCfFake(0),
  fModelKStarHeavyCfFakeIdeal(0),
  fModelCfFakeIdealCfFakeRatio(0),
  fTransformMatrices(0)

{
  fAnalysisName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  int iStart;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) iStart=0;
  else iStart = 2;

  for(int i=iStart; i<fNFitPartialAnalysis+iStart; i++)
  {
    BFieldType tBFieldType = static_cast<BFieldType>(i);

    TString tFileLocation = aFileLocationBase + cBFieldTags[tBFieldType];
    tFileLocation += ".root";

    TString tFileLocationMC = aFileLocationBaseMC + cBFieldTags[tBFieldType];
    tFileLocationMC += ".root";

    TString tFitPartialAnalysisName = fAnalysisName + cBFieldTags[tBFieldType];

    FitPartialAnalysis* tFitPartialAnalysis = new FitPartialAnalysis(tFileLocation, tFileLocationMC, tFitPartialAnalysisName, fAnalysisType, fCentralityType, tBFieldType, fAnalysisRunType, aDirNameModifier);

    fFitPartialAnalysisCollection.push_back(tFitPartialAnalysis);
  } 

  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  BuildKStarCfHeavy(fKStarMinNorm,fKStarMaxNorm);
  BuildModelKStarTrueVsRecMixed();
  BuildModelCfFakeIdealCfFakeRatio(fKStarMinNorm,fKStarMaxNorm,1);

  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters();

}





//________________________________________________________________________________________________________________
FitPairAnalysis::~FitPairAnalysis()
{
  cout << "FitPairAnalysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarTrueVsRecMixed()
{
  TString tName = "ModelKStarTrueVsRecMixed_" + TString(cAnalysisBaseTags[fAnalysisType]);

  TH2* tPre = (TH2*)fFitPartialAnalysisCollection[0]->GetModelKStarTrueVsRecMixed();
  fModelKStarTrueVsRecMixed = (TH2*)tPre->Clone(tName);

  for(int i=1; i<fNFitPartialAnalysis; i++)
  {
    TH2* tToAdd = (TH2*)fFitPartialAnalysisCollection[i]->GetModelKStarTrueVsRecMixed();
    fModelKStarTrueVsRecMixed->Add(tToAdd);
  }

}



//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildKStarCfHeavy(double aMinNorm, double aMaxNorm)
{
  fKStarMinNorm = aMinNorm;
  fKStarMaxNorm = aMaxNorm;

  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated
    fFitPartialAnalysisCollection[iAnaly]->BuildKStarCf(fKStarMinNorm,fKStarMaxNorm);

    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetKStarCfLite());
  }

  TString tCfBaseName = "KStarHeavyCf_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fKStarCfHeavy = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,fKStarMinNorm,fKStarMaxNorm);
  fKStarCf = fKStarCfHeavy->GetHeavyCf();
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::RebinKStarCfHeavy(int aRebinFactor, double aMinNorm, double aMaxNorm)
{
  fKStarMinNorm = aMinNorm;
  fKStarMaxNorm = aMaxNorm;

  fKStarCfHeavy->Rebin(aRebinFactor,fKStarMinNorm,fKStarMaxNorm);
  fKStarCf = fKStarCfHeavy->GetHeavyCf();
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::DrawKStarCfHeavy(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fKStarCfHeavy->GetHeavyCf();

  TAxis *xax1 = tCfToDraw->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = tCfToDraw->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();


  //------------------------------------------------------
  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);
  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  //tCfToDraw->SetTitle(tTitle);



  //------------------------------------------------------
  TLine *line = new TLine(0,1,1,1);
  line->SetLineColor(14);

  tCfToDraw->Draw(aOption);
  line->Draw();

}



//________________________________________________________________________________________________________________
void FitPairAnalysis::CreateFitNormParameters()
{
  fFitNormParameters.clear();

  for(int i=0; i<fNFitPartialAnalysis; i++)
  {
    fFitNormParameters.push_back(fFitPartialAnalysisCollection[i]->GetFitNormParameter());
  }

  fNFitNormParams = fFitNormParameters.size();

  if(fAnalysisRunType==kGrid && fNFitNormParams != 5)
  {
    cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitNormParams != 5 (the typical value)" << endl;
    cout << "Instead, fNFitNormParams = " << fNFitNormParams << endl;
    cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }
  else if(!(fAnalysisRunType==kGrid) && fNFitNormParams != 2)
  {
    cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitNormParams != 2 (the typical value)" << endl;
    cout << "Instead, fNFitNormParams = " << fNFitNormParams << endl;
    cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }



}


//________________________________________________________________________________________________________________
void FitPairAnalysis::ShareFitParameters()
{
  fFitParameters.clear();

  CreateFitNormParameters();

  vector<int> tAllShared (fNFitPartialAnalysis);
  for(int i=0; i<fNFitPartialAnalysis; i++) {tAllShared[i] = i;}

  for(int i=0; i<fNFitParamsToShare; i++)
  {
    ParameterType tParamType = static_cast<ParameterType>(i);
    fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType)->SetSharedLocal(true,tAllShared);
  }

  for(int iAnaly=1; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    for(int iPar=0; iPar<fNFitParamsToShare; iPar++)
    {
      ParameterType tParamType = static_cast<ParameterType>(iPar);
      fFitPartialAnalysisCollection[iAnaly]->GetFitParameter(tParamType)->SetSharedLocal(true,tAllShared);
      fFitPartialAnalysisCollection[iAnaly]->SetFitParameter(fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType));
    }
  }

  for(int i=0; i<fNFitParamsToShare; i++)
  {
    ParameterType tParamType = static_cast<ParameterType>(i);
    fFitParameters.push_back(fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType));
  }


  fNFitParams = fFitParameters.size();

  if(fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM)
  {
    if(fNFitParams != 8)
    {
      cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitParams != 8 (the typical value)" << endl;
      cout << "Instead, fNFitParams = " << fNFitParams << endl;
      cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
  }

  else
  {
    if(fNFitParams != 5)
    {
      cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitParams != 5 (the typical value)" << endl;
      cout << "Instead, fNFitParams = " << fNFitParams << endl;
      cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
  }

}
/*
//________________________________________________________________________________________________________________
void FitPairAnalysis::SetFitParameter(FitParameter* aParam)
{
  //Created a shallow copy, which I think is what I want
  fFitParameters[aParam->GetType()] = aParam;
}
*/

//________________________________________________________________________________________________________________
void FitPairAnalysis::SetFitParameter(FitParameter* aParam)
{
  assert(int(aParam->GetType()) < 8); //I do not want this function to touch the normalizations (kNorm = 8)

  for(int i=0; i<fNFitPartialAnalysis; i++) {fFitPartialAnalysisCollection[i]->SetFitParameter(aParam);}
  fFitParameters[aParam->GetType()] = aParam;
}

//________________________________________________________________________________________________________________
void FitPairAnalysis::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetRangeUser(aMin,aMax);

  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}

//________________________________________________________________________________________________________________
void FitPairAnalysis::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}


//________________________________________________________________________________________________________________
void FitPairAnalysis::DrawFit(const char* aTitle)
{
  gStyle->SetOptFit();

  gStyle->SetStatH(0.15);
  gStyle->SetStatW(0.30);

  gStyle->SetStatX(0.85);
  gStyle->SetStatY(0.60);

  TAxis *xax = fKStarCf->GetXaxis();
  SetupAxis(xax,0.,0.5,"k* (GeV/c)",0.05,0.9,false,0.03,0.005,510);
  TAxis *yax = fKStarCf->GetYaxis();
  SetupAxis(yax,0.9,1.04,"C(k*)",0.05,0.9,false,0.03,0.005,510);

  fKStarCf->SetTitle(aTitle);
  fKStarCf->SetMarkerStyle(20);
  fKStarCf->SetMarkerSize(0.5);

  fKStarCf->Draw();
  fFit->SetLineColor(1);
  fFit->Draw("same");

  TLine *line = new TLine(0,1,0.5,1);
  line->SetLineColor(14);
  line->Draw();
/*
  TH1F* tMomResCorrectedFitHisto = GetMomResCorrectedFitHisto();
  tMomResCorrectedFitHisto->SetLineColor(2);
  tMomResCorrectedFitHisto->Draw("Lsame");
*/
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarHeavyCfFake(double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetModelKStarCfFake(aMinNorm,aMaxNorm,aRebin));
  }

  TString tCfBaseName = "ModelKStarHeavyCfFake_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFake = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarHeavyCfFakeIdeal(double aMinNorm, double aMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetModelKStarCfFakeIdeal(aMinNorm,aMaxNorm,aRebin));
  }

  TString tCfBaseName = "ModelKStarHeavyCfFakeIdeal_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFakeIdeal = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aMinNorm,aMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelCfFakeIdealCfFakeRatio(double aMinNorm, double aMaxNorm, int aRebinFactor)
{
  TString tName = "ModelCfFakeIdealCfFakeRatio_" + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  BuildModelKStarHeavyCfFake(aMinNorm,aMaxNorm,aRebinFactor);
  BuildModelKStarHeavyCfFakeIdeal(aMinNorm,aMaxNorm,aRebinFactor);

  fModelCfFakeIdealCfFakeRatio = (TH1*)fModelKStarHeavyCfFakeIdeal->GetHeavyCf()->Clone(tName);
  fModelCfFakeIdealCfFakeRatio->SetTitle(tName);
  fModelCfFakeIdealCfFakeRatio->Divide((TH1*)fModelKStarHeavyCfFake->GetHeavyCf());


}


//________________________________________________________________________________________________________________
TH1F* FitPairAnalysis::GetMomResCorrectedFitHisto()
{
  int tNbinsX = fKStarCf->GetNbinsX();
  double tKStarMin = fKStarCf->GetBinLowEdge(1);
  double tKStarMax = fKStarCf->GetBinLowEdge(tNbinsX+1);

  TH1F* tUncorrected = new TH1F("tUncorrected","tUncorrected",tNbinsX,tKStarMin,tKStarMax);
  for(int i=1; i<=tNbinsX; i++)
  {
    tUncorrected->SetBinContent(i,fFit->Eval(tUncorrected->GetBinCenter(i)));
    tUncorrected->SetBinError(i,0.);
  }

  TString tName = "MomResCorrectedFitHisto_" + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  TH1F* tReturnHisto = new TH1F(tName,tName,tNbinsX,tKStarMin,tKStarMax);
  for(int j=1; j<=tUncorrected->GetNbinsX(); j++)
  {
    double tValue = 0.;
    assert(tUncorrected->GetBinCenter(j) == fModelKStarTrueVsRecMixed->GetYaxis()->GetBinCenter(j));
    for(int i=1; i<=fModelKStarTrueVsRecMixed->GetNbinsX(); i++)
    {
      assert(tUncorrected->GetBinCenter(i) == fModelKStarTrueVsRecMixed->GetXaxis()->GetBinCenter(i));
      assert(tUncorrected->GetBinContent(i) > 0.);
      tValue += tUncorrected->GetBinContent(i)*fModelKStarTrueVsRecMixed->GetBinContent(i,j);
    }
    tValue /= fModelKStarTrueVsRecMixed->Integral(1,fModelKStarTrueVsRecMixed->GetNbinsX(),j,j);
    tReturnHisto->SetBinContent(j,tValue);
    tReturnHisto->SetBinError(j,0.);
  }

  delete tUncorrected;
  return tReturnHisto;
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::LoadTransformMatrices(TString aFileLocation)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TString tName2 = cAnalysisBaseTags[fAnalysisType] + TString("Transform");

  TString tName1Sig   = TString("SigTo");
  TString tName1XiC   = TString("XiCTo");
  TString tName1Xi0   = TString("Xi0To");
  TString tName1Omega = TString("OmegaTo");

  TString tFullNameSig, tFullNameXiC, tFullNameXi0, tFullNameOmega;

  switch(fAnalysisType) {
  case kLamKchP:
  case kLamKchM:
    tFullNameSig = TString("f") + tName1Sig + tName2;
    tFullNameXiC = TString("f") + tName1XiC + tName2;
    tFullNameXi0 = TString("f") + tName1Xi0 + tName2;
    tFullNameOmega = TString("f") + tName1Omega + tName2;
    break;

  case kALamKchP:
  case kALamKchM:
    tFullNameSig = TString("fA") + tName1Sig + tName2;
    tFullNameXiC = TString("fA") + tName1XiC + tName2;
    tFullNameXi0 = TString("fA") + tName1Xi0 + tName2;
    tFullNameOmega = TString("fA") + tName1Omega + tName2;
    break;

  default:
    cout << "ERROR:  fAnalysisType = " << fAnalysisType << " is not apropriate" << endl << endl;
    assert(0);
  }

  fTransformMatrices.clear();
  fTransformMatrices.push_back((TH2D*)tFile->Get(tFullNameSig));
  fTransformMatrices.push_back((TH2D*)tFile->Get(tFullNameXiC));
  fTransformMatrices.push_back((TH2D*)tFile->Get(tFullNameXi0));
  fTransformMatrices.push_back((TH2D*)tFile->Get(tFullNameOmega));
}




