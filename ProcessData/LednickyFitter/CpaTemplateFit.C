#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

#include "Fit/Fitter.h"
#include "TFractionFitter.h"

vector<TString> tDefaultBins{"Primary", 
                             "#Lambda", "K^{0}_{S}", 
                             "#Sigma^{0}", "#Xi^{0}", "#Xi^{-}", 
                             "#Sigma^{*+}", "#Sigma^{*-}", "#Sigma^{*0}", "K^{*0}", "K^{*+}", 
                             "Other", "Fake", "Material"}; 


//________________________________________________________________________________________________________________
TH1F* Get1dHist(TString aFileLocation, TString aDirectoryName, TString aHistName, TString aNewName)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
  else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
  else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
  else assert(0);

  tFemtoListName += TString("_femtolist");
  tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  aDirectoryName.ReplaceAll("0010","010");

  tFemtolist->SetOwner();

  TObjArray *tDir = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    tDir->SetOwner();

  //-----------------------------------------------------------------------------

  TH1F *tHisto = (TH1F*)tDir->FindObject(aHistName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1F *ReturnHist = (TH1F*)tHisto->Clone(aNewName);
  ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}


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

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;

  //-------------------------------------------------------------------

  return (TH1F*)ReturnHist;
}

//________________________________________________________________________________________________________________
TH2F* Get2dHist(TString aFileLocation, TString aDirectoryName, TString aHistName, TString aNewName)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
  else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
  else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
  else assert(0);

  tFemtoListName += TString("_femtolist");
  tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  aDirectoryName.ReplaceAll("0010","010");

  tFemtolist->SetOwner();

  TObjArray *tDir = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    tDir->SetOwner();

  //-----------------------------------------------------------------------------

  TH2F *tHisto = (TH2F*)tDir->FindObject(aHistName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHisto);
  //----------------------------------

  TH2F *ReturnHist = (TH2F*)tHisto->Clone(aNewName);
  ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}


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

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;

  //-------------------------------------------------------------------

  return (TH2F*)ReturnHist;
}


//________________________________________________________________________________________________________________
TH1F* Get1dProjection(TH2F* a2dHist, TString aBinName, int aMarkerStyle=20, int aColor=1, int aFillStyle=0)
{
  int tBinNumber = -1;
  for(int i=1; i<=a2dHist->GetNbinsY(); i++) if(aBinName.EqualTo(a2dHist->GetYaxis()->GetBinLabel(i))) tBinNumber=i;
  assert(tBinNumber > 0);

  TString tNewName = TString(aBinName.Data());;
  if     (((TString)a2dHist->GetName()).Contains("Cpa")) tNewName += TString(" Cpa");
  else if(((TString)a2dHist->GetName()).Contains("Dca")) tNewName += TString(" Dca");
  else assert(0);

  TH1F* tReturnHist = (TH1F*)a2dHist->ProjectionX(tNewName, tBinNumber, tBinNumber);
  tReturnHist->SetMarkerStyle(aMarkerStyle);
  tReturnHist->SetMarkerColor(aColor);
  tReturnHist->SetLineColor(aColor);
  tReturnHist->SetLineWidth(2);
  tReturnHist->SetFillStyle(aFillStyle);
  tReturnHist->SetFillColor(aColor);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!tReturnHist->GetSumw2N()) {tReturnHist->Sumw2();}

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1F* Get1dProjection(TH2F* a2dHist, int aDefaultBin, int aMarkerStyle=20, int aColor=1, int aFillStyle=0)
{
  TString tBinName = tDefaultBins[aDefaultBin];
  return Get1dProjection(a2dHist, tBinName, aMarkerStyle, aColor, aFillStyle);
}

//________________________________________________________________________________________________________________
TObjArray* DoProjections(TH2F* a2dMC, int aNRes=3)
{
  bool bIncludeKSt=false;
  if(TString(a2dMC->GetName()).Contains("Lam")) bIncludeKSt=false;
  else if(TString(a2dMC->GetName()).Contains("K0")) bIncludeKSt=true;
  else assert(0);

  //First, grab all MC hisograms
  TH1F* tPrimary  = Get1dProjection(a2dMC, 0,  20, 1, 0);
//  TH1F* tLambda   = Get1dProjection(a2dMC, 1,  20, 1, 0);
//  TH1F* tK0s      = Get1dProjection(a2dMC, 2,  20, 1, 0);
  TH1F* tSig0     = Get1dProjection(a2dMC, 3,  20, 2, 0);
  TH1F* tXi0      = Get1dProjection(a2dMC, 4,  20, 3, 0);
  TH1F* tXiC      = Get1dProjection(a2dMC, 5,  20, 4, 0);
  TH1F* tSigStP   = Get1dProjection(a2dMC, 6,  20, kTeal, 0);
  TH1F* tSigStM   = Get1dProjection(a2dMC, 7,  20, kTeal-5, 0);
  TH1F* tSigSt0   = Get1dProjection(a2dMC, 8,  20, kTeal-7, 0);
  TH1F* tKSt0     = Get1dProjection(a2dMC, 9,  20, kOrange, 0);
  TH1F* tKStP     = Get1dProjection(a2dMC, 10, 20, kOrange+6, 0);
  TH1F* tOther    = Get1dProjection(a2dMC, 11, 20, kPink+6, 0);
  TH1F* tFake     = Get1dProjection(a2dMC, 12, 20, kAzure+10, 0);
  TH1F* tMaterial = Get1dProjection(a2dMC, 13, 20, kYellow-7, 0);

  //---------------------------------------------

  TObjArray* tMCArr = new TObjArray();
  tMCArr->Add(tPrimary);
  tMCArr->Add(tSig0);
  tMCArr->Add(tXi0);
  tMCArr->Add(tXiC);
  if(aNRes==10)
  {
/*
    tMCArr->Add(tSigStP);
    tMCArr->Add(tSigStM);
    tMCArr->Add(tSigSt0);
*/
    TH1F* tSigStAll = (TH1F*)tSigStP->Clone("#Sigma^{*}");
    tSigStAll->Add(tSigStM);
    tSigStAll->Add(tSigSt0);
    tMCArr->Add(tSigStAll);
    if(bIncludeKSt)
    {
      tMCArr->Add(tKSt0);
      tMCArr->Add(tKStP);
    }
  }
  else
  {
    tOther->Add(tSigStP);
    tOther->Add(tSigStM);
    tOther->Add(tSigSt0);
    if(bIncludeKSt)
    {
      tOther->Add(tKSt0);
      tOther->Add(tKStP);
    }
  }
  tMCArr->Add(tOther);
  tMCArr->Add(tFake);
  tMCArr->Add(tMaterial);

  //---------------------------------------------
  return tMCArr;
}

//________________________________________________________________________________________________________________
TH1F* GetSumOf2d(TH2F* a2dMC)
{
  TString tSumHistName = TString("tSumHist");
  if     (((TString)a2dMC->GetName()).Contains("Cpa")) tSumHistName += TString("Cpa");
  else if(((TString)a2dMC->GetName()).Contains("Dca")) tSumHistName += TString("Dca");
  else assert(0);

  TObjArray* tMCArr = DoProjections(a2dMC, 10);

  TH1F* tSumHist;
  for(int i=0; i<tMCArr->GetEntries(); i++)
  {
    if(i==0) tSumHist = (TH1F*)tMCArr->At(i)->Clone(tSumHistName);
    else tSumHist->Add((TH1F*)tMCArr->At(i));
  }
  return tSumHist;
}

//________________________________________________________________________________________________________________
TH1F* GetScaledData(TH1F* aData, TH2F* a2dMC)
{
  TString tNameMod;
  if     (((TString)aData->GetName()).Contains("Cpa")) tNameMod = TString("Cpa");
  else if(((TString)aData->GetName()).Contains("Dca")) tNameMod = TString("Dca");
  else assert(0);

  TH1F* tSumHist = GetSumOf2d(a2dMC);

  TH1F* tDataScaled = (TH1F*)aData->Clone(TString::Format("tDataScaled%s", tNameMod.Data()));
//  double tScale = tSumHist->Integral()/aData->Integral();
  double tScale = tSumHist->GetBinContent(tSumHist->GetMaximumBin())/aData->GetBinContent(aData->GetMaximumBin());
  tDataScaled->Scale(tScale);

  return tDataScaled;
}

//________________________________________________________________________________________________________________
void DrawAllIn2d(TCanvas* aCan, TH1F* aData, TH2F* a2dMC, int aNRes, bool tDrawSimpleSum=true)
{
  aCan->cd();
  aCan->SetLogy();
  gStyle->SetOptStat(0);

  TLegend* tLeg = new TLegend(0.15, 0.50, 0.45, 0.90, "", "NDC");
  tLeg->SetBorderSize(1);
  tLeg->SetFillColor(0);

  aData->GetYaxis()->SetRangeUser(1, 100000000);
  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(1);
  aData->SetLineColor(1);
  aData->Draw("Ep");

  TObjArray* tMCArr = DoProjections(a2dMC, aNRes);

  TH1F* tHistToDraw;
  TH1F* tSumHist;
  for(int i=0; i<tMCArr->GetEntries(); i++)
  {
    if(i==0) tSumHist = (TH1F*)tMCArr->At(i)->Clone("tSumHist");
    else tSumHist->Add((TH1F*)tMCArr->At(i));

    tHistToDraw = (TH1F*)tMCArr->At(i);
    if(TString(tHistToDraw->GetName()).Contains("Other")) tHistToDraw->SetLineWidth(4);  //Emphasize Other category in plots
    tHistToDraw->DrawCopy("HISTsame");

    tLeg->AddEntry(tHistToDraw, tHistToDraw->GetName(), "lf");
  }
  tLeg->Draw();

  tSumHist->SetMarkerStyle(20);
  tSumHist->SetMarkerColor(2);
  tSumHist->SetLineColor(2);
  if(tDrawSimpleSum) tSumHist->Draw("Epsame");

  TH1F* tRatio = (TH1F*)aData->Clone("tRatio");
  tRatio->Divide(tSumHist);
  tRatio->SetMarkerStyle(20);
  tRatio->SetMarkerColor(3);
  tRatio->SetLineColor(3);
  tRatio->Draw("Epsame");
}



//________________________________________________________________________________________________________________
void DoTemplateFit(TH1F* aData, TH2F* a2dMC, int aNRes=3)
{
/*
  int tNHist=0;
  if(aNRes==3) tNHist = 7;
  if(aNRes==10) tNHist = 12;
*/

  TString tCanName = TString("tCanTemplateFit");
  TString tNameMod;
  if     (((TString)aData->GetName()).Contains("Cpa")) tNameMod = TString("Cpa");
  else if(((TString)aData->GetName()).Contains("Dca")) tNameMod = TString("Dca");
  else assert(0);
  tCanName += tNameMod;

  TObjArray* tMCArr = DoProjections(a2dMC, aNRes);
  assert(((TH1F*)tMCArr->At(0))->GetXaxis()->GetBinWidth(1)==aData->GetXaxis()->GetBinWidth(1));

  //---------------------------------------------
  TCanvas* tCan = new TCanvas(tCanName, tCanName);
  tCan->cd();
  gStyle->SetOptStat(0);

  TFractionFitter* tFracFit = new TFractionFitter(aData, tMCArr);
  for(int i=0; i<tMCArr->GetEntries(); i++) tFracFit->Constrain(i, 0., 1.);  //Constrain all parameters between 0 and 1
  tFracFit->SetRangeX(1, aData->GetNbinsX());  //set the number of bins to fit

  //-------- Set start values, etc. ----------------------------------------------------------
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(0).SetValue(0.40);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(0).SetLimits(0.1, 1.);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(0).SetName("Primary");

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(1).SetValue(0.25);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(1).SetLimits(0.1, 1.);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(1).SetName("#Sigma^{0}");

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(2).SetValue(0.10);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(2).SetLimits(0.01, 1.);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(2).SetName("#Xi^{0}");

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(3).SetValue(0.15);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(3).SetLimits(0.01, 1.);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(3).SetName("#Xi^{-}");

  int tStartOther = 4;
  if(aNRes==10)
  {
    tStartOther = 5;
    ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(4).SetValue(0.15);
    ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(4).SetLimits(0.01, 1.);
    ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(4).SetName("#Sigma^{*}");
  }

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther).SetValue(0.10);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther).SetLimits(0.0, 0.2);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther).SetName("Other");

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+1).SetValue(0.05);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+1).SetLimits(0.0, 0.2);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+1).SetName("Fake");

  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+2).SetValue(0.01);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+2).SetLimits(0.0, 0.1);
  ((ROOT::Fit::Fitter*)tFracFit->GetFitter())->Config().ParSettings(tStartOther+2).SetName("Material");



  //------------------------------------------------------------------------------------------

  int tStatus = tFracFit->Fit();
  cout << "Fit status: " << tStatus << endl;

  DrawAllIn2d(tCan, aData, a2dMC, aNRes, false);
//  if(tStatus==0)
//  {
    TH1F* tResult = (TH1F*)tFracFit->GetPlot();
    tResult->SetMarkerStyle(30);
    tResult->SetMarkerColor(kGray);
    tResult->SetLineColor(kGray);
    tResult->Draw("Epsame");
//  }

}

//________________________________________________________________________________________________________________
TCanvas* GetDrawAllIn2d(TH1F* aData, TH2F* a2dMC, int aNRes)
{
  TString tCanName = TString("tCanDrawAllIn2d");
  if     (((TString)aData->GetName()).Contains("Cpa")) tCanName += TString("Cpa");
  else if(((TString)aData->GetName()).Contains("Dca")) tCanName += TString("Dca");
  else assert(0);

  TCanvas *tCanDrawAllIn2d = new TCanvas(tCanName, tCanName);
  DrawAllIn2d(tCanDrawAllIn2d, aData, a2dMC, aNRes);

  return tCanDrawAllIn2d;
}

//________________________________________________________________________________________________________________
TCanvas* CompareTotalDataToMC(TH1F* aData, TH2F* a2dMC, TH1F* a1dMCUnbiased)
{
  //It appears that the CPA distribution from MC matches better the shape of the data when 
  //all particles are accepted (as in case of CosPointing histograms) as compared to when
  //only particles with HiddenInfo are selected (as in case of summing histograms in CosPointingAnglewParentInfo)

  TString tCanName = TString("tCanCompTotDataToMC");
  TString tNameMod;
  if     (((TString)aData->GetName()).Contains("Cpa")) tNameMod = TString("Cpa");
  else if(((TString)aData->GetName()).Contains("Dca")) tNameMod = TString("Dca");
  else assert(0);
  tCanName += tNameMod;

  TCanvas *tCanCompTotDataToMC = new TCanvas(tCanName, tCanName);
  tCanCompTotDataToMC->cd();
  tCanCompTotDataToMC->SetLogy();
  gStyle->SetOptStat(0);

  TLegend* tLeg = new TLegend(0.15, 0.50, 0.45, 0.90, "", "NDC");
  tLeg->SetBorderSize(1);
  tLeg->SetFillColor(0);

  aData->GetYaxis()->SetRangeUser(1, 100000000);
  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(1);
  aData->SetLineColor(1);
  aData->Draw("Ep");

  TH1F* tSumHist = GetSumOf2d(a2dMC);
  tSumHist->SetMarkerStyle(20);
  tSumHist->SetMarkerColor(2);
  tSumHist->SetLineColor(2);
  tSumHist->Draw("Epsame");

  a1dMCUnbiased->SetMarkerStyle(20);
  a1dMCUnbiased->SetMarkerColor(4);
  a1dMCUnbiased->SetLineColor(4);
  a1dMCUnbiased->Draw("Epsame");


  TH1F* tRatio = (TH1F*)aData->Clone(TString::Format("tRatio%s", tNameMod.Data()));
  tRatio->Divide(tSumHist);
  tRatio->SetMarkerStyle(20);
  tRatio->SetMarkerColor(kRed+2);
  tRatio->SetLineColor(kRed+2);
  tRatio->Draw("Epsame");

  TH1F* tRatioUnbiased = (TH1F*)aData->Clone(TString::Format("tRatioUnbiased%s", tNameMod.Data()));
  tRatioUnbiased->Divide(a1dMCUnbiased);
  tRatioUnbiased->SetMarkerStyle(20);
  tRatioUnbiased->SetMarkerColor(kBlue+2);
  tRatioUnbiased->SetLineColor(kBlue+2);
  tRatioUnbiased->Draw("Epsame");

  TH1F* tDataScaled = GetScaledData(aData, a2dMC);
  tDataScaled->SetMarkerStyle(24);
  tDataScaled->SetMarkerColor(1);
  tDataScaled->SetLineColor(1);
  tDataScaled->Draw("Epsame");

  tLeg->AddEntry(aData, "Data", "lpf");
  tLeg->AddEntry(tDataScaled, "Data (Scaled)", "lpf");
  tLeg->AddEntry(tSumHist, "MC (Sum)", "lpf");
  tLeg->AddEntry(a1dMCUnbiased, "MC (Unbiased)", "lpf");
  tLeg->AddEntry(tRatio, "Ratio", "lpf");
  tLeg->AddEntry(tRatioUnbiased, "Ratio (Unbiased)", "lpf");
  tLeg->Draw();

  return tCanCompTotDataToMC;
}

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  TString tResultsDate_Data = "20180903_NoCpaOrDcaV0Cuts";
  TString tResultsDate_MC   = "20180903_NoCpaOrDcaV0Cuts";

  AnalysisType tAnType = kLamK0;
  ParticleType tParticleType = kLam; //kLam, kALam, kK0
  CentralityType tCentType = k0010;
  int tRebin = 2;
  int tNRes = 10;  //3 or 10

  bool tSaveImages = false;
  bool tRunCpa = false;
  bool tRunDca = true;
  TString tSaveFileType = "eps";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase_Data = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_Data.Data());
  TString tFileLocationBase_Data = TString::Format("%sResults_%s_%s",tDirectoryBase_Data.Data(),tGeneralAnTypeName.Data(),tResultsDate_Data.Data());
  TString tFileLocation_Data_FemtoMinus = TString::Format("%s_FemtoMinus.root", tFileLocationBase_Data.Data());
  TString tFileLocation_Data_FemtoPlus  = TString::Format("%s_FemtoPlus.root", tFileLocationBase_Data.Data());

  TString tDirectoryBase_MC = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_MC.Data());
  TString tFileLocationBase_MC = TString::Format("%sResults_%sMC_%s",tDirectoryBase_MC.Data(),tGeneralAnTypeName.Data(),tResultsDate_MC.Data());
  TString tFileLocation_MC_FemtoMinus = TString::Format("%s_FemtoMinus.root", tFileLocationBase_MC.Data());
  TString tFileLocation_MC_FemtoPlus  = TString::Format("%s_FemtoPlus.root", tFileLocationBase_MC.Data());

  TString tDirName = TString::Format("%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);

//-----------------------------------------------------------------------------
  if(tRunCpa)
  {
    TString t1dHistNameCpa = TString::Format("CosPointingAngle_%s_Pass", cParticleTags[tParticleType]);
    TString t2dHistNameCpa = TString::Format("CosPointingAnglewParentInfo_%s_Pass", cParticleTags[tParticleType]);
    //-----  
    TH1F* tDataCpa_FemtoMinus = Get1dHist(tFileLocation_Data_FemtoMinus, tDirName, t1dHistNameCpa, t1dHistNameCpa);
    TH1F* tDataCpa_FemtoPlus  = Get1dHist(tFileLocation_Data_FemtoPlus,  tDirName, t1dHistNameCpa, t1dHistNameCpa);
    TH1F* tDataCpa = (TH1F*)tDataCpa_FemtoMinus->Clone(TString::Format("DataCpa_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tDataCpa->Add(tDataCpa_FemtoPlus);
      tDataCpa->Rebin(tRebin);

    TH2F* tMCCpa_FemtoMinus = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistNameCpa, t2dHistNameCpa);
    TH2F* tMCCpa_FemtoPlus  = Get2dHist(tFileLocation_MC_FemtoPlus, tDirName, t2dHistNameCpa, t2dHistNameCpa);
    TH2F* tMCCpa = (TH2F*)tMCCpa_FemtoMinus->Clone(TString::Format("MCCpa_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tMCCpa->Add(tMCCpa_FemtoPlus);
      tMCCpa->RebinX(tRebin);

    TH1F* tMCCpaUnbiased_FemtoMinus = Get1dHist(tFileLocation_MC_FemtoMinus, tDirName, t1dHistNameCpa, t1dHistNameCpa);
    TH1F* tMCCpaUnbiased_FemtoPlus  = Get1dHist(tFileLocation_MC_FemtoPlus, tDirName, t1dHistNameCpa, t1dHistNameCpa);
    TH1F* tMCCpaUnbiased = (TH1F*)tMCCpaUnbiased_FemtoMinus->Clone(TString::Format("MCCpaUnbiased_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tMCCpaUnbiased->Add(tMCCpaUnbiased_FemtoPlus);
      tMCCpaUnbiased->Rebin(tRebin);
    //-----  

//    TH2F* tTest2d = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistNameCpa, t2dHistNameCpa);

//    TCanvas* tCanDrawAllIn2d = GetDrawAllIn2d(tDataCpa, tMCCpa, tNRes);

//    TCanvas* tCompTotDataCpaToMC = CompareTotalDataToMC(tDataCpa, tMCCpa, tMCCpaUnbiased);

/*  
    TH1F* tTestProj = Get1dProjection(tMCCpa, 11);
    TCanvas* tCanTestProj = new TCanvas("tCanTestProj", "tCanTestProj");
    tCanTestProj->cd();
    tCanTestProj->SetLogy();
    tTestProj->Draw();
*/

    DoTemplateFit(tDataCpa, tMCCpa, tNRes);
  }

//-----------------------------------------------------------------------------
  if(tRunDca)
  {
    TString t1dHistNameDca = TString::Format("DcaV0ToPrimVertex_%s_Pass", cParticleTags[tParticleType]);
    TString t2dHistNameDca = TString::Format("DcaV0ToPrimVertexwParentInfo_%s_Pass", cParticleTags[tParticleType]);
    //-----  
    TH1F* tDataDca_FemtoMinus = Get1dHist(tFileLocation_Data_FemtoMinus, tDirName, t1dHistNameDca, t1dHistNameDca);
    TH1F* tDataDca_FemtoPlus  = Get1dHist(tFileLocation_Data_FemtoPlus,  tDirName, t1dHistNameDca, t1dHistNameDca);
    TH1F* tDataDca = (TH1F*)tDataDca_FemtoMinus->Clone(TString::Format("DataDca_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tDataDca->Add(tDataDca_FemtoPlus);
      tDataDca->Rebin(tRebin);

    TH2F* tMCDca_FemtoMinus = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistNameDca, t2dHistNameDca);
    TH2F* tMCDca_FemtoPlus  = Get2dHist(tFileLocation_MC_FemtoPlus, tDirName, t2dHistNameDca, t2dHistNameDca);
    TH2F* tMCDca = (TH2F*)tMCDca_FemtoMinus->Clone(TString::Format("MCDca_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tMCDca->Add(tMCDca_FemtoPlus);
      tMCDca->RebinX(tRebin);

    TH1F* tMCDcaUnbiased_FemtoMinus = Get1dHist(tFileLocation_MC_FemtoMinus, tDirName, t1dHistNameDca, t1dHistNameDca);
    TH1F* tMCDcaUnbiased_FemtoPlus  = Get1dHist(tFileLocation_MC_FemtoPlus, tDirName, t1dHistNameDca, t1dHistNameDca);
    TH1F* tMCDcaUnbiased = (TH1F*)tMCDcaUnbiased_FemtoMinus->Clone(TString::Format("MCDcaUnbiased_%s%s", cParticleTags[tParticleType], cCentralityTags[tCentType]));
      tMCDcaUnbiased->Add(tMCDcaUnbiased_FemtoPlus);
      tMCDcaUnbiased->Rebin(tRebin);
    //-----  

//    TH2F* tTest2d = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistNameDca, t2dHistNameDca);

//    TCanvas* tCanDrawAllIn2d = GetDrawAllIn2d(tDataDca, tMCDca, tNRes);

//    TCanvas* tCompTotDataDcaToMC = CompareTotalDataToMC(tDataDca, tMCDca, tMCDcaUnbiased);

/*  
    TH1F* tTestProj = Get1dProjection(tMCDca, 11);
    TCanvas* tCanTestProj = new TCanvas("tCanTestProj", "tCanTestProj");
    tCanTestProj->cd();
    tCanTestProj->SetLogy();
    tTestProj->Draw();
*/

    DoTemplateFit(tDataDca, tMCDca, tNRes);
  }




//-------------------------------------------------------------------------------

  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
