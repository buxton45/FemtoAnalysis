#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

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
TH1F* Get1dProjection(TH2F* a2dHist, TString aBinName)
{
  int tBinNumber = -1;
  for(int i=1; i<=a2dHist->GetNbinsY(); i++) if(aBinName.EqualTo(a2dHist->GetYaxis()->GetBinLabel(i))) tBinNumber=i;
  assert(tBinNumber > 0);

  TString tNewName = TString::Format("%s CPA", aBinName.Data());
  TH1F* tReturnHist = (TH1F*)a2dHist->ProjectionX(tNewName, tBinNumber, tBinNumber);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1F* Get1dProjection(TH2F* a2dHist, int aDefaultBin)
{
  TString tBinName = tDefaultBins[aDefaultBin];
  return Get1dProjection(a2dHist, tBinName);
}

//________________________________________________________________________________________________________________
TObjArray* DoProjections(TH2F* a2dMC, int aNRes=3)
{
  //First, grab all MC hisograms
  TH1F* tPrimary  = Get1dProjection(a2dMC, 0);
  TH1F* tLambda   = Get1dProjection(a2dMC, 1);
  TH1F* tK0s      = Get1dProjection(a2dMC, 2);
  TH1F* tSig0     = Get1dProjection(a2dMC, 3);
  TH1F* tXi0      = Get1dProjection(a2dMC, 4);
  TH1F* tXiC      = Get1dProjection(a2dMC, 5);
  TH1F* tSigStP   = Get1dProjection(a2dMC, 6);
  TH1F* tSigStM   = Get1dProjection(a2dMC, 7);
  TH1F* tSigSt0   = Get1dProjection(a2dMC, 8);
  TH1F* tKSt0     = Get1dProjection(a2dMC, 9);
  TH1F* tKStP     = Get1dProjection(a2dMC, 10);
  TH1F* tOther    = Get1dProjection(a2dMC, 11);
  TH1F* tFake     = Get1dProjection(a2dMC, 12);
  TH1F* tMaterial = Get1dProjection(a2dMC, 13);

  //---------------------------------------------

  TObjArray* tMCArr = new TObjArray();
  tMCArr->Add(tPrimary);
  tMCArr->Add(tSig0);
  tMCArr->Add(tXi0);
  tMCArr->Add(tXiC);
  if(aNRes==10)
  {
    tMCArr->Add(tSigStP);
    tMCArr->Add(tSigStM);
    tMCArr->Add(tSigSt0);
    tMCArr->Add(tKSt0);
    tMCArr->Add(tKStP);
  }
  else
  {
    tOther->Add(tSigStP);
    tOther->Add(tSigStM);
    tOther->Add(tSigSt0);
    tOther->Add(tKSt0);
    tOther->Add(tKStP);
  }
  tMCArr->Add(tOther);
  tMCArr->Add(tFake);
  tMCArr->Add(tMaterial);

  //---------------------------------------------
  return tMCArr;
}


//________________________________________________________________________________________________________________
void DoTemplateFit(TH1F* aData, TH2F* a2dMC, int aNRes=3)
{
/*
  int tNHist=0;
  if(aNRes==3) tNHist = 7;
  if(aNRes==10) tNHist = 12;
*/

  TObjArray* tMCArr = DoProjections(a2dMC, aNRes);
  assert(((TH1F*)tMCArr->At(0))->GetXaxis()->GetBinWidth(1)==aData->GetXaxis()->GetBinWidth(1));

  //---------------------------------------------
  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();
  gStyle->SetOptStat(0);

  TFractionFitter* tFracFit = new TFractionFitter(aData, tMCArr);
  for(int i=0; i<tMCArr->GetEntries(); i++) tFracFit->Constrain(i, 0., 1.);  //Constrain all parameters between 0 and 1
  tFracFit->SetRangeX(1, aData->GetNbinsX());  //set the number of bins to fit
  int tStatus = tFracFit->Fit();
  cout << "Fit status: " << tStatus << endl;
  if(tStatus==0)
  {
    TH1F* tResult = (TH1F*)tFracFit->GetPlot();
    aData->Draw("Ep");
    tResult->Draw("same");
  }

}

//________________________________________________________________________________________________________________
TH1F* GetSumOf2d(TH2F* a2dMC)
{
  TObjArray* tMCArr = DoProjections(a2dMC, 10);

  TH1F* tSumHist;
  for(int i=0; i<tMCArr->GetEntries(); i++)
  {
    if(i==0) tSumHist = (TH1F*)tMCArr->At(i)->Clone("tSumHist");
    else tSumHist->Add((TH1F*)tMCArr->At(i));
  }
  return tSumHist;
}


//________________________________________________________________________________________________________________
TCanvas* DrawAllIn2d(TH1F* aData, TH2F* a2dMC, int aNRes)
{
  TObjArray* tMCArr = DoProjections(a2dMC, aNRes);
  TCanvas *tCanDrawAllIn2d = new TCanvas("tCanDrawAllIn2d", "tCanDrawAllIn2d");
  tCanDrawAllIn2d->cd();
  tCanDrawAllIn2d->SetLogy();
  gStyle->SetOptStat(0);

  vector<int> tColors{1, 2, 3, 4, 5, 6, 7, 8, 9, 28, 46, 41};
//  vector<int> tFillStyles{0, 0, 0, 0, 3305, 3325, 3352, 3395, 3644, 3481, 3418, 3357};
  vector<int> tFillStyles{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  TLegend* tLeg = new TLegend(0.15, 0.50, 0.45, 0.90, "", "NDC");
  tLeg->SetBorderSize(1);
  tLeg->SetFillColor(0);

  aData->GetYaxis()->SetRangeUser(1, 100000000);
  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(1);
  aData->SetLineColor(1);
  aData->Draw("Ep");


  TH1F* tHistToDraw;
  TH1F* tSumHist;
  for(int i=0; i<tMCArr->GetEntries(); i++)
  {
    if(i==0) tSumHist = (TH1F*)tMCArr->At(i)->Clone("tSumHist");
    else tSumHist->Add((TH1F*)tMCArr->At(i));

    tHistToDraw = (TH1F*)tMCArr->At(i);
    tHistToDraw->Rebin(2);
    tHistToDraw->SetMarkerStyle(20);
    tHistToDraw->SetMarkerColor(tColors[i]);
    tHistToDraw->SetLineColor(tColors[i]);
    tHistToDraw->SetLineWidth(2);
    tHistToDraw->SetFillStyle(tFillStyles[i]);
    tHistToDraw->SetFillColor(tColors[i]);
    tHistToDraw->DrawCopy("HISTsame");

    tLeg->AddEntry(tHistToDraw, tHistToDraw->GetName(), "lpf");
  }
  tLeg->Draw();

  tSumHist->SetMarkerStyle(20);
  tSumHist->SetMarkerColor(2);
  tSumHist->SetLineColor(2);
  tSumHist->Draw("Epsame");

  TH1F* tRatio = (TH1F*)aData->Clone("tRatio");
  tRatio->Divide(tSumHist);
  tRatio->SetMarkerStyle(20);
  tRatio->SetMarkerColor(3);
  tRatio->SetLineColor(3);
  tRatio->Draw("Epsame");

  return tCanDrawAllIn2d;
}

//________________________________________________________________________________________________________________
TCanvas* CompareTotalDataToMC(TH1F* aData, TH2F* a2dMC, TH1F* a1dMCUnbiased)
{
  //It appears that the CPA distribution from MC matches better the shape of the data when 
  //all particles are accepted (as in case of CosPointing histograms) as compared to when
  //only particles with HiddenInfo are selected (as in case of summing histograms in CosPointingAnglewParentInfo)

  TCanvas *tCanCompTotDataToMC = new TCanvas("tCanCompTotDataToMC", "tCanCompTotDataToMC");
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


  TH1F* tRatio = (TH1F*)aData->Clone("tRatio");
  tRatio->Divide(tSumHist);
  tRatio->SetMarkerStyle(20);
  tRatio->SetMarkerColor(kRed+2);
  tRatio->SetLineColor(kRed+2);
  tRatio->Draw("Epsame");

  TH1F* tRatioUnbiased = (TH1F*)aData->Clone("tRatioUnbiased");
  tRatioUnbiased->Divide(a1dMCUnbiased);
  tRatioUnbiased->SetMarkerStyle(20);
  tRatioUnbiased->SetMarkerColor(kBlue+2);
  tRatioUnbiased->SetLineColor(kBlue+2);
  tRatioUnbiased->Draw("Epsame");

  tLeg->AddEntry(aData, "Data", "lpf");
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

  TString tResultsDate_Data = "20180830_NoCpaCut";
  TString tResultsDate_MC   = "20180830_NoCpaCut";

  AnalysisType tAnType = kLamK0;
  ParticleType tParticleType = kLam; //kLam, kALam, kK0
  CentralityType tCentType = k0010;

  bool SaveImages = false;
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

//-----------------------------------------------------------------------------

  TString t1dHistName = TString::Format("CosPointingAngle_%s_Pass", cParticleTags[tParticleType]);
  TString t2dHistName = TString::Format("CosPointingAnglewParentInfo_%s_Pass", cParticleTags[tParticleType]);
  TString tDirName = TString::Format("%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TH1F* tData_FemtoMinus = Get1dHist(tFileLocation_Data_FemtoMinus, tDirName, t1dHistName, t1dHistName);
  TH1F* tData_FemtoPlus  = Get1dHist(tFileLocation_Data_FemtoPlus,  tDirName, t1dHistName, t1dHistName);
  TH1F* tData = (TH1F*)tData_FemtoMinus->Clone(TString::Format("Data_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]));
    tData->Add(tData_FemtoPlus);

  TH2F* tMC_FemtoMinus = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistName, t2dHistName);
  TH2F* tMC_FemtoPlus  = Get2dHist(tFileLocation_MC_FemtoPlus, tDirName, t2dHistName, t2dHistName);
  TH2F* tMC = (TH2F*)tMC_FemtoMinus->Clone(TString::Format("MC_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]));
    tMC->Add(tMC_FemtoPlus);

  TH1F* tMCUnbiased_FemtoMinus = Get1dHist(tFileLocation_MC_FemtoMinus, tDirName, t1dHistName, t1dHistName);
  TH1F* tMCUnbiased_FemtoPlus  = Get1dHist(tFileLocation_MC_FemtoPlus, tDirName, t1dHistName, t1dHistName);
  TH1F* tMCUnbiased = (TH1F*)tMCUnbiased_FemtoMinus->Clone(TString::Format("MCUnbiased_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]));
    tMCUnbiased->Add(tMCUnbiased_FemtoPlus);

//  TH2F* tTest2d = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistName, t2dHistName);

  TCanvas* tCanDrawAllIn2d = DrawAllIn2d(tData, tMC, 10);

  TCanvas* tCompTotDataToMC = CompareTotalDataToMC(tData, tMC, tMCUnbiased);

/*  
  TH1F* tTestProj = Get1dProjection(tMC, 11);
  TCanvas* tCanTestProj = new TCanvas("tCanTestProj", "tCanTestProj");
  tCanTestProj->cd();
  tCanTestProj->SetLogy();
  tTestProj->Draw();
*/

//  DoTemplateFit(tData, tMC, 3);

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
