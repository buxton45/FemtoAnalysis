#include "FitSharedAnalyses.h"
#include "TLegend.h"

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


void MakePretty(TH1* aHisto, int aColor, int aMarkerStyle)
{
  aHisto->SetMarkerStyle(aMarkerStyle);
  aHisto->SetMarkerColor(aColor);
  aHisto->SetLineColor(aColor);

  aHisto->SetMarkerSize(0.50);

  aHisto->GetXaxis()->SetTitle("k* (GeV/c)");
  aHisto->GetYaxis()->SetTitle("C(k*)");
  aHisto->GetYaxis()->SetTitleOffset(1.3);
}

int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  vector<ParameterType> ShareAllButNorm(5);
    ShareAllButNorm[0] = kLambda;
    ShareAllButNorm[1] = kRadius;
    ShareAllButNorm[2] = kRef0;
    ShareAllButNorm[3] = kImf0;
    ShareAllButNorm[4] = kd0;

  vector<int> Share01(2);
    Share01[0] = 0;
    Share01[1] = 1;

//-----------------------------------------------------------------------------

  TString FileLocationBaseNew = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  TString FileLocationBaseOld = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";


  bool bSaveFigures = true;
  TString tSaveFiguresLocation = "~/Analysis/Presentations/AliFemto/20160330/";


  FitPairAnalysis* tLamKchPNew = new FitPairAnalysis(FileLocationBaseNew,kLamKchP,k0010);
    TH1* tCfLamKchPNew = tLamKchPNew->GetKStarCf();
    MakePretty(tCfLamKchPNew,1,20);
  FitPairAnalysis* tLamKchMNew = new FitPairAnalysis(FileLocationBaseNew,kLamKchM,k0010);
    TH1* tCfLamKchMNew = tLamKchMNew->GetKStarCf();
    MakePretty(tCfLamKchMNew,1,20);
  FitPairAnalysis* tALamKchPNew = new FitPairAnalysis(FileLocationBaseNew,kALamKchP,k0010);
    TH1* tCfALamKchPNew = tALamKchPNew->GetKStarCf();
    MakePretty(tCfALamKchPNew,1,20);
  FitPairAnalysis* tALamKchMNew = new FitPairAnalysis(FileLocationBaseNew,kALamKchM,k0010);
    TH1* tCfALamKchMNew = tALamKchMNew->GetKStarCf();
    MakePretty(tCfALamKchMNew,1,20);

  FitPairAnalysis* tLamKchPOld = new FitPairAnalysis(FileLocationBaseOld,kLamKchP,k0010);
    TH1* tCfLamKchPOld = tLamKchPOld->GetKStarCf();
    MakePretty(tCfLamKchPOld,2,22);
  FitPairAnalysis* tLamKchMOld = new FitPairAnalysis(FileLocationBaseOld,kLamKchM,k0010);
    TH1* tCfLamKchMOld = tLamKchMOld->GetKStarCf();
    MakePretty(tCfLamKchMOld,2,22);
  FitPairAnalysis* tALamKchPOld = new FitPairAnalysis(FileLocationBaseOld,kALamKchP,k0010);
    TH1* tCfALamKchPOld = tALamKchPOld->GetKStarCf();
    MakePretty(tCfALamKchPOld,2,22);
  FitPairAnalysis* tALamKchMOld = new FitPairAnalysis(FileLocationBaseOld,kALamKchM,k0010);
    TH1* tCfALamKchMOld = tALamKchMOld->GetKStarCf();
    MakePretty(tCfALamKchMOld,2,22);

//---------------------------------
  TLegend *tLeg = new TLegend(0.65,0.20,0.85,0.35);
    tLeg->AddEntry(tCfLamKchPNew,"MisID Fix","p");
    tLeg->AddEntry(tCfLamKchPOld,"Old","p");

  TPaveText* textLamKchP = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
    textLamKchP->SetFillColor(0);
    textLamKchP->SetTextSize(0.05);
    textLamKchP->SetBorderSize(1);
    textLamKchP->AddText("#LambdaK+");

  TPaveText* textLamKchM = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
    textLamKchM->SetFillColor(0);
    textLamKchM->SetTextSize(0.05);
    textLamKchM->SetBorderSize(1);
    textLamKchM->AddText("#LambdaK-");

  TPaveText* textALamKchP = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
    textALamKchP->SetFillColor(0);
    textALamKchP->SetTextSize(0.05);
    textALamKchP->SetBorderSize(1);
    textALamKchP->AddText("#bar{#Lambda}K+");

  TPaveText* textALamKchM = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
    textALamKchM->SetFillColor(0);
    textALamKchM->SetTextSize(0.05);
    textALamKchM->SetBorderSize(1);
    textALamKchM->AddText("#bar{#Lambda}K-");


//---------------------------------

  TCanvas* tCanLamKchPwConj = new TCanvas("tCanLamKchPwConj","tCanLamKchPwConj");
  tCanLamKchPwConj->Divide(2,1);

  tCanLamKchPwConj->cd(1);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    tCfLamKchPNew->GetXaxis()->SetRangeUser(0.,0.5);
    tCfLamKchPNew->GetYaxis()->SetRangeUser(0.86,1.02);
  tCfLamKchPNew->DrawCopy();
  tCfLamKchPOld->DrawCopy("same");
  tLeg->Draw();
  textLamKchP->Draw();

  tCanLamKchPwConj->cd(2);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    tCfALamKchMNew->GetXaxis()->SetRangeUser(0.,0.5);
    tCfALamKchMNew->GetYaxis()->SetRangeUser(0.86,1.02);
  tCfALamKchMNew->DrawCopy();
  tCfALamKchMOld->DrawCopy("same");
  tLeg->Draw();
  textALamKchM->Draw();

  if(bSaveFigures) tCanLamKchPwConj->SaveAs(tSaveFiguresLocation+"CompareKchFix2_LamKchPwConj.eps");

  TCanvas* tCanLamKchMwConj = new TCanvas("tCanLamKchMwConj","tCanLamKchMwConj");
  tCanLamKchMwConj->Divide(2,1);

  tCanLamKchMwConj->cd(1);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    tCfLamKchMNew->GetXaxis()->SetRangeUser(0.,0.5);
    tCfLamKchMNew->GetYaxis()->SetRangeUser(0.94,1.02);
  tCfLamKchMNew->DrawCopy();
  tCfLamKchMOld->DrawCopy("same");
  tLeg->Draw();
  textLamKchM->Draw();

  tCanLamKchMwConj->cd(2);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    tCfALamKchPNew->GetXaxis()->SetRangeUser(0.,0.5);
    tCfALamKchPNew->GetYaxis()->SetRangeUser(0.94,1.02);
  tCfALamKchPNew->DrawCopy();
  tCfALamKchPOld->DrawCopy("same");
  tLeg->Draw();
  textALamKchP->Draw();

  if(bSaveFigures) tCanLamKchMwConj->SaveAs(tSaveFiguresLocation+"CompareKchFix2_LamKchMwConj.eps");

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
