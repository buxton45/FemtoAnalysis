#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "CanvasPartition.h"

//________________________________________________________________________________________________________________
void NormalizeByTotalEntries(TH2* aHisto)
{
  double tNorm = aHisto->GetEntries();
  aHisto->Scale(1./tNorm);
}
//________________________________________________________________________________________________________________
void NormalizeEachRow(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  for(int j=1; j<=tNbinsY; j++)
  {
    double tScale = aHisto->Integral(1,tNbinsX,j,j);
    if(tScale > 0.)
    {
      for(int i=1; i<=tNbinsX; i++)
      {
        double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
        aHisto->SetBinContent(i,j,tNewContent);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void SetupAxes(TH2* aHist)
{
  aHist->GetXaxis()->SetTitle("#it{k}*_{True} (GeV/#it{c})");
  aHist->GetXaxis()->SetTitleSize(0.035);
  aHist->GetXaxis()->SetTitleOffset(1.2);

  aHist->GetYaxis()->SetTitle("#it{k}*_{Rec.} (GeV/#it{c})");
  aHist->GetYaxis()->SetTitleSize(0.04);
  aHist->GetYaxis()->SetTitleOffset(1.2);
}


//________________________________________________________________________________________________________________
TH2* BuildDiff(TH2* aHist1, TH2* aHist2)
{
  TH2* tHist1Clone = (TH2*)aHist1->Clone(TString::Format("%s_Clone1", aHist1->GetName()));
  TH2* tHist2Clone = (TH2*)aHist2->Clone(TString::Format("%s_Clone2", aHist2->GetName()));

  if(tHist1Clone->GetXaxis()->GetBinWidth(1) != tHist2Clone->GetXaxis()->GetBinWidth(1))
  {
    int tRebin;
    if(tHist1Clone->GetXaxis()->GetBinWidth(1) > tHist2Clone->GetXaxis()->GetBinWidth(1))
    {
      tRebin = int(tHist1Clone->GetXaxis()->GetBinWidth(1)/tHist2Clone->GetXaxis()->GetBinWidth(1));
      tHist2Clone->Rebin2D(tRebin, tRebin);
    }
    else
    {
      tRebin = int(tHist2Clone->GetXaxis()->GetBinWidth(1)/tHist1Clone->GetXaxis()->GetBinWidth(1));
      tHist1Clone->Rebin2D(tRebin, tRebin);
    }
  }

  assert(tHist1Clone->GetXaxis()->GetBinWidth(1) == tHist2Clone->GetXaxis()->GetBinWidth(1));
  TH2D *tFinalHist1, *tFinalHist2;
  if(tHist1Clone->GetNbinsX() != tHist2Clone->GetNbinsX())
  {
    int tNbins;
    double tMaxVal;
    if(tHist1Clone->GetNbinsX() > tHist2Clone->GetNbinsX())
    {
      tNbins = tHist2Clone->GetNbinsX();
      tMaxVal = tHist2Clone->GetXaxis()->GetBinUpEdge(tNbins);
    }
    else
    {
      tNbins = tHist1Clone->GetNbinsX();
      tMaxVal = tHist1Clone->GetXaxis()->GetBinUpEdge(tNbins);
    }
    tFinalHist1 = new TH2D(TString::Format("FinalHist1_%s", aHist1->GetName()),
                                 TString::Format("FinalHist1_%s", aHist1->GetName()),
                                 tNbins, 0., tMaxVal,
                                 tNbins, 0., tMaxVal);
    tFinalHist2 = new TH2D(TString::Format("FinalHist2_%s", aHist2->GetName()),
                                 TString::Format("FinalHist2_%s", aHist2->GetName()),
                                 tNbins, 0., tMaxVal,
                                 tNbins, 0., tMaxVal);
    for(int i=1; i<=tNbins; i++)
    {
      for(int j=1; j<=tNbins; j++)
      {
        tFinalHist1->SetBinContent(i, j, tHist1Clone->GetBinContent(i, j));
        tFinalHist2->SetBinContent(i, j, tHist2Clone->GetBinContent(i, j));
      }
    }
  }
  else
  {
    tFinalHist1 = (TH2D*)tHist1Clone->Clone(TString::Format("FinalHist1_%s", aHist1->GetName()));
    tFinalHist2 = (TH2D*)tHist2Clone->Clone(TString::Format("FinalHist2_%s", aHist2->GetName()));
  }

//  NormalizeEachRow(tFinalHist1);
//  NormalizeEachRow(tFinalHist2);

//  tFinalHist1->Scale(tFinalHist2->Integral()/tFinalHist1->Integral());

  TH2D* tDiff = (TH2D*)tFinalHist1->Clone(TString::Format("Diff_%s_%s", aHist1->GetName(), aHist2->GetName()));
  tDiff->Add(tFinalHist2, -1);
  return tDiff;
}

//________________________________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  bool bSave = false;
  bool bDrawSeparateFigures = false;
  TString tSaveFileType = "pdf";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/5_Fitting/5.3_MomentumResolutionCorrections/Figures/";

  AnalysisType tAnType = kLamKchP;
  CentralityType tCentType = k0010;

//  TString FileLocationBaseMC_1 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMC_RmMisID_20160225";
//  AnalysisRunType tRunType_1=kGrid;
  TString FileLocationBaseMC_1 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/Results_cLamcKchMC_20180505";
  AnalysisRunType tRunType_1=kTrain;
  int tNPartialAnalysis_1=2;
  if(tRunType_1==kGrid) tNPartialAnalysis_1=5;
  Analysis* tAn_1 = new Analysis(FileLocationBaseMC_1,tAnType,tCentType,tRunType_1,tNPartialAnalysis_1);

  TString FileLocationBaseMC_2 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMC_RmMisID_20160225";
  AnalysisRunType tRunType_2=kGrid;
  int tNPartialAnalysis_2=2;
  if(tRunType_2==kGrid) tNPartialAnalysis_2=5;
  Analysis* tAn_2 = new Analysis(FileLocationBaseMC_2,tAnType,tCentType,tRunType_2,tNPartialAnalysis_2);

  TString FileLocationBaseMC_3 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMC_20160224";
  AnalysisRunType tRunType_3=kGrid;
  int tNPartialAnalysis_3=1;
  Analysis* tAn_3 = new Analysis(FileLocationBaseMC_3,tAnType,tCentType,tRunType_3,tNPartialAnalysis_3);

//-----------------------------------------------------------------------------
  tAn_1->BuildModelKStarTrueVsRecTotal(kSame);
  tAn_1->BuildModelKStarTrueVsRecTotal(kMixed);

  TH2* LamKchPTrueVsRecRotSame_1 = tAn_1->GetModelKStarTrueVsRecTotal(kSame);
  TH2* LamKchPTrueVsRecRotMixed_1 = tAn_1->GetModelKStarTrueVsRecTotal(kMixed);

  TCanvas *canKStarTrueVsRec_1 = new TCanvas("canKStarTrueVsRec_1","canKStarTrueVsRec_1");
  canKStarTrueVsRec_1->Divide(2,1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  canKStarTrueVsRec_1->cd(1);
    gPad->SetLogz();
    LamKchPTrueVsRecRotSame_1->Draw("colz");

  canKStarTrueVsRec_1->cd(2);
    gPad->SetLogz();
    LamKchPTrueVsRecRotMixed_1->Draw("colz");

//-----------------------------------------------------------------------------
  tAn_2->BuildModelKStarTrueVsRecTotal(kSame);
  tAn_2->BuildModelKStarTrueVsRecTotal(kMixed);

  TH2* LamKchPTrueVsRecRotSame_2 = tAn_2->GetModelKStarTrueVsRecTotal(kSame);
  TH2* LamKchPTrueVsRecRotMixed_2 = tAn_2->GetModelKStarTrueVsRecTotal(kMixed);

  TCanvas *canKStarTrueVsRec_2 = new TCanvas("canKStarTrueVsRec_2","canKStarTrueVsRec_2");
  canKStarTrueVsRec_2->Divide(2,1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  canKStarTrueVsRec_2->cd(1);
    gPad->SetLogz();
    LamKchPTrueVsRecRotSame_2->Draw("colz");

  canKStarTrueVsRec_2->cd(2);
    gPad->SetLogz();
    LamKchPTrueVsRecRotMixed_2->Draw("colz");


//-----------------------------------------------------------------------------
  tAn_3->BuildModelKStarTrueVsRecTotal(kSame);
  tAn_3->BuildModelKStarTrueVsRecTotal(kMixed);

  TH2* LamKchPTrueVsRecRotSame_3 = tAn_3->GetModelKStarTrueVsRecTotal(kSame);
  TH2* LamKchPTrueVsRecRotMixed_3 = tAn_3->GetModelKStarTrueVsRecTotal(kMixed);

  TCanvas *canKStarTrueVsRec_3 = new TCanvas("canKStarTrueVsRec_3","canKStarTrueVsRec_3");
  canKStarTrueVsRec_3->Divide(2,1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  canKStarTrueVsRec_3->cd(1);
    gPad->SetLogz();
    LamKchPTrueVsRecRotSame_3->Draw("colz");

  canKStarTrueVsRec_3->cd(2);
    gPad->SetLogz();
    LamKchPTrueVsRecRotMixed_3->Draw("colz");
//-------------------------------------------------------------------------------
/*
  TCanvas* tCanDiff32 = new TCanvas("tCanDiff32", "tCanDiff32");
  tCanDiff32->cd();
    gPad->SetLogz();

  TH2* tDiff32 = BuildDiff(LamKchPTrueVsRecRotSame_3, LamKchPTrueVsRecRotSame_2);
  tDiff32->Draw("colz");

  TCanvas* tCanDiff12 = new TCanvas("tCanDiff12", "tCanDiff12");
  tCanDiff12->cd();
    gPad->SetLogz();

  TH2* tDiff12 = BuildDiff(LamKchPTrueVsRecRotSame_1, LamKchPTrueVsRecRotSame_2);
  tDiff12->Draw("colz");

  TCanvas* tCanDiff31 = new TCanvas("tCanDiff31", "tCanDiff31");
  tCanDiff31->cd();
    gPad->SetLogz();

  TH2* tDiff31 = BuildDiff(LamKchPTrueVsRecRotSame_3, LamKchPTrueVsRecRotSame_1);
  tDiff31->Draw("colz");

*/

//-----------------------FINAL FIGURE-------------------------------------------------------
  TH2* tDiff12 = BuildDiff(LamKchPTrueVsRecRotSame_1, LamKchPTrueVsRecRotSame_2);

  double tZMin = 0.1;
  double tZMax = 50000;

  SetupAxes(LamKchPTrueVsRecRotSame_1);
  SetupAxes(LamKchPTrueVsRecRotSame_2);
  SetupAxes(tDiff12);

  LamKchPTrueVsRecRotSame_1->Scale(LamKchPTrueVsRecRotSame_2->Integral()/LamKchPTrueVsRecRotSame_1->Integral());

  LamKchPTrueVsRecRotSame_1->GetZaxis()->SetRangeUser(tZMin, tZMax);
  LamKchPTrueVsRecRotSame_2->GetZaxis()->SetRangeUser(tZMin, tZMax);

  LamKchPTrueVsRecRotSame_1->GetXaxis()->SetRangeUser(0., 1.);
  LamKchPTrueVsRecRotSame_1->GetYaxis()->SetRangeUser(0., 1.);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.05);

  TCanvas *tFinalCan, *tFinalCan2;
  if(!bDrawSeparateFigures)
  {
    tFinalCan = new TCanvas("tFinalCan", "tFinalCan", 2100, 500);
    tFinalCan->Divide(3,1);
      gStyle->SetOptStat(0);
      gStyle->SetOptTitle(0);
    tFinalCan->cd(1);
      gPad->SetLogz();
    LamKchPTrueVsRecRotSame_1->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Analysis Cuts (A)");

    tFinalCan->cd(2);
      gPad->SetLogz();
    LamKchPTrueVsRecRotSame_2->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Remove all misidentified (B)");

    tFinalCan->cd(3);
      gPad->SetLogz();
    tDiff12->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Difference (A-B)");
  }
  else
  {
    tFinalCan = new TCanvas("tFinalCan", "tFinalCan", 1400, 500);
    tFinalCan->Divide(2,1);
      gStyle->SetOptStat(0);
      gStyle->SetOptTitle(0);
    tFinalCan->cd(1);
      gPad->SetLogz();
    LamKchPTrueVsRecRotSame_1->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Analysis Cuts (A)");

    tFinalCan->cd(2);
      gPad->SetLogz();
    LamKchPTrueVsRecRotSame_2->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Remove all misidentified (B)");


    tFinalCan2 = new TCanvas("tFinalCan2", "tFinalCan2", 700, 500);
    tFinalCan2->cd();
      gStyle->SetOptStat(0);
      gStyle->SetOptTitle(0);
      gPad->SetLogz();
    tDiff12->Draw("colz");
    tTex->DrawLatex(0.025, 0.95, TString::Format("%s (%s)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[tCentType]));
    tTex->SetTextSize(0.045);
    tTex->DrawLatex(0.025, 0.85, "Difference (A-B)");
  }

//-------------------------------------------------------------------------------

  if(bSave)
  {
    if(!bDrawSeparateFigures)
    {
      tFinalCan->SaveAs(TString::Format("%sMomResMatrix%s%s.%s", tSaveDir.Data(), cAnalysisBaseTags[tAnType], cCentralityTags[tCentType], tSaveFileType.Data()));
    }
    else
    {
      tFinalCan->SaveAs(TString::Format("%sMomResMatrixSep1%s%s.%s", tSaveDir.Data(), cAnalysisBaseTags[tAnType], cCentralityTags[tCentType], tSaveFileType.Data()));
      tFinalCan2->SaveAs(TString::Format("%sMomResMatrixSep2%s%s.%s", tSaveDir.Data(), cAnalysisBaseTags[tAnType], cCentralityTags[tCentType], tSaveFileType.Data()));
    }
  }

//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
