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
#include "TLegendEntry.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "DataAndModel.h"
class DataAndModel;


Analysis* CombineMCAnalyses(Analysis* aAnalysis1, Analysis* aAnalysis2, TString aName)
{
  vector<PartialAnalysis*> tVecTot;
  vector<PartialAnalysis*> tVec1 = aAnalysis1->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tVec2 = aAnalysis2->GetPartialAnalysisCollection();
  assert(tVec1.size() == tVec2.size());
  for(unsigned int i=0; i<tVec1.size(); i++)
  {
    tVecTot.push_back(tVec1[i]);
    tVecTot.push_back(tVec2[i]);
  }


  Analysis* aReturnAnalysis = new Analysis(aName,tVecTot);

  return aReturnAnalysis;
}


//________________________________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  //-----Data
  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  //TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_As";


  //-----MC
  //TString FileLocationBaseMC = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229";
  //TString FileLocationBaseMCd = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMCd_KchAndLamFix2_20160229";

  TString FileLocationBaseMC = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMC_RmMisID_20160225";
  TString FileLocationBaseMCd = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMCd_RmMisID_20160225";

  TString FileLocationBaseMCFine = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_RmMisID_20160225/Results_cLamcKch_AsRcMC_RmMisIDFine_20160225";

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!
//-----------------------------------------------------------------------------


  Analysis* LamKchP = new Analysis(FileLocationBase,kLamKchP,k0010);
  Analysis* LamKchPMC = new Analysis(FileLocationBaseMC,kLamKchP,k0010);
//  Analysis* LamKchPMCd = new Analysis(FileLocationBaseMCd,kLamKchP,k0010);
//  Analysis* LamKchPMCTot = CombineMCAnalyses(LamKchPMC,LamKchPMCd,"LamKchPMCTot_0010");
//  DataAndModel* tDataAndModel = new DataAndModel(LamKchP,LamKchPMCTot,0.32,0.40,2);
  DataAndModel* tDataAndModel = new DataAndModel(LamKchP,LamKchPMC,0.32,0.40,2);

  Analysis* LamKchP2 = new Analysis(FileLocationBase,kLamKchP,k0010);
  Analysis* LamKchPMCFine = new Analysis(FileLocationBaseMCFine,kLamKchP,k0010);
  DataAndModel* tDataAndModelFine = new DataAndModel(LamKchP2,LamKchPMCFine,0.32,0.40,1);

/*
  Analysis* ALamKchP = new Analysis(FileLocationBase,kALamKchP,k0010);
  Analysis* ALamKchPMC = new Analysis(FileLocationBaseMC,kALamKchP,k0010);
//  Analysis* ALamKchPMCd = new Analysis(FileLocationBaseMCd,kALamKchP,k0010);
//  Analysis* ALamKchPMCTot = CombineMCAnalyses(ALamKchPMC,ALamKchPMCd,"ALamKchPMCTot_0010");
//  DataAndModel* tDataAndModel = new DataAndModel(ALamKchP,ALamKchPMCTot,0.32,0.40,2);
  DataAndModel* tDataAndModel = new DataAndModel(ALamKchP,ALamKchPMC,0.32,0.40,2);

  Analysis* ALamKchP2 = new Analysis(FileLocationBase,kALamKchP,k0010);
  Analysis* ALamKchPMCFine = new Analysis(FileLocationBaseMCFine,kALamKchP,k0010);
  DataAndModel* tDataAndModelFine = new DataAndModel(ALamKchP2,ALamKchPMCFine,0.32,0.40,1);
*/
/*
  Analysis* LamKchM = new Analysis(FileLocationBase,kLamKchM,k0010);
  Analysis* LamKchMMC = new Analysis(FileLocationBaseMC,kLamKchM,k0010);
//  Analysis* LamKchMMCd = new Analysis(FileLocationBaseMCd,kLamKchM,k0010);
//  Analysis* LamKchMMCTot = CombineMCAnalyses(LamKchMMC,LamKchMMCd,"LamKchMMCTot_0010");
//  DataAndModel* tDataAndModel = new DataAndModel(LamKchM,LamKchMMCTot,0.32,0.40,2);
  DataAndModel* tDataAndModel = new DataAndModel(LamKchM,LamKchMMC,0.32,0.40,2);

  Analysis* LamKchM2 = new Analysis(FileLocationBase,kLamKchM,k0010);
  Analysis* LamKchMMCFine = new Analysis(FileLocationBaseMCFine,kLamKchM,k0010);
  DataAndModel* tDataAndModelFine = new DataAndModel(LamKchM2,LamKchMMCFine,0.32,0.40,1);
*/
/*
  Analysis* ALamKchM = new Analysis(FileLocationBase,kALamKchM,k0010);
  Analysis* ALamKchMMC = new Analysis(FileLocationBaseMC,kALamKchM,k0010);
//  Analysis* ALamKchMMCd = new Analysis(FileLocationBaseMCd,kALamKchM,k0010);
//  Analysis* ALamKchMMCTot = CombineMCAnalyses(ALamKchMMC,ALamKchMMCd,"ALamKchMMCTot_0010");
//  DataAndModel* tDataAndModel = new DataAndModel(ALamKchM,ALamKchMMCTot,0.32,0.40,2);
  DataAndModel* tDataAndModel = new DataAndModel(ALamKchM,ALamKchMMC,0.32,0.40,2);

  Analysis* ALamKchM2 = new Analysis(FileLocationBase,kALamKchM,k0010);
  Analysis* ALamKchMMCFine = new Analysis(FileLocationBaseMCFine,kALamKchM,k0010);
  DataAndModel* tDataAndModelFine = new DataAndModel(ALamKchM2,ALamKchMMCFine,0.32,0.40,1);
*/


//-----------------------------------------------------------------------------
//--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!--**!!
//-----------------------------------------------------------------------------

  bool bDrawMatrices = true;
  bool bCompareMatrices = false;
  bool bDrawCorrectionFactors = false;
  bool bDrawCorrectedCfs = false;
  bool bCompareFine = false;

  bool bSaveFigures = false;
//  TString tSaveFiguresLocation = "~/Analysis/Presentations/Group Meetings/20160317/";
  TString tSaveFiguresLocation = "~/Analysis/Presentations/AliFemto/20160330/";

  if(bDrawMatrices)
  {
    tDataAndModel->GetAnalysisModel()->BuildAllModelKStarTrueVsRecTotal();
    //----------------------
    TH2* tMomResMatrixMixed = tDataAndModel->GetAnalysisModel()->GetModelKStarTrueVsRecTotal(kMixed);

    tMomResMatrixMixed->GetXaxis()->SetTitle("k*_{true}");
    tMomResMatrixMixed->GetYaxis()->SetTitle("k*_{rec}");

    TCanvas* tCanMomResMixed = new TCanvas("tCanMomResMixed","tCanMomResMixed");
    tCanMomResMixed->cd();
    gPad->SetLogz();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    tMomResMatrixMixed->Draw("colz");

    if(bSaveFigures) tCanMomResMixed->SaveAs(tSaveFiguresLocation+"MomResMatrixMixed_"+TString(cAnalysisBaseTags[tDataAndModel->GetAnalysisType()])+".eps");


    //----------------------
    TH2* tMomResMatrixSame = tDataAndModel->GetAnalysisModel()->GetModelKStarTrueVsRecTotal(kSame);

    tMomResMatrixSame->GetXaxis()->SetTitle("k*_{true}");
    tMomResMatrixSame->GetYaxis()->SetTitle("k*_{rec}");

    TCanvas* tCanMomResSame = new TCanvas("tCanMomResSame","tCanMomResSame");
    tCanMomResSame->cd();
    gPad->SetLogz();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    tMomResMatrixSame->Draw("colz");

    if(bSaveFigures) tCanMomResSame->SaveAs(tSaveFiguresLocation+"MomResMatrixSame_"+TString(cAnalysisBaseTags[tDataAndModel->GetAnalysisType()])+".eps");
  }

  if(bCompareMatrices)
  {
    tDataAndModel->GetAnalysisModel()->BuildAllModelKStarTrueVsRecTotal();

    TH2* tKTrueVsRecSame = tDataAndModel->GetAnalysisModel()->GetModelKStarTrueVsRecTotal(kSame);
    TH2* tKTrueVsRecMixed = tDataAndModel->GetAnalysisModel()->GetModelKStarTrueVsRecTotal(kMixed);

    tDataAndModel->GetAnalysisModel()->NormalizeTH2EachRow(tKTrueVsRecSame);
    tDataAndModel->GetAnalysisModel()->NormalizeTH2EachRow(tKTrueVsRecMixed);

    TCanvas* tCanCompMatrices = new TCanvas("tCanCompMatrices","tCanCompMatrices");
    tCanCompMatrices->Divide(2,1);

    tCanCompMatrices->cd(1);
    gPad->SetLogz();
    tKTrueVsRecSame->GetZaxis()->SetRangeUser(0.0001,1);
    tKTrueVsRecSame->DrawCopy("colz");

    tCanCompMatrices->cd(2);
    gPad->SetLogz();
    tKTrueVsRecMixed->GetZaxis()->SetRangeUser(0.0001,1);
    tKTrueVsRecMixed->DrawCopy("colz");

    //-----------------------------------------------
    assert(tKTrueVsRecSame->GetNbinsX() == tKTrueVsRecMixed->GetNbinsX());
    assert(tKTrueVsRecSame->GetNbinsY() == tKTrueVsRecMixed->GetNbinsY());

    for(int j=1; j<=tKTrueVsRecSame->GetNbinsY(); j++)
    {
      assert(tKTrueVsRecSame->GetYaxis()->GetBinCenter(j) == tKTrueVsRecMixed->GetYaxis()->GetBinCenter(j));
      cout << "----------------------------------------------------------------------------------------------------" << endl;
      cout << "j = " << j << endl;
      for(int i=1; i<=tKTrueVsRecSame->GetNbinsX(); i++)
      {
        assert(tKTrueVsRecSame->GetXaxis()->GetBinCenter(i) == tKTrueVsRecMixed->GetXaxis()->GetBinCenter(i));
        if(tKTrueVsRecSame->GetBinContent(i,j) > 0. && tKTrueVsRecMixed->GetBinContent(i,j) > 0.)
        {
          double tSameContent = tKTrueVsRecSame->GetBinContent(i,j);
          double tMixedContent = tKTrueVsRecMixed->GetBinContent(i,j);
          double tDiff = tMixedContent-tSameContent;
          cout << "i = " << i << endl;
          cout << "Same content = " << tSameContent << endl;
          cout << "Mixed content = " << tMixedContent << endl;
          cout << "Diff (Mixed-Same) = " << tDiff << endl;
          if(fabs(tDiff) > 0.01) cout << "Diff > 0.01 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \t\t\t\t\t !!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
          cout << endl;
        } 
        
      }

    }

  }

  if(bDrawCorrectionFactors)
  {
    TH1D* tCfCorrectionwMatrix = tDataAndModel->GetKStarCorrectedwMatrix(0,kMixed,0.,1.,true);
    TH1D* tCfCorrectionwMatrixNumDenSmeared = tDataAndModel->GetKStarCorrectedwMatrixNumDenSmeared(kMixed,0.,0.1,true);
    TH1D* tCfCorrectionwMatrixFit = tDataAndModel->GetKStarCorrectedwMatrix(1,kRotMixed,0.,0.1,true);

  TH1D* tFakeCorrectionHist = (TH1D*)tDataAndModel->GetFakeCorrectionHist();
    tFakeCorrectionHist->SetMarkerStyle(20);
    tFakeCorrectionHist->SetMarkerColor(1);
    tFakeCorrectionHist->SetLineColor(1);
    for(int i=1; i<= tFakeCorrectionHist->GetNbinsX(); i++) tFakeCorrectionHist->SetBinError(i,0.00001);


    TCanvas *tCanAllCorrections = new TCanvas("tCanAllCorrections","tCanAllCorrections");
    tCanAllCorrections->cd();
    gStyle->SetOptStat(0);

    tCfCorrectionwMatrix->GetXaxis()->SetRangeUser(0.,1);
    tCfCorrectionwMatrix->GetYaxis()->SetRangeUser(0.96,1.02);

    tCfCorrectionwMatrix->DrawCopy();
    tCfCorrectionwMatrixNumDenSmeared->DrawCopy("same");
    tCfCorrectionwMatrixFit->DrawCopy("same");
    tFakeCorrectionHist->DrawCopy("same");
    tCfCorrectionwMatrix->DrawCopy("same");

    TLegend *tLeg = new TLegend(0.65,0.15,0.85,0.35);
      tLeg->SetHeader("Correction Factors");
      tLeg->AddEntry(tCfCorrectionwMatrix,"Matrix","p");
      tLeg->AddEntry(tCfCorrectionwMatrixNumDenSmeared,"Smeared","p");
      tLeg->AddEntry(tCfCorrectionwMatrixFit,"MatrixFit","p");
      tLeg->AddEntry(tFakeCorrectionHist,"FakeHist","p");
      tLeg->Draw();

    TLine *line = new TLine(0,1,0.1,1);
    line->SetLineColor(14);
    line->Draw();

    if(bSaveFigures) tCanAllCorrections->SaveAs(tSaveFiguresLocation+"CorrectionFactors_"+TString(cAnalysisBaseTags[tDataAndModel->GetAnalysisType()])+".eps");

  }

  if(bCompareFine)
  {
    TH1D* tCfCorrectionwMatrix = tDataAndModel->GetKStarCorrectedwMatrix(0,kMixed,0.,1.,true);
     tCfCorrectionwMatrix->SetMarkerStyle(22);
     tCfCorrectionwMatrix->SetMarkerColor(kGray+1);
     tCfCorrectionwMatrix->SetLineColor(kGray+1);

  TH1D* tFakeCorrectionHist = (TH1D*)tDataAndModel->GetFakeCorrectionHist();
    tFakeCorrectionHist->SetMarkerStyle(20);
    tFakeCorrectionHist->SetMarkerColor(1);
    tFakeCorrectionHist->SetLineColor(1);

    //-----------------------------------------------
    TH1D* tCfCorrectionwMatrixFine = tDataAndModelFine->GetKStarCorrectedwMatrix(0,kMixed,0.,1.,true);
    tCfCorrectionwMatrixFine->SetMarkerStyle(22);
    tCfCorrectionwMatrixFine->SetMarkerColor(kRed-9);
    tCfCorrectionwMatrixFine->SetLineColor(kRed-9);

    tDataAndModelFine->GetAnalysisModel()->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
  TH1D* tFakeCorrectionHistFine = (TH1D*)tDataAndModelFine->GetAnalysisModel()->GetModelCfFakeIdealCfFakeRatio();
//  TH1D* tFakeCorrectionHistFine = (TH1D*)tDataAndModelFine->GetFakeCorrectionHist();
    tFakeCorrectionHistFine->SetMarkerStyle(20);
    tFakeCorrectionHistFine->SetMarkerColor(2);
    tFakeCorrectionHistFine->SetLineColor(2);

    //-----------------------------------------------
    for(int i=1; i<=tCfCorrectionwMatrix->GetNbinsX(); i++) tCfCorrectionwMatrix->SetBinError(i,0.00001);
    for(int i=1; i<= tFakeCorrectionHist->GetNbinsX(); i++) tFakeCorrectionHist->SetBinError(i,0.00001);
    for(int i=1; i<=tCfCorrectionwMatrixFine->GetNbinsX(); i++) tCfCorrectionwMatrixFine->SetBinError(i,0.00001);
    for(int i=1; i<= tFakeCorrectionHistFine->GetNbinsX(); i++) tFakeCorrectionHistFine->SetBinError(i,0.00001);
    //-----------------------------------------------


    TCanvas *tCanAllCorrections = new TCanvas("tCanAllCorrections","tCanAllCorrections");
    tCanAllCorrections->cd();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    tCfCorrectionwMatrix->GetXaxis()->SetRangeUser(0.,0.1);
    tCfCorrectionwMatrix->GetYaxis()->SetRangeUser(0.996,1.001);

    TAxis* xax1 = tCfCorrectionwMatrix->GetXaxis();
      xax1->SetTitle("k* (GeV/c)");
      xax1->SetTitleSize(0.04);
      xax1->SetTitleOffset(1.0);
//      xax1->CenterTitle();

    TAxis* yax1 = tCfCorrectionwMatrix->GetYaxis();
      yax1->SetTitle("C(k*_{true})/C(k*_{rec})");
      yax1->SetTitleSize(0.04);
      yax1->SetTitleOffset(1.3);
//      yax1->CenterTitle();

    tCfCorrectionwMatrix->DrawCopy();
    tFakeCorrectionHist->DrawCopy("same");
    tCfCorrectionwMatrixFine->DrawCopy("same");
    tFakeCorrectionHistFine->DrawCopy("same");

    tCfCorrectionwMatrix->DrawCopy("same");

    TLegend *tLeg = new TLegend(0.55,0.15,0.85,0.50);
      tLeg->SetHeader("Correction Factors");
      tLeg->AddEntry((TObject*)0, "Course Binning", "");
      tLeg->AddEntry(tFakeCorrectionHist,"TwoCFs","p");
      tLeg->AddEntry(tCfCorrectionwMatrix,"Matrix","p");
      tLeg->AddEntry((TObject*)0, "Fine Binning", "");
      tLeg->AddEntry(tFakeCorrectionHistFine,"TwoCFsFine","p");
      tLeg->AddEntry(tCfCorrectionwMatrixFine,"MatrixFine","p");
    TLegendEntry* tHeader0 = (TLegendEntry*)tLeg->GetListOfPrimitives()->At(0);
      tHeader0->SetTextFont(20);
      tHeader0->SetTextSize(0.045);
    TLegendEntry* tHeader1 = (TLegendEntry*)tLeg->GetListOfPrimitives()->At(1);
      tHeader1->SetTextFont(20);
      tHeader1->SetTextSize(0.045);
    TLegendEntry* tHeader2 = (TLegendEntry*)tLeg->GetListOfPrimitives()->At(4);
      tHeader2->SetTextFont(20);
      tHeader2->SetTextSize(0.045);

      tLeg->Draw();

    TLine *line = new TLine(0,1,0.1,1);
    line->SetLineColor(14);
    line->Draw();

    TPaveText* text1 = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
      text1->SetFillColor(0);
      text1->SetTextSize(0.05);
      text1->AddText(TString(cAnalysisRootTags[tDataAndModel->GetAnalysisType()]));
      text1->SetBorderSize(1);
      text1->Draw();

    if(bSaveFigures) tCanAllCorrections->SaveAs(tSaveFiguresLocation+"CorrectionFactors_CompareFine_"+TString(cAnalysisBaseTags[tDataAndModel->GetAnalysisType()])+".eps");

  }



  if(bDrawCorrectedCfs)
  {
    TH1D* tCfCorrectedwMatrix = tDataAndModel->GetKStarCorrectedwMatrix(0,kSame,0.,1.);
    TH1D* tCfCorrectedwMatrixNumDenSmeared = tDataAndModel->GetKStarCorrectedwMatrixNumDenSmeared(kSame,0.,1.);
    TH1D* tCfCorrectedwMatrixFit = tDataAndModel->GetKStarCorrectedwMatrix(1,kRotSame,0.,0.1);

    TH1D* tUncorrected = (TH1D*)tDataAndModel->GetKStarCfUncorrected();
      tUncorrected->SetMarkerStyle(20);
      tUncorrected->SetMarkerColor(1);
      tUncorrected->SetLineColor(1);

    TH1D* tCorrectedwFakeHist = (TH1D*)tDataAndModel->GetKStarCfCorrectedwFakeHist();
      tCorrectedwFakeHist->SetMarkerStyle(24);
      tCorrectedwFakeHist->SetMarkerColor(1);
      tCorrectedwFakeHist->SetLineColor(1);

    TCanvas *tCanCorrected = new TCanvas("tCanCorrected","tCanCorrected");
    tCanCorrected->Divide(1,2);

    tCanCorrected->cd(1);
    tCfCorrectedwMatrix->GetXaxis()->SetRangeUser(0.,0.1);
    tCfCorrectedwMatrix->GetYaxis()->SetRangeUser(0.75,1.02);
    tCfCorrectedwMatrix->DrawCopy();
    tUncorrected->DrawCopy("same");

    tCanCorrected->cd(2);
    tCfCorrectedwMatrixFit->GetYaxis()->SetRangeUser(0.75,1.02);
    tCfCorrectedwMatrixFit->DrawCopy();
    tUncorrected->DrawCopy("same");

    TCanvas *tCanAllCorrected = new TCanvas("tCanAllCorrected","tCanAllCorrected");
    tCanAllCorrected->cd();
    gStyle->SetOptStat(0);
    tUncorrected->GetXaxis()->SetRangeUser(0.,0.1);
    tUncorrected->GetYaxis()->SetRangeUser(0.79,1.00);
    tUncorrected->DrawCopy();
    tCfCorrectedwMatrix->DrawCopy("same");
    tCfCorrectedwMatrixNumDenSmeared->DrawCopy("same");
    tCfCorrectedwMatrixFit->DrawCopy("same");
    tCorrectedwFakeHist->DrawCopy("same");
    tUncorrected->DrawCopy("same");

    TLegend *tLeg = new TLegend(0.65,0.15,0.85,0.35);
      tLeg->SetHeader("Corrected Cfs");
      tLeg->AddEntry(tUncorrected,"Uncorrected","p");
      tLeg->AddEntry(tCfCorrectedwMatrix,"Matrix","p");
      tLeg->AddEntry(tCfCorrectedwMatrixNumDenSmeared,"Smeared","p");
      tLeg->AddEntry(tCfCorrectedwMatrixFit,"MatrixFit","p");
      tLeg->AddEntry(tCorrectedwFakeHist,"FakeHist","p");
      tLeg->Draw();

    if(bSaveFigures) tCanAllCorrected->SaveAs(tSaveFiguresLocation+"CorrectedCfs_"+TString(cAnalysisBaseTags[tDataAndModel->GetAnalysisType()])+".eps");

  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
