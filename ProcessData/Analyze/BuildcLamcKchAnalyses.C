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

#include "DataAndModel.h"
class DataAndModel;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //-----Data
  TString FileLocationBase = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  //TString FileLocationBase = "~/Analysis/K0Lam/Results_cLamcKch_As";
  Analysis* LamKchP = new Analysis(FileLocationBase,kLamKchP,k0010);
  Analysis* ALamKchP = new Analysis(FileLocationBase,kALamKchP,k0010);
  Analysis* LamKchM = new Analysis(FileLocationBase,kLamKchM,k0010);
  Analysis* ALamKchM = new Analysis(FileLocationBase,kALamKchM,k0010);

  TString SaveFileName = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_20151007/0010/Results_cLamcKch_AsRc_20151007_0010TEST.root";

  //-----MC

  TString FileLocationBaseMC = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229";
  Analysis* LamKchPMC = new Analysis(FileLocationBaseMC,kLamKchP,k0010);
  Analysis* ALamKchPMC = new Analysis(FileLocationBaseMC,kALamKchP,k0010);
  Analysis* LamKchMMC = new Analysis(FileLocationBaseMC,kLamKchM,k0010);
  Analysis* ALamKchMMC = new Analysis(FileLocationBaseMC,kALamKchM,k0010);
/*
  TString FileLocationBaseMCd = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMCd_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMCd_KchAndLamFix2_20160229";
  Analysis* LamKchPMCd = new Analysis(FileLocationBaseMCd,kLamKchP,k0010);
  Analysis* ALamKchPMCd = new Analysis(FileLocationBaseMCd,kALamKchP,k0010);
  Analysis* LamKchMMCd = new Analysis(FileLocationBaseMCd,kLamKchM,k0010);
  Analysis* ALamKchMMCd = new Analysis(FileLocationBaseMCd,kALamKchM,k0010);
*/
  vector<PartialAnalysis*> tLamKchPMCTotVec;
  vector<PartialAnalysis*> tTempLamKchPMCVec = LamKchPMC->GetPartialAnalysisCollection();
  //vector<PartialAnalysis*> tTempLamKchPMCdVec = LamKchPMCd->GetPartialAnalysisCollection();
  //assert(tTempLamKchPMCVec.size() == tTempLamKchPMCdVec.size());
  for(unsigned int i=0; i<tTempLamKchPMCVec.size(); i++)
  {
    tLamKchPMCTotVec.push_back(tTempLamKchPMCVec[i]);
    //tLamKchPMCTotVec.push_back(tTempLamKchPMCdVec[i]);
  }

  vector<PartialAnalysis*> tALamKchPMCTotVec;
  vector<PartialAnalysis*> tTempALamKchPMCVec = ALamKchPMC->GetPartialAnalysisCollection();
  //vector<PartialAnalysis*> tTempALamKchPMCdVec = ALamKchPMCd->GetPartialAnalysisCollection();
  //assert(tTempALamKchPMCVec.size() == tTempALamKchPMCdVec.size());
  for(unsigned int i=0; i<tTempALamKchPMCVec.size(); i++)
  {
    tALamKchPMCTotVec.push_back(tTempALamKchPMCVec[i]);
    //tALamKchPMCTotVec.push_back(tTempALamKchPMCdVec[i]);
  }

  vector<PartialAnalysis*> tLamKchMMCTotVec;
  vector<PartialAnalysis*> tTempLamKchMMCVec = LamKchMMC->GetPartialAnalysisCollection();
  //vector<PartialAnalysis*> tTempLamKchMMCdVec = LamKchMMCd->GetPartialAnalysisCollection();
  //assert(tTempLamKchMMCVec.size() == tTempLamKchMMCdVec.size());
  for(unsigned int i=0; i<tTempLamKchMMCVec.size(); i++)
  {
    tLamKchMMCTotVec.push_back(tTempLamKchMMCVec[i]);
    //tLamKchMMCTotVec.push_back(tTempLamKchMMCdVec[i]);
  }

  vector<PartialAnalysis*> tALamKchMMCTotVec;
  vector<PartialAnalysis*> tTempALamKchMMCVec = ALamKchMMC->GetPartialAnalysisCollection();
  //vector<PartialAnalysis*> tTempALamKchMMCdVec = ALamKchMMCd->GetPartialAnalysisCollection();
  //assert(tTempALamKchMMCVec.size() == tTempALamKchMMCdVec.size());
  for(unsigned int i=0; i<tTempALamKchMMCVec.size(); i++)
  {
    tALamKchMMCTotVec.push_back(tTempALamKchMMCVec[i]);
    //tALamKchMMCTotVec.push_back(tTempALamKchMMCdVec[i]);
  }

  Analysis* LamKchPMCTot = new Analysis("LamKchPMCTot_0010",tLamKchPMCTotVec);
  Analysis* ALamKchPMCTot = new Analysis("ALamKchPMCTot_0010",tALamKchPMCTotVec);
  Analysis* LamKchMMCTot = new Analysis("LamKchMMCTot_0010",tLamKchMMCTotVec);
  Analysis* ALamKchMMCTot = new Analysis("ALamKchMMCTot_0010",tALamKchMMCTotVec);


//-----------------------------------------------------------------------------


  bool bSaveFile = false;
  TFile *mySaveFile;
  if(bSaveFile) {mySaveFile = new TFile(SaveFileName, "RECREATE");}

  bool bContainsPurity = false;
  bool bContainsKStarCfs = false;
  bool bContainsAvgSepCfs = false;

  bool bContainsKStar2dCfs = false;

  bool bContainsSepHeavyCfs = false;
  bool bContainsAvgSepCowSailCfs = false;

  bool bViewPart1MassFail = false;

  bool bDrawMC = true;

  bool bDrawModelCfTrueIdealCfTrueRatio = true;
  bool bDrawModelCfFakeIdealCfFakeRatio = true;
  bool bDrawCorrectedKStarCfs = false;
  bool bDrawAllKStarCfs = false;
  bool bDrawAllKStarTrueVsRec = true;

  bool bSaveFigures = false;
  TString tSaveFiguresLocation = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_20151007/0010/";
  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    LamKchP->BuildKStarHeavyCf();
    ALamKchP->BuildKStarHeavyCf();
    LamKchM->BuildKStarHeavyCf();
    ALamKchM->BuildKStarHeavyCf();

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(LamKchP->GetKStarHeavyCf()->GetHeavyCf(),LamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(LamKchM->GetKStarHeavyCf()->GetHeavyCf(),LamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(ALamKchP->GetKStarHeavyCf()->GetHeavyCf(),ALamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(ALamKchM->GetKStarHeavyCf()->GetHeavyCf(),ALamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TString tNewNameLamKchP = LamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameLamKchP += " & " ;
      tNewNameLamKchP += LamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameLamKchP += TString(cCentralityTags[LamKchP->GetCentralityType()]);
    LamKchP->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameLamKchP);

    TString tNewNameALamKchP = ALamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameALamKchP += " & " ;
      tNewNameALamKchP += ALamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameALamKchP += TString(cCentralityTags[ALamKchP->GetCentralityType()]);
    ALamKchP->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameALamKchP);

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");
    canKStar->Divide(2,1);


    LamKchP->DrawKStarHeavyCf((TPad*)canKStar->cd(1),2);
    LamKchM->DrawKStarHeavyCf((TPad*)canKStar->cd(1),4,"same");
    canKStar->cd(1);
    leg1->Draw();


    ALamKchP->DrawKStarHeavyCf((TPad*)canKStar->cd(2),4);
    ALamKchM->DrawKStarHeavyCf((TPad*)canKStar->cd(2),2,"same");
    canKStar->cd(2);
    leg2->Draw();

    if(bSaveFigures)
    {
      TString aName = "cLamcKchKStarCfs.eps";
      canKStar->SaveAs(tSaveFiguresLocation+aName);
    }

    //----------------------------------

    LamKchP->OutputPassFailInfo();
    LamKchM->OutputPassFailInfo();
    ALamKchP->OutputPassFailInfo();
    ALamKchM->OutputPassFailInfo();

    //----------------------------------
    if(bSaveFile)
    {
      LamKchP->SaveAllKStarHeavyCf(mySaveFile);
      LamKchM->SaveAllKStarHeavyCf(mySaveFile);
      ALamKchP->SaveAllKStarHeavyCf(mySaveFile);
      ALamKchM->SaveAllKStarHeavyCf(mySaveFile);
    }

  }


  if(bContainsKStarCfs && bDrawMC)
  {
    LamKchP->BuildKStarHeavyCf();
    ALamKchP->BuildKStarHeavyCf();
    LamKchM->BuildKStarHeavyCf();
    ALamKchM->BuildKStarHeavyCf();

    LamKchPMCTot->BuildKStarHeavyCf();
    ALamKchPMCTot->BuildKStarHeavyCf();
    LamKchMMCTot->BuildKStarHeavyCf();
    ALamKchMMCTot->BuildKStarHeavyCf();

    LamKchPMCTot->BuildKStarHeavyCfMCTrue();
    ALamKchPMCTot->BuildKStarHeavyCfMCTrue();
    LamKchMMCTot->BuildKStarHeavyCfMCTrue();
    ALamKchMMCTot->BuildKStarHeavyCfMCTrue();

    TCanvas* canKStarvMC = new TCanvas("canKStarvMC","canKStarvMC");
    canKStarvMC->Divide(2,2);

    LamKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),2);
    LamKchPMCTot->GetKStarHeavyCf()->Rebin(2);
    LamKchPMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),1,"same",20);
    LamKchPMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    LamKchPMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(1),1,"same",24);

    ALamKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),2);
    ALamKchMMCTot->GetKStarHeavyCf()->Rebin(2);
    ALamKchMMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),1,"same",20);
    ALamKchMMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    ALamKchMMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(2),1,"same",24);

    LamKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),4);
    LamKchMMCTot->GetKStarHeavyCf()->Rebin(2);
    LamKchMMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),1,"same",20);
    LamKchMMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    LamKchMMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(3),1,"same",24);

    ALamKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),4);
    ALamKchPMCTot->GetKStarHeavyCf()->Rebin(2);
    ALamKchPMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),1,"same",20);
    ALamKchPMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    ALamKchPMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(4),1,"same",24);

    //------------------------------------------------------------

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(LamKchP->GetKStarHeavyCf()->GetHeavyCf(),LamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(LamKchPMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(LamKchPMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg1->AddEntry(LamKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(LamKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(1);
    leg1->Draw();

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(ALamKchM->GetKStarHeavyCf()->GetHeavyCf(),ALamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(ALamKchMMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(ALamKchMMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg2->AddEntry(ALamKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(ALamKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(2);
    leg2->Draw();

    TLegend* leg3 = new TLegend(0.60,0.12,0.89,0.32);
      leg3->SetFillColor(0);
      leg3->AddEntry(LamKchM->GetKStarHeavyCf()->GetHeavyCf(),LamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg3->AddEntry(LamKchMMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(LamKchMMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg3->AddEntry(LamKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(LamKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(3);
    leg3->Draw();

    TLegend* leg4 = new TLegend(0.60,0.12,0.89,0.32);
      leg4->SetFillColor(0);
      leg4->AddEntry(ALamKchP->GetKStarHeavyCf()->GetHeavyCf(),ALamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg4->AddEntry(ALamKchPMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(ALamKchPMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg4->AddEntry(ALamKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(ALamKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(4);
    leg4->Draw();

    if(bSaveFigures)
    {
      TString aName = "cLamcKchKStarvMCCfs.eps";
      canKStarvMC->SaveAs(tSaveFiguresLocation+aName);
    }

  }



  if(bDrawModelCfTrueIdealCfTrueRatio)
  {
    LamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

    LamKchPMCTot->FitModelCfTrueIdealCfTrueRatio();
    ALamKchPMCTot->FitModelCfTrueIdealCfTrueRatio();
    LamKchMMCTot->FitModelCfTrueIdealCfTrueRatio();
    ALamKchMMCTot->FitModelCfTrueIdealCfTrueRatio();

    TCanvas* canTrueIdealTrueRatio = new TCanvas("canTrueIdealTrueRatio","canTrueIdealTrueRatio");
    canTrueIdealTrueRatio->Divide(2,2);

    TH1* tRatioLamKchP = LamKchPMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchP = LamKchPMCTot->GetMomResFit();
    TH1* tRatioALamKchP = ALamKchPMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchP = ALamKchPMCTot->GetMomResFit();
    TH1* tRatioLamKchM = LamKchMMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchM = LamKchMMCTot->GetMomResFit();
    TH1* tRatioALamKchM = ALamKchMMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchM = ALamKchMMCTot->GetMomResFit();

    double tYmin = 0.94;
    double tYmax = 1.02;

    canTrueIdealTrueRatio->cd(1);
    tRatioLamKchP->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tRatioLamKchP->Draw();
    tMomResFitLamKchP->Draw("same");

    canTrueIdealTrueRatio->cd(2);
    tRatioALamKchM->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tRatioALamKchM->Draw();
    tMomResFitALamKchM->Draw("same");

    canTrueIdealTrueRatio->cd(3);
    tRatioLamKchM->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tRatioLamKchM->Draw();
    tMomResFitLamKchM->Draw("same");

    canTrueIdealTrueRatio->cd(4);
    tRatioALamKchP->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tRatioALamKchP->Draw();
    tMomResFitALamKchP->Draw("same");
  }


  if(bDrawModelCfFakeIdealCfFakeRatio)
  {
    LamKchPMCTot->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    ALamKchPMCTot->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    LamKchMMCTot->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    ALamKchMMCTot->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);

    LamKchPMCTot->FitModelCfFakeIdealCfFakeRatio();
    ALamKchPMCTot->FitModelCfFakeIdealCfFakeRatio();
    LamKchMMCTot->FitModelCfFakeIdealCfFakeRatio();
    ALamKchMMCTot->FitModelCfFakeIdealCfFakeRatio();

    TCanvas* canFakeIdealFakeRatio = new TCanvas("canFakeIdealFakeRatio","canFakeIdealFakeRatio");
    canFakeIdealFakeRatio->Divide(2,2);

    TH1* tFakeRatioLamKchP = LamKchPMCTot->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeLamKchP = LamKchPMCTot->GetMomResFitFake();
    TH1* tFakeRatioALamKchP = ALamKchPMCTot->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeALamKchP = ALamKchPMCTot->GetMomResFitFake();
    TH1* tFakeRatioLamKchM = LamKchMMCTot->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeLamKchM = LamKchMMCTot->GetMomResFitFake();
    TH1* tFakeRatioALamKchM = ALamKchMMCTot->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeALamKchM = ALamKchMMCTot->GetMomResFitFake();

    double tYmin = 0.94;
    double tYmax = 1.02;

    canFakeIdealFakeRatio->cd(1);
    tFakeRatioLamKchP->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tFakeRatioLamKchP->Draw();
    tMomResFitFakeLamKchP->Draw("same");

    canFakeIdealFakeRatio->cd(2);
    tFakeRatioALamKchM->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tFakeRatioALamKchM->Draw();
    tMomResFitFakeALamKchM->Draw("same");

    canFakeIdealFakeRatio->cd(3);
    tFakeRatioLamKchM->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tFakeRatioLamKchM->Draw();
    tMomResFitFakeLamKchM->Draw("same");

    canFakeIdealFakeRatio->cd(4);
    tFakeRatioALamKchP->GetYaxis()->SetRangeUser(tYmin,tYmax);
    tFakeRatioALamKchP->Draw();
    tMomResFitFakeALamKchP->Draw("same");
  }


/*
  if(bDrawCorrectedKStarCfs)
  {
    LamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

    LamKchP->BuildKStarHeavyCf();
    ALamKchP->BuildKStarHeavyCf();
    LamKchM->BuildKStarHeavyCf();
    ALamKchM->BuildKStarHeavyCf();

    //----- Rebin -----
    LamKchP->GetKStarHeavyCf()->Rebin(2);
    ALamKchP->GetKStarHeavyCf()->Rebin(2);
    LamKchM->GetKStarHeavyCf()->Rebin(2);
    ALamKchM->GetKStarHeavyCf()->Rebin(2);


    //-----------------
    TH1* tRatioLamKchP = LamKchPMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchP = LamKchPMCTot->GetMomResFit();
    TH1* tRatioALamKchP = ALamKchPMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchP = ALamKchPMCTot->GetMomResFit();
    TH1* tRatioLamKchM = LamKchMMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchM = LamKchMMCTot->GetMomResFit();
    TH1* tRatioALamKchM = ALamKchMMCTot->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchM = ALamKchMMCTot->GetMomResFit();
    //-----

    double yMin = 0.86;
    double yMax = 1.05;

    double tMarkerStyle1 = 20;
    double tMarkerStyle1o = 24;

    double tMarkerStyle2 = 21;
    double tMarkerStyle2o = 25;

    double tMarkerStyle3 = 22;
    double tMarkerStyle3o = 26;

    TH1* LamKchPUncorrected = LamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchPUncorrected->GetYaxis()->SetRangeUser(yMin,yMax);
      LamKchPUncorrected->SetMarkerStyle(tMarkerStyle1);
    TH1* LamKchPCorrectedwHist = LamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchPCorrectedwHist->Multiply(tRatioLamKchP);
      LamKchPCorrectedwHist->SetLineColor(2);
      LamKchPCorrectedwHist->SetMarkerColor(2);
      LamKchPCorrectedwHist->SetMarkerStyle(tMarkerStyle2o);
    TH1* LamKchPCorrectedwFcn = LamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchPCorrectedwFcn->Multiply(tMomResFitLamKchP);
      LamKchPCorrectedwFcn->SetLineColor(4);
      LamKchPCorrectedwFcn->SetMarkerColor(4);
      LamKchPCorrectedwFcn->SetMarkerStyle(tMarkerStyle3o);

    TH1* ALamKchPUncorrected = ALamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchPUncorrected->GetYaxis()->SetRangeUser(yMin,yMax);
      ALamKchPUncorrected->SetMarkerStyle(tMarkerStyle1);
    TH1* ALamKchPCorrectedwHist = ALamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchPCorrectedwHist->Multiply(tRatioALamKchP);
      ALamKchPCorrectedwHist->SetLineColor(2);
      ALamKchPCorrectedwHist->SetMarkerColor(2);
      ALamKchPCorrectedwHist->SetMarkerStyle(tMarkerStyle2o);
    TH1* ALamKchPCorrectedwFcn = ALamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchPCorrectedwFcn->Multiply(tMomResFitALamKchP);
      ALamKchPCorrectedwFcn->SetLineColor(4);
      ALamKchPCorrectedwFcn->SetMarkerColor(4);
      ALamKchPCorrectedwFcn->SetMarkerStyle(tMarkerStyle3o);

    TH1* LamKchMUncorrected = LamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchMUncorrected->GetYaxis()->SetRangeUser(yMin,yMax);
      LamKchMUncorrected->SetMarkerStyle(tMarkerStyle1);
    TH1* LamKchMCorrectedwHist = LamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchMCorrectedwHist->Multiply(tRatioLamKchM);
      LamKchMCorrectedwHist->SetLineColor(2);
      LamKchMCorrectedwHist->SetMarkerColor(2);
      LamKchMCorrectedwHist->SetMarkerStyle(tMarkerStyle2o);
    TH1* LamKchMCorrectedwFcn = LamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      LamKchMCorrectedwFcn->Multiply(tMomResFitLamKchM);
      LamKchMCorrectedwFcn->SetLineColor(4);
      LamKchMCorrectedwFcn->SetMarkerColor(4);
      LamKchMCorrectedwFcn->SetMarkerStyle(tMarkerStyle3o);

    TH1* ALamKchMUncorrected = ALamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchMUncorrected->GetYaxis()->SetRangeUser(yMin,yMax);
      ALamKchMUncorrected->SetMarkerStyle(tMarkerStyle1);
    TH1* ALamKchMCorrectedwHist = ALamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchMCorrectedwHist->Multiply(tRatioALamKchM);
      ALamKchMCorrectedwHist->SetLineColor(2);
      ALamKchMCorrectedwHist->SetMarkerColor(2);
      ALamKchMCorrectedwHist->SetMarkerStyle(tMarkerStyle2o);
    TH1* ALamKchMCorrectedwFcn = ALamKchM->GetKStarHeavyCf()->GetHeavyCfClone();
      ALamKchMCorrectedwFcn->Multiply(tMomResFitALamKchM);
      ALamKchMCorrectedwFcn->SetLineColor(4);
      ALamKchMCorrectedwFcn->SetMarkerColor(4);
      ALamKchMCorrectedwFcn->SetMarkerStyle(tMarkerStyle3o);

    TCanvas* canCorrected = new TCanvas("canCorrected","canCorrected");
    canCorrected->Divide(2,2);

    canCorrected->cd(1);
    LamKchPUncorrected->Draw();
    LamKchPCorrectedwHist->Draw("same");
    LamKchPCorrectedwFcn->Draw("same");

    canCorrected->cd(2);
    ALamKchMUncorrected->Draw();
    ALamKchMCorrectedwHist->Draw("same");
    ALamKchMCorrectedwFcn->Draw("same");

    canCorrected->cd(3);
    LamKchMUncorrected->Draw();
    LamKchMCorrectedwHist->Draw("same");
    LamKchMCorrectedwFcn->Draw("same");

    canCorrected->cd(4);
    ALamKchPUncorrected->Draw();
    ALamKchPCorrectedwHist->Draw("same");
    ALamKchPCorrectedwFcn->Draw("same");

  }
*/

  if(bDrawCorrectedKStarCfs)
  {
    DataAndModel* tLamKchPwModel = new DataAndModel(LamKchP,LamKchPMCTot,0.32,0.40,2);
    DataAndModel* tALamKchPwModel = new DataAndModel(ALamKchP,ALamKchPMCTot,0.32,0.40,2);

    DataAndModel* tLamKchMwModel = new DataAndModel(LamKchM,LamKchMMCTot,0.32,0.40,2);
    DataAndModel* tALamKchMwModel = new DataAndModel(ALamKchM,ALamKchMMCTot,0.32,0.40,2);

    TCanvas* canCorrected = new TCanvas("canCorrected","canCorrected");
    canCorrected->Divide(2,2);

    tLamKchPwModel->DrawAllCorrectedCfs((TPad*)canCorrected->cd(1));
    tALamKchMwModel->DrawAllCorrectedCfs((TPad*)canCorrected->cd(2));
    tLamKchMwModel->DrawAllCorrectedCfs((TPad*)canCorrected->cd(3));
    tALamKchPwModel->DrawAllCorrectedCfs((TPad*)canCorrected->cd(4));


    //---------------------------------------------------------------
    TCanvas* canCorrwFit = new TCanvas("canCorrwFit","canCorrwFit");
    canCorrwFit->Divide(2,2);

    tLamKchPwModel->DrawTrueCorrectionwFit((TPad*)canCorrwFit->cd(1));
    tALamKchMwModel->DrawTrueCorrectionwFit((TPad*)canCorrwFit->cd(2));
    tLamKchMwModel->DrawTrueCorrectionwFit((TPad*)canCorrwFit->cd(3));
    tALamKchPwModel->DrawTrueCorrectionwFit((TPad*)canCorrwFit->cd(4));

    TCanvas* canCorrwFakeFit = new TCanvas("canCorrwFakeFit","canCorrwFakeFit");
    canCorrwFakeFit->Divide(2,2);

    tLamKchPwModel->DrawFakeCorrectionwFit((TPad*)canCorrwFakeFit->cd(1));
    tALamKchMwModel->DrawFakeCorrectionwFit((TPad*)canCorrwFakeFit->cd(2));
    tLamKchMwModel->DrawFakeCorrectionwFit((TPad*)canCorrwFakeFit->cd(3));
    tALamKchPwModel->DrawFakeCorrectionwFit((TPad*)canCorrwFakeFit->cd(4));

  }

  if(bDrawAllKStarCfs)
  {
    double Style1 = 20;
    double Style2 = 21;
    double Style3 = 22;
    double Style4 = 29;

    double Color1 = 1;
    double Color2 = 2;
    double Color3 = 3;
    double Color4 = 4;

    LamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMCTot->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

    LamKchPMCTot->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    ALamKchPMCTot->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    LamKchMMCTot->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    ALamKchMMCTot->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);

    LamKchP->BuildKStarHeavyCf();
    ALamKchP->BuildKStarHeavyCf();
    LamKchM->BuildKStarHeavyCf();
    ALamKchM->BuildKStarHeavyCf();

    //----- Rebin -----
    LamKchP->GetKStarHeavyCf()->Rebin(2);
    ALamKchP->GetKStarHeavyCf()->Rebin(2);
    LamKchM->GetKStarHeavyCf()->Rebin(2);
    ALamKchM->GetKStarHeavyCf()->Rebin(2);

    //--------------------------------------------------------------------------------------------
    TH1* LamKchPTrue = LamKchPMCTot->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      LamKchPTrue->SetLineColor(Color1);
      LamKchPTrue->SetMarkerColor(Color1);
      LamKchPTrue->SetMarkerStyle(Style1);
    TH1* LamKchPTrueIdeal = LamKchPMCTot->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      LamKchPTrueIdeal->SetLineColor(Color2);
      LamKchPTrueIdeal->SetMarkerColor(Color2);
      LamKchPTrueIdeal->SetMarkerStyle(Style2);
    TH1* LamKchPFake = LamKchPMCTot->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      LamKchPFake->SetLineColor(Color3);
      LamKchPFake->SetMarkerColor(Color3);
      LamKchPFake->SetMarkerStyle(Style3);
    TH1* LamKchPFakeIdeal = LamKchPMCTot->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      LamKchPFakeIdeal->SetLineColor(Color4);
      LamKchPFakeIdeal->SetMarkerColor(Color4);
      LamKchPFakeIdeal->SetMarkerStyle(Style4);

    TH1* ALamKchPTrue = ALamKchPMCTot->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      ALamKchPTrue->SetLineColor(Color1);
      ALamKchPTrue->SetMarkerColor(Color1);
      ALamKchPTrue->SetMarkerStyle(Style1);
    TH1* ALamKchPTrueIdeal = ALamKchPMCTot->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      ALamKchPTrueIdeal->SetLineColor(Color2);
      ALamKchPTrueIdeal->SetMarkerColor(Color2);
      ALamKchPTrueIdeal->SetMarkerStyle(Style2);
    TH1* ALamKchPFake = ALamKchPMCTot->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      ALamKchPFake->SetLineColor(Color3);
      ALamKchPFake->SetMarkerColor(Color3);
      ALamKchPFake->SetMarkerStyle(Style3);
    TH1* ALamKchPFakeIdeal = ALamKchPMCTot->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      ALamKchPFakeIdeal->SetLineColor(Color4);
      ALamKchPFakeIdeal->SetMarkerColor(Color4);
      ALamKchPFakeIdeal->SetMarkerStyle(Style4);

    TH1* LamKchMTrue = LamKchMMCTot->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      LamKchMTrue->SetLineColor(Color1);
      LamKchMTrue->SetMarkerColor(Color1);
      LamKchMTrue->SetMarkerStyle(Style1);
    TH1* LamKchMTrueIdeal = LamKchMMCTot->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      LamKchMTrueIdeal->SetLineColor(Color2);
      LamKchMTrueIdeal->SetMarkerColor(Color2);
      LamKchMTrueIdeal->SetMarkerStyle(Style2);
    TH1* LamKchMFake = LamKchMMCTot->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      LamKchMFake->SetLineColor(Color3);
      LamKchMFake->SetMarkerColor(Color3);
      LamKchMFake->SetMarkerStyle(Style3);
    TH1* LamKchMFakeIdeal = LamKchMMCTot->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      LamKchMFakeIdeal->SetLineColor(Color4);
      LamKchMFakeIdeal->SetMarkerColor(Color4);
      LamKchMFakeIdeal->SetMarkerStyle(Style4);

    TH1* ALamKchMTrue = ALamKchMMCTot->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      ALamKchMTrue->SetLineColor(Color1);
      ALamKchMTrue->SetMarkerColor(Color1);
      ALamKchMTrue->SetMarkerStyle(Style1);
    TH1* ALamKchMTrueIdeal = ALamKchMMCTot->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      ALamKchMTrueIdeal->SetLineColor(Color2);
      ALamKchMTrueIdeal->SetMarkerColor(Color2);
      ALamKchMTrueIdeal->SetMarkerStyle(Style2);
    TH1* ALamKchMFake = ALamKchMMCTot->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      ALamKchMFake->SetLineColor(Color3);
      ALamKchMFake->SetMarkerColor(Color3);
      ALamKchMFake->SetMarkerStyle(Style3);
    TH1* ALamKchMFakeIdeal = ALamKchMMCTot->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      ALamKchMFakeIdeal->SetLineColor(Color4);
      ALamKchMFakeIdeal->SetMarkerColor(Color4);
      ALamKchMFakeIdeal->SetMarkerStyle(Style4);

    TCanvas *canAllKStarCfs = new TCanvas("canAllKStarCfs","canAllKStarCfs");
    canAllKStarCfs->Divide(2,2);

    double Ymin = 0.94;
    double Ymax = 1.02;

    canAllKStarCfs->cd(1);
    LamKchPTrue->GetYaxis()->SetRangeUser(Ymin,Ymax);
      LamKchPTrue->Draw();
      LamKchPTrueIdeal->Draw("same");
      LamKchPFake->Draw("same");
      LamKchPFakeIdeal->Draw("same");

      TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
        leg1->SetFillColor(0);
        leg1->AddEntry(LamKchPTrue,"Rec","lp");
        leg1->AddEntry(LamKchPTrueIdeal,"Truth","lp");
        leg1->AddEntry(LamKchPFake,"FakeRec","lp");
        leg1->AddEntry(LamKchPFakeIdeal,"FakeTruth","lp");
      leg1->Draw();


    canAllKStarCfs->cd(2);
    ALamKchMTrue->GetYaxis()->SetRangeUser(Ymin,Ymax);
      ALamKchMTrue->Draw();
      ALamKchMTrueIdeal->Draw("same");
      ALamKchMFake->Draw("same");
      ALamKchMFakeIdeal->Draw("same");

      TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
        leg2->SetFillColor(0);
        leg2->AddEntry(ALamKchMTrue,"Rec","lp");
        leg2->AddEntry(ALamKchMTrueIdeal,"Truth","lp");
        leg2->AddEntry(ALamKchMFake,"FakeRec","lp");
        leg2->AddEntry(ALamKchMFakeIdeal,"FakeTruth","lp");
      leg2->Draw();


    canAllKStarCfs->cd(3);
    LamKchMTrue->GetYaxis()->SetRangeUser(Ymin,Ymax);
      LamKchMTrue->Draw();
      LamKchMTrueIdeal->Draw("same");
      LamKchMFake->Draw("same");
      LamKchMFakeIdeal->Draw("same");

      TLegend* leg3 = new TLegend(0.60,0.12,0.89,0.32);
        leg3->SetFillColor(0);
        leg3->AddEntry(LamKchMTrue,"Rec","lp");
        leg3->AddEntry(LamKchMTrueIdeal,"Truth","lp");
        leg3->AddEntry(LamKchMFake,"FakeRec","lp");
        leg3->AddEntry(LamKchMFakeIdeal,"FakeTruth","lp");
      leg3->Draw();



    canAllKStarCfs->cd(4);
    ALamKchPTrue->GetYaxis()->SetRangeUser(Ymin,Ymax);
      ALamKchPTrue->Draw();
      ALamKchPTrueIdeal->Draw("same");
      ALamKchPFake->Draw("same");
      ALamKchPFakeIdeal->Draw("same");

      TLegend* leg4 = new TLegend(0.60,0.12,0.89,0.32);
        leg4->SetFillColor(0);
        leg4->AddEntry(ALamKchPTrue,"Rec","lp");
        leg4->AddEntry(ALamKchPTrueIdeal,"Truth","lp");
        leg4->AddEntry(ALamKchPFake,"FakeRec","lp");
        leg4->AddEntry(ALamKchPFakeIdeal,"FakeTruth","lp");
      leg4->Draw();

  }

  if(bDrawAllKStarTrueVsRec)
  {
    LamKchPMCTot->BuildAllModelKStarTrueVsRecTotal();
    ALamKchPMCTot->BuildAllModelKStarTrueVsRecTotal();
    LamKchMMCTot->BuildAllModelKStarTrueVsRecTotal();
    ALamKchMMCTot->BuildAllModelKStarTrueVsRecTotal();

    TH2* LamKchPTrueVsRecSame = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kSame);
    TH2* LamKchPTrueVsRecRotSame = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kRotSame);
    TH2* LamKchPTrueVsRecMixed = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kMixed);
    TH2* LamKchPTrueVsRecRotMixed = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kRotMixed);
/*
    TH2* LamKchPTrueVsRecSame2 = (TH2*)LamKchPTrueVsRecSame->Clone();
    TH2* ALamKchPTrueVsRecSame = ALamKchPMCTot->GetModelKStarTrueVsRecTotal(kSame);
    TH2* LamKchMTrueVsRecSame = LamKchMMCTot->GetModelKStarTrueVsRecTotal(kSame);
    TH2* ALamKchMTrueVsRecSame = ALamKchMMCTot->GetModelKStarTrueVsRecTotal(kSame);
*/
    TCanvas *canKStarTrueVsRec = new TCanvas("canKStarTrueVsRec","canKStarTrueVsRec");
    canKStarTrueVsRec->Divide(2,2);
    //gStyle->SetOptStat(0);

    canKStarTrueVsRec->cd(1);
      gPad->SetLogz();
      LamKchPTrueVsRecSame->Draw("colz");

    canKStarTrueVsRec->cd(2);
      gPad->SetLogz();
      LamKchPTrueVsRecMixed->Draw("colz");

    canKStarTrueVsRec->cd(3);
      gPad->SetLogz();
      LamKchPTrueVsRecRotSame->Draw("colz");

    canKStarTrueVsRec->cd(4);
      gPad->SetLogz();
      LamKchPTrueVsRecRotMixed->Draw("colz");
/*
TCanvas* canTest = new TCanvas("canTest","canTest");
canTest->cd();
TH1D* tProject = LamKchPTrueVsRecSame->ProjectionX();
tProject->DrawCopy();

TCanvas* canTest2 = new TCanvas("canTest2","canTest2");
canTest2->cd();
TH1D* tProject2 = LamKchPTrueVsRecSame->ProjectionY();
tProject2->DrawCopy();

TCanvas* canTest3 = new TCanvas("canTest3","canTest3");
canTest3->cd();
TH1D* tProject3 = (TH1D*)tProject->Clone();
tProject3->Divide(tProject2);
tProject3->DrawCopy();
*/
/*
    TCanvas *canKStarTrueVsRec2 = new TCanvas("canKStarTrueVsRec2","canKStarTrueVsRec2");
    canKStarTrueVsRec2->Divide(2,2);
    //gStyle->SetOptStat(0);

    canKStarTrueVsRec2->cd(1);
      gPad->SetLogz();
      LamKchPTrueVsRecSame2->Draw("lego2");

    canKStarTrueVsRec2->cd(2);
      gPad->SetLogz();
      ALamKchMTrueVsRecSame->Draw("lego2");

    canKStarTrueVsRec2->cd(3);
      gPad->SetLogz();
      LamKchMTrueVsRecSame->Draw("lego2");

    canKStarTrueVsRec2->cd(4);
      gPad->SetLogz();
      ALamKchPTrueVsRecSame->Draw("lego2");
*/
  }


  if(bContainsAvgSepCfs)
  {
    LamKchP->BuildAllAvgSepHeavyCfs();
    ALamKchP->BuildAllAvgSepHeavyCfs();
    LamKchM->BuildAllAvgSepHeavyCfs();
    ALamKchM->BuildAllAvgSepHeavyCfs();

    TCanvas *canAvgSepLamKchP = new TCanvas("canAvgSepLamKchP","canAvgSepLamKchP");
    TCanvas *canAvgSepALamKchP = new TCanvas("canAvgSepALamKchP","canAvgSepALamKchP");
    TCanvas *canAvgSepLamKchM = new TCanvas("canAvgSepLamKchM","canAvgSepLamKchM");
    TCanvas *canAvgSepALamKchM = new TCanvas("canAvgSepALamKchM","canAvgSepALamKchM");

    canAvgSepLamKchP->Divide(1,2);
    canAvgSepALamKchP->Divide(1,2);
    canAvgSepLamKchM->Divide(1,2);
    canAvgSepALamKchM->Divide(1,2);

    LamKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepLamKchP->cd(1));
    LamKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepLamKchP->cd(2));

    ALamKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepALamKchP->cd(1));
    ALamKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepALamKchP->cd(2));

    LamKchM->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepLamKchM->cd(1));
    LamKchM->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepLamKchM->cd(2));

    ALamKchM->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepALamKchM->cd(1));
    ALamKchM->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepALamKchM->cd(2));

    //----------------------------------
    if(bSaveFile)
    {
      LamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
      LamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
      ALamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
      ALamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
    }


  }




  if(bContainsKStar2dCfs)
  {
    LamKchP->BuildKStar2dHeavyCfs();
    ALamKchP->BuildKStar2dHeavyCfs();
    LamKchM->BuildKStar2dHeavyCfs();
    ALamKchM->BuildKStar2dHeavyCfs();

    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    canKStarRatios->Divide(2,2);

    LamKchP->RebinKStar2dHeavyCfs(2);
    ALamKchM->RebinKStar2dHeavyCfs(2);
    LamKchM->RebinKStar2dHeavyCfs(2);
    ALamKchP->RebinKStar2dHeavyCfs(2);

    LamKchP->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(1));
    ALamKchM->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(2));
    LamKchM->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(3));
    ALamKchP->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(4));

    if(bSaveFigures)
    {
      TString aName = "cLamcKchKStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveFiguresLocation+aName);
    }

  }







  if(bContainsSepHeavyCfs)
  {
    LamKchP->BuildAllSepHeavyCfs();
    ALamKchP->BuildAllSepHeavyCfs();
    LamKchM->BuildAllSepHeavyCfs();
    ALamKchM->BuildAllSepHeavyCfs();

    TCanvas *canSepCfsLamKchP = new TCanvas("canSepCfsLamKchP","canSepCfsLamKchP");
    TCanvas *canSepCfsALamKchP = new TCanvas("canSepCfsALamKchP","canSepCfsALamKchP");
    TCanvas *canSepCfsLamKchM = new TCanvas("canSepCfsLamKchM","canSepCfsLamKchM");
    TCanvas *canSepCfsALamKchM = new TCanvas("canSepCfsALamKchM","canSepCfsALamKchM");

    canSepCfsLamKchP->Divide(1,2);
    canSepCfsALamKchP->Divide(1,2);
    canSepCfsLamKchM->Divide(1,2);
    canSepCfsALamKchM->Divide(1,2);


    LamKchP->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsLamKchP->cd(1));
    LamKchP->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsLamKchP->cd(2));

    ALamKchP->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsALamKchP->cd(1));
    ALamKchP->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsALamKchP->cd(2));

    LamKchM->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsLamKchM->cd(1));
    LamKchM->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsLamKchM->cd(2));

    ALamKchM->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsALamKchM->cd(1));
    ALamKchM->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsALamKchM->cd(2));

  }



  if(bContainsAvgSepCowSailCfs)
  {
    LamKchP->BuildAllAvgSepCowSailHeavyCfs();
    ALamKchP->BuildAllAvgSepCowSailHeavyCfs();
    LamKchM->BuildAllAvgSepCowSailHeavyCfs();
    ALamKchM->BuildAllAvgSepCowSailHeavyCfs();

    TCanvas *canAvgSepCowSailLamKchP = new TCanvas("canAvgSepCowSailLamKchP","canAvgSepCowSailLamKchP");
    TCanvas *canAvgSepCowSailALamKchP = new TCanvas("canAvgSepCowSailALamKchP","canAvgSepCowSailALamKchP");
    TCanvas *canAvgSepCowSailLamKchM = new TCanvas("canAvgSepCowSailLamKchM","canAvgSepCowSailLamKchM");
    TCanvas *canAvgSepCowSailALamKchM = new TCanvas("canAvgSepCowSailALamKchM","canAvgSepCowSailALamKchM");

    canAvgSepCowSailLamKchP->Divide(1,2);
    canAvgSepCowSailALamKchP->Divide(1,2);
    canAvgSepCowSailLamKchM->Divide(1,2);
    canAvgSepCowSailALamKchM->Divide(1,2);

    LamKchP->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailLamKchP->cd(1));
    LamKchP->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailLamKchP->cd(2));

    ALamKchP->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailALamKchP->cd(1));
    ALamKchP->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailALamKchP->cd(2));

    LamKchM->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailLamKchM->cd(1));
    LamKchM->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailLamKchM->cd(2));

    ALamKchM->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailALamKchM->cd(1));
    ALamKchM->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailALamKchM->cd(2));

  }

  if(bViewPart1MassFail)
  {
    bool tDrawWideRangeToo = true;

    TCanvas* canPart1MassFail = new TCanvas("canPart1MassFail","canPart1MassFail");
    canPart1MassFail->Divide(2,2);

    LamKchP->DrawPart1MassFail((TPad*)canPart1MassFail->cd(1),tDrawWideRangeToo);
    ALamKchM->DrawPart1MassFail((TPad*)canPart1MassFail->cd(2),tDrawWideRangeToo);
    LamKchM->DrawPart1MassFail((TPad*)canPart1MassFail->cd(3),tDrawWideRangeToo);
    ALamKchP->DrawPart1MassFail((TPad*)canPart1MassFail->cd(4),tDrawWideRangeToo);

    if(bSaveFigures)
    {
      TString aName = "cLamcKchPart1MassFail.eps";
      canPart1MassFail->SaveAs(tSaveFiguresLocation+aName);
    }
  }




  if(bContainsPurity)
  {
    LamKchP->BuildPurityCollection();
    ALamKchP->BuildPurityCollection();
    LamKchM->BuildPurityCollection();
    ALamKchM->BuildPurityCollection();

    TCanvas* canPurity = new TCanvas("canPurity","canPurity");
    canPurity->Divide(2,2);

    LamKchP->DrawAllPurityHistos((TPad*)canPurity->cd(1));
    LamKchM->DrawAllPurityHistos((TPad*)canPurity->cd(2));
    ALamKchP->DrawAllPurityHistos((TPad*)canPurity->cd(3));
    ALamKchM->DrawAllPurityHistos((TPad*)canPurity->cd(4));

    if(bSaveFigures)
    {
      TString aName = "cLamcKchPurity.eps";
      canPurity->SaveAs(tSaveFiguresLocation+aName);

      TString aName2 = "LamPurity_LamKchP.eps";
      canPurity->cd(1)->SaveAs(tSaveFiguresLocation+aName2);
    }

  }

  if(bContainsPurity && bDrawMC)
  {
    LamKchPMCTot->GetMCKchPurity(true);
    LamKchPMCTot->GetMCKchPurity(false);

    LamKchMMCTot->GetMCKchPurity(true);
    LamKchMMCTot->GetMCKchPurity(false);

    ALamKchPMCTot->GetMCKchPurity(true);
    ALamKchPMCTot->GetMCKchPurity(false);

    ALamKchMMCTot->GetMCKchPurity(true);
    ALamKchMMCTot->GetMCKchPurity(false);
  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  if(bSaveFile) {mySaveFile->Close();}

  return 0;
}
