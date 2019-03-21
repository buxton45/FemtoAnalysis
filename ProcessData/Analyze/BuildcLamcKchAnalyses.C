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

//  TString tResultsDate = "20161027";
//  TString tResultsDate = "20180423_NoAvgSepCut";
  TString tResultsDate = "20180505";

  AnalysisType tAnType1 = kLamKchP;
  AnalysisType tConjAnType1 = kALamKchM;

  AnalysisType tAnType2 = kLamKchM;
  AnalysisType tConjAnType2 = kALamKchP;

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;  //TODO

//-----------------------------------------------------------------------------
  bool bSaveFigures = false;
  bool bSaveFile = false;

  bool bContainsPurity = true;
  bool bPrintPurity = true;

  bool bContainsKStarCfs = false;
  bool bContainsAvgSepCfs = true;

  bool bContainsKStar2dCfs = false;

  bool bContainsSepHeavyCfs = false;
  bool bContainsAvgSepCowSailCfs = false;

  bool bViewPart1MassFail = false;  //NOTE: kTrainSys do not include fail cut monitors

  bool bDrawMC = false;

  bool bDrawModelCfTrueIdealCfTrueRatio = false;
  bool bDrawModelCfFakeIdealCfFakeRatio = false;
  bool bDrawCorrectedKStarCfs = false;
  bool bDrawAllKStarCfs = false;
  bool bDrawAllKStarTrueVsRec = false;

  bool bDrawKchdEdx = false;
//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType1==kLamK0 || tAnType1==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType1==kLamKchP || tAnType1==kALamKchM || tAnType1==kLamKchM || tAnType1==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  //TODO PreTrain results with MCd?

  TString tSaveDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/3_DataSelection/Figures/";
//  TString tSaveDirectoryBase = tDirectoryBase;

  TFile *mySaveFile;
  TString tSaveFileName = TString::Format("%sResults_%s_%s.root", tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  if(bSaveFile) {mySaveFile = new TFile(tSaveFileName, "RECREATE");}

  //-----Data
  Analysis* LamKchP = new Analysis(tFileLocationBase,tAnType1,tCentType,tAnRunType);
  Analysis* ALamKchP = new Analysis(tFileLocationBase,tConjAnType2,tCentType,tAnRunType);
  Analysis* LamKchM = new Analysis(tFileLocationBase,tAnType2,tCentType,tAnRunType);
  Analysis* ALamKchM = new Analysis(tFileLocationBase,tConjAnType1,tCentType,tAnRunType);

  //-----MC
  Analysis* LamKchPMC = new Analysis(tFileLocationBaseMC,tAnType1,tCentType,tAnRunType);
  Analysis* ALamKchPMC = new Analysis(tFileLocationBaseMC,tConjAnType2,tCentType,tAnRunType);
  Analysis* LamKchMMC = new Analysis(tFileLocationBaseMC,tAnType2,tCentType,tAnRunType);
  Analysis* ALamKchMMC = new Analysis(tFileLocationBaseMC,tConjAnType1,tCentType,tAnRunType);

//-----------------------------------------------------------------------------



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
      canKStar->SaveAs(tSaveDirectoryBase+aName);
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

    LamKchPMC->BuildKStarHeavyCf();
    ALamKchPMC->BuildKStarHeavyCf();
    LamKchMMC->BuildKStarHeavyCf();
    ALamKchMMC->BuildKStarHeavyCf();

    LamKchPMC->BuildKStarHeavyCfMCTrue();
    ALamKchPMC->BuildKStarHeavyCfMCTrue();
    LamKchMMC->BuildKStarHeavyCfMCTrue();
    ALamKchMMC->BuildKStarHeavyCfMCTrue();

    TCanvas* canKStarvMC = new TCanvas("canKStarvMC","canKStarvMC");
    canKStarvMC->Divide(2,2);

    LamKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),2);
    LamKchPMC->GetKStarHeavyCf()->Rebin(2);
    LamKchPMC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),1,"same",20);
    LamKchPMC->GetKStarHeavyCfMCTrue()->Rebin(2);
    LamKchPMC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(1),1,"same",24);

    ALamKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),2);
    ALamKchMMC->GetKStarHeavyCf()->Rebin(2);
    ALamKchMMC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),1,"same",20);
    ALamKchMMC->GetKStarHeavyCfMCTrue()->Rebin(2);
    ALamKchMMC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(2),1,"same",24);

    LamKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),4);
    LamKchMMC->GetKStarHeavyCf()->Rebin(2);
    LamKchMMC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),1,"same",20);
    LamKchMMC->GetKStarHeavyCfMCTrue()->Rebin(2);
    LamKchMMC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(3),1,"same",24);

    ALamKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),4);
    ALamKchPMC->GetKStarHeavyCf()->Rebin(2);
    ALamKchPMC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),1,"same",20);
    ALamKchPMC->GetKStarHeavyCfMCTrue()->Rebin(2);
    ALamKchPMC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(4),1,"same",24);

    //------------------------------------------------------------

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(LamKchP->GetKStarHeavyCf()->GetHeavyCf(),LamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(LamKchPMC->GetKStarHeavyCf()->GetHeavyCf(),TString(LamKchPMC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg1->AddEntry(LamKchPMC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(LamKchPMC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(1);
    leg1->Draw();

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(ALamKchM->GetKStarHeavyCf()->GetHeavyCf(),ALamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(ALamKchMMC->GetKStarHeavyCf()->GetHeavyCf(),TString(ALamKchMMC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg2->AddEntry(ALamKchMMC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(ALamKchMMC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(2);
    leg2->Draw();

    TLegend* leg3 = new TLegend(0.60,0.12,0.89,0.32);
      leg3->SetFillColor(0);
      leg3->AddEntry(LamKchM->GetKStarHeavyCf()->GetHeavyCf(),LamKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg3->AddEntry(LamKchMMC->GetKStarHeavyCf()->GetHeavyCf(),TString(LamKchMMC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg3->AddEntry(LamKchMMC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(LamKchMMC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(3);
    leg3->Draw();

    TLegend* leg4 = new TLegend(0.60,0.12,0.89,0.32);
      leg4->SetFillColor(0);
      leg4->AddEntry(ALamKchP->GetKStarHeavyCf()->GetHeavyCf(),ALamKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg4->AddEntry(ALamKchPMC->GetKStarHeavyCf()->GetHeavyCf(),TString(ALamKchPMC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg4->AddEntry(ALamKchPMC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(ALamKchPMC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(4);
    leg4->Draw();

    if(bSaveFigures)
    {
      TString aName = "cLamcKchKStarvMCCfs.eps";
      canKStarvMC->SaveAs(tSaveDirectoryBase+aName);
    }

  }



  if(bDrawModelCfTrueIdealCfTrueRatio)
  {
    LamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

    LamKchPMC->FitModelCfTrueIdealCfTrueRatio();
    ALamKchPMC->FitModelCfTrueIdealCfTrueRatio();
    LamKchMMC->FitModelCfTrueIdealCfTrueRatio();
    ALamKchMMC->FitModelCfTrueIdealCfTrueRatio();

    TCanvas* canTrueIdealTrueRatio = new TCanvas("canTrueIdealTrueRatio","canTrueIdealTrueRatio");
    canTrueIdealTrueRatio->Divide(2,2);

    TH1* tRatioLamKchP = LamKchPMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchP = LamKchPMC->GetMomResFit();
    TH1* tRatioALamKchP = ALamKchPMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchP = ALamKchPMC->GetMomResFit();
    TH1* tRatioLamKchM = LamKchMMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchM = LamKchMMC->GetMomResFit();
    TH1* tRatioALamKchM = ALamKchMMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchM = ALamKchMMC->GetMomResFit();

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
    LamKchPMC->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    ALamKchPMC->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    LamKchMMC->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
    ALamKchMMC->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);

    LamKchPMC->FitModelCfFakeIdealCfFakeRatio();
    ALamKchPMC->FitModelCfFakeIdealCfFakeRatio();
    LamKchMMC->FitModelCfFakeIdealCfFakeRatio();
    ALamKchMMC->FitModelCfFakeIdealCfFakeRatio();

    TCanvas* canFakeIdealFakeRatio = new TCanvas("canFakeIdealFakeRatio","canFakeIdealFakeRatio");
    canFakeIdealFakeRatio->Divide(2,2);

    TH1* tFakeRatioLamKchP = LamKchPMC->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeLamKchP = LamKchPMC->GetMomResFitFake();
    TH1* tFakeRatioALamKchP = ALamKchPMC->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeALamKchP = ALamKchPMC->GetMomResFitFake();
    TH1* tFakeRatioLamKchM = LamKchMMC->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeLamKchM = LamKchMMC->GetMomResFitFake();
    TH1* tFakeRatioALamKchM = ALamKchMMC->GetModelCfFakeIdealCfFakeRatio();
      TF1* tMomResFitFakeALamKchM = ALamKchMMC->GetMomResFitFake();

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
    LamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

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
    TH1* tRatioLamKchP = LamKchPMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchP = LamKchPMC->GetMomResFit();
    TH1* tRatioALamKchP = ALamKchPMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchP = ALamKchPMC->GetMomResFit();
    TH1* tRatioLamKchM = LamKchMMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitLamKchM = LamKchMMC->GetMomResFit();
    TH1* tRatioALamKchM = ALamKchMMC->GetModelCfTrueIdealCfTrueRatio();
      TF1* tMomResFitALamKchM = ALamKchMMC->GetMomResFit();
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
    DataAndModel* tLamKchPwModel = new DataAndModel(LamKchP,LamKchPMC,0.32,0.40,2);
    DataAndModel* tALamKchPwModel = new DataAndModel(ALamKchP,ALamKchPMC,0.32,0.40,2);

    DataAndModel* tLamKchMwModel = new DataAndModel(LamKchM,LamKchMMC,0.32,0.40,2);
    DataAndModel* tALamKchMwModel = new DataAndModel(ALamKchM,ALamKchMMC,0.32,0.40,2);

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

    LamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchPMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    LamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);
    ALamKchMMC->BuildModelCfTrueIdealCfTrueRatio(0.32,0.40,2);

    LamKchPMC->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    ALamKchPMC->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    LamKchMMC->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);
    ALamKchMMC->BuildModelCfFakeIdealCfFakeRatio(0.32,0.40,2);

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
    TH1* LamKchPTrue = LamKchPMC->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      LamKchPTrue->SetLineColor(Color1);
      LamKchPTrue->SetMarkerColor(Color1);
      LamKchPTrue->SetMarkerStyle(Style1);
    TH1* LamKchPTrueIdeal = LamKchPMC->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      LamKchPTrueIdeal->SetLineColor(Color2);
      LamKchPTrueIdeal->SetMarkerColor(Color2);
      LamKchPTrueIdeal->SetMarkerStyle(Style2);
    TH1* LamKchPFake = LamKchPMC->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      LamKchPFake->SetLineColor(Color3);
      LamKchPFake->SetMarkerColor(Color3);
      LamKchPFake->SetMarkerStyle(Style3);
    TH1* LamKchPFakeIdeal = LamKchPMC->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      LamKchPFakeIdeal->SetLineColor(Color4);
      LamKchPFakeIdeal->SetMarkerColor(Color4);
      LamKchPFakeIdeal->SetMarkerStyle(Style4);

    TH1* ALamKchPTrue = ALamKchPMC->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      ALamKchPTrue->SetLineColor(Color1);
      ALamKchPTrue->SetMarkerColor(Color1);
      ALamKchPTrue->SetMarkerStyle(Style1);
    TH1* ALamKchPTrueIdeal = ALamKchPMC->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      ALamKchPTrueIdeal->SetLineColor(Color2);
      ALamKchPTrueIdeal->SetMarkerColor(Color2);
      ALamKchPTrueIdeal->SetMarkerStyle(Style2);
    TH1* ALamKchPFake = ALamKchPMC->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      ALamKchPFake->SetLineColor(Color3);
      ALamKchPFake->SetMarkerColor(Color3);
      ALamKchPFake->SetMarkerStyle(Style3);
    TH1* ALamKchPFakeIdeal = ALamKchPMC->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      ALamKchPFakeIdeal->SetLineColor(Color4);
      ALamKchPFakeIdeal->SetMarkerColor(Color4);
      ALamKchPFakeIdeal->SetMarkerStyle(Style4);

    TH1* LamKchMTrue = LamKchMMC->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      LamKchMTrue->SetLineColor(Color1);
      LamKchMTrue->SetMarkerColor(Color1);
      LamKchMTrue->SetMarkerStyle(Style1);
    TH1* LamKchMTrueIdeal = LamKchMMC->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      LamKchMTrueIdeal->SetLineColor(Color2);
      LamKchMTrueIdeal->SetMarkerColor(Color2);
      LamKchMTrueIdeal->SetMarkerStyle(Style2);
    TH1* LamKchMFake = LamKchMMC->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      LamKchMFake->SetLineColor(Color3);
      LamKchMFake->SetMarkerColor(Color3);
      LamKchMFake->SetMarkerStyle(Style3);
    TH1* LamKchMFakeIdeal = LamKchMMC->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
      LamKchMFakeIdeal->SetLineColor(Color4);
      LamKchMFakeIdeal->SetMarkerColor(Color4);
      LamKchMFakeIdeal->SetMarkerStyle(Style4);

    TH1* ALamKchMTrue = ALamKchMMC->GetModelKStarHeavyCfTrue()->GetHeavyCfClone();
      ALamKchMTrue->SetLineColor(Color1);
      ALamKchMTrue->SetMarkerColor(Color1);
      ALamKchMTrue->SetMarkerStyle(Style1);
    TH1* ALamKchMTrueIdeal = ALamKchMMC->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();
      ALamKchMTrueIdeal->SetLineColor(Color2);
      ALamKchMTrueIdeal->SetMarkerColor(Color2);
      ALamKchMTrueIdeal->SetMarkerStyle(Style2);
    TH1* ALamKchMFake = ALamKchMMC->GetModelKStarHeavyCfFake()->GetHeavyCfClone();
      ALamKchMFake->SetLineColor(Color3);
      ALamKchMFake->SetMarkerColor(Color3);
      ALamKchMFake->SetMarkerStyle(Style3);
    TH1* ALamKchMFakeIdeal = ALamKchMMC->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();
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
    LamKchPMC->BuildAllModelKStarTrueVsRecTotal();
    ALamKchPMC->BuildAllModelKStarTrueVsRecTotal();
    LamKchMMC->BuildAllModelKStarTrueVsRecTotal();
    ALamKchMMC->BuildAllModelKStarTrueVsRecTotal();

    TH2* LamKchPTrueVsRecSame = LamKchPMC->GetModelKStarTrueVsRecTotal(kSame);
    TH2* LamKchPTrueVsRecRotSame = LamKchPMC->GetModelKStarTrueVsRecTotal(kRotSame);
    TH2* LamKchPTrueVsRecMixed = LamKchPMC->GetModelKStarTrueVsRecTotal(kMixed);
    TH2* LamKchPTrueVsRecRotMixed = LamKchPMC->GetModelKStarTrueVsRecTotal(kRotMixed);
/*
    TH2* LamKchPTrueVsRecSame2 = (TH2*)LamKchPTrueVsRecSame->Clone();
    TH2* ALamKchPTrueVsRecSame = ALamKchPMC->GetModelKStarTrueVsRecTotal(kSame);
    TH2* LamKchMTrueVsRecSame = LamKchMMC->GetModelKStarTrueVsRecTotal(kSame);
    TH2* ALamKchMTrueVsRecSame = ALamKchMMC->GetModelKStarTrueVsRecTotal(kSame);
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
  if(bContainsAvgSepCfs && bDrawMC)
  {
    LamKchPMC->BuildAllAvgSepHeavyCfs();
    ALamKchPMC->BuildAllAvgSepHeavyCfs();
    LamKchMMC->BuildAllAvgSepHeavyCfs();
    ALamKchMMC->BuildAllAvgSepHeavyCfs();

    //---- Rebinning ------------------------
    int tRebin = 2;
    LamKchPMC->GetAvgSepHeavyCf(kTrackPos, tRebin);
    LamKchPMC->GetAvgSepHeavyCf(kTrackNeg, tRebin);

    ALamKchPMC->GetAvgSepHeavyCf(kTrackPos, tRebin);
    ALamKchPMC->GetAvgSepHeavyCf(kTrackNeg, tRebin);

    LamKchMMC->GetAvgSepHeavyCf(kTrackPos, tRebin);
    LamKchMMC->GetAvgSepHeavyCf(kTrackNeg, tRebin);

    ALamKchMMC->GetAvgSepHeavyCf(kTrackPos, tRebin);
    ALamKchMMC->GetAvgSepHeavyCf(kTrackNeg, tRebin);

    //---------------------------------------

    TCanvas *canAvgSepLamKchPMC = new TCanvas("canAvgSepLamKchPMC","canAvgSepLamKchPMC");
    TCanvas *canAvgSepALamKchPMC = new TCanvas("canAvgSepALamKchPMC","canAvgSepALamKchPMC");
    TCanvas *canAvgSepLamKchMMC = new TCanvas("canAvgSepLamKchMMC","canAvgSepLamKchMMC");
    TCanvas *canAvgSepALamKchMMC = new TCanvas("canAvgSepALamKchMMC","canAvgSepALamKchMMC");

    canAvgSepLamKchPMC->Divide(1,2);
    canAvgSepALamKchPMC->Divide(1,2);
    canAvgSepLamKchMMC->Divide(1,2);
    canAvgSepALamKchMMC->Divide(1,2);

    LamKchPMC->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepLamKchPMC->cd(1));
    LamKchPMC->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepLamKchPMC->cd(2));

    ALamKchPMC->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepALamKchPMC->cd(1));
    ALamKchPMC->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepALamKchPMC->cd(2));

    LamKchMMC->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepLamKchMMC->cd(1));
    LamKchMMC->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepLamKchMMC->cd(2));

    ALamKchMMC->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepALamKchMMC->cd(1));
    ALamKchMMC->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepALamKchMMC->cd(2));

    //----------------------------------
    if(bSaveFile)
    {
      LamKchPMC->SaveAllAvgSepHeavyCfs(mySaveFile);
      LamKchMMC->SaveAllAvgSepHeavyCfs(mySaveFile);
      ALamKchPMC->SaveAllAvgSepHeavyCfs(mySaveFile);
      ALamKchMMC->SaveAllAvgSepHeavyCfs(mySaveFile);
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
      canKStarRatios->SaveAs(tSaveDirectoryBase+aName);
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
      canPart1MassFail->SaveAs(tSaveDirectoryBase+aName);
    }
  }




  if(bContainsPurity)
  {
    vector<TString> tPrintPurText{"", "_wPrintPurity"};

    LamKchP->BuildPurityCollection();
    ALamKchP->BuildPurityCollection();
    LamKchM->BuildPurityCollection();
    ALamKchM->BuildPurityCollection();

    TCanvas* canPurity = new TCanvas("canPurity","canPurity");
    canPurity->Divide(2,2);

    LamKchP->DrawAllPurityHistos((TPad*)canPurity->cd(1), bPrintPurity);
    LamKchM->DrawAllPurityHistos((TPad*)canPurity->cd(2), bPrintPurity);
    ALamKchP->DrawAllPurityHistos((TPad*)canPurity->cd(3), bPrintPurity);
    ALamKchM->DrawAllPurityHistos((TPad*)canPurity->cd(4), bPrintPurity);

    if(bSaveFigures)
    {
      TString aName = TString::Format("cLamcKchPurity%s.pdf", tPrintPurText[bPrintPurity].Data());
      canPurity->SaveAs(tSaveDirectoryBase+aName);

      TString aName2 = TString::Format("LamPurity%s_LamKchP.pdf", tPrintPurText[bPrintPurity].Data());
      canPurity->cd(1)->SaveAs(tSaveDirectoryBase+aName2);
    }

  }

  if(bContainsPurity && bDrawMC)
  {
    LamKchPMC->GetMCKchPurity(true);
    LamKchPMC->GetMCKchPurity(false);

    LamKchMMC->GetMCKchPurity(true);
    LamKchMMC->GetMCKchPurity(false);

    ALamKchPMC->GetMCKchPurity(true);
    ALamKchPMC->GetMCKchPurity(false);

    ALamKchMMC->GetMCKchPurity(true);
    ALamKchMMC->GetMCKchPurity(false);
  }


  if(bDrawKchdEdx)
  {
    bool tDrawLogz = false;
    TCanvas* tCandEdX_LamKchP_KchP = LamKchP->DrawKchdEdx(kKchP, tDrawLogz);
    TCanvas* tCandEdX_LamKchP_KchM = LamKchM->DrawKchdEdx(kKchM, tDrawLogz);

    if(bSaveFigures)
    {
      tCandEdX_LamKchP_KchP->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCandEdX_LamKchP_KchP->GetName()));
      tCandEdX_LamKchP_KchM->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCandEdX_LamKchP_KchM->GetName()));
    }
  }


//-------------------------------------------------------------------------------
  cout << "DONE" << endl;
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  if(bSaveFile) {mySaveFile->Close();}

  return 0;
}
