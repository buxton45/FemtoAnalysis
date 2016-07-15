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
/*
//________________________________________________________________________________________________________________
void NormalizeEachColumn(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  for(int i=1; i<=tNbinsX; i++)
  {
    double tScale = aHisto->Integral(i,i,1,tNbinsY);
    if(tScale > 0.)
    {
      for(int j=1; j<=tNbinsY; j++)
      {
        double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
        aHisto->SetBinContent(i,j,tNewContent);
      }
    }
  }
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
*/
//________________________________________________________________________________________________________________
extern double FitGaus(double *x, double *par);
/*
{
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0)));
}
*/
//________________________________________________________________________________________________________________
extern double FitPoly(double *x, double *par);
/*
{
  return par[0]*x[0]*x[0];
}
*/
extern double FitQuadratic(double *x, double *par);
//________________________________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

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



  LamKchPMCTot->BuildAllModelKStarTrueVsRecTotal();

  TH2* LamKchPTrueVsRecRotSame = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kRotSame);
  TH2* LamKchPTrueVsRecRotMixed = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kRotMixed);

/*
  NormalizeEachColumn(LamKchPTrueVsRecRotSame);
  NormalizeEachColumn(LamKchPTrueVsRecRotMixed);
*/

  TCanvas *canKStarTrueVsRec = new TCanvas("canKStarTrueVsRec","canKStarTrueVsRec");
  canKStarTrueVsRec->Divide(2,1);
  //gStyle->SetOptStat(0);

  canKStarTrueVsRec->cd(1);
    gPad->SetLogz();
    LamKchPTrueVsRecRotSame->Draw("colz");

  canKStarTrueVsRec->cd(2);
    gPad->SetLogz();
    LamKchPTrueVsRecRotMixed->Draw("colz");

//---------------------------------
/*
  TH1D* SameProjY = LamKchPTrueVsRecRotSame->ProjectionY("SameProjY",1,20);
  TH1D* SameProjX = LamKchPTrueVsRecRotSame->ProjectionX("SameProjX",1,200);
*/
  TH1D* SameProjY = LamKchPTrueVsRecRotMixed->ProjectionY("SameProjY",1,20);
  TH1D* SameProjX = LamKchPTrueVsRecRotMixed->ProjectionX("SameProjX",1,200);

  TF1* tFitSameProjY = new TF1("tFitSameProjY",FitGaus,-0.1,0.1,3);
    tFitSameProjY->SetParameter(0,23800);
    tFitSameProjY->SetParameter(1,0.0001307);
    tFitSameProjY->SetParameter(2,0.00305);
  SameProjY->Fit("tFitSameProjY","0R");

  TF1* tFitSameProjX = new TF1("tFitSameProjX",FitQuadratic,0.,0.15,1);
  SameProjX->Fit("tFitSameProjX","0R");

  TCanvas* tCanSameProj = new TCanvas("tCanSameProj","tCanSameProj");
  tCanSameProj->Divide(2,1);

  tCanSameProj->cd(1);
  SameProjY->GetXaxis()->SetRangeUser(-0.1,0.1);
  SameProjY->Draw();
  tFitSameProjY->Draw("same");


  tCanSameProj->cd(2);
  SameProjX->Draw();
  tFitSameProjX->Draw("same");


//---------------------------------
  TH2D* tMomResSame = new TH2D("tMomResSame","tMomResSame",100,0.,0.1,100,0.,0.1);
  double tSum, tDiff;
  double tKTrue, tKRec;

  TH2D* tMomResSame2 = new TH2D("tMomResSame2","tMomResSame2",100,0.,0.1,100,0.,0.1);
  double tSum2, tDiff2;
  double tKTrue2, tKRec2;

  for(int i=0; i<100000000; i++)
  {
    tSum = tFitSameProjX->GetRandom();
    tDiff =  tFitSameProjY->GetRandom();

    tKTrue = tSum - tDiff/sqrt(2);
    tKRec = tSum + tDiff/sqrt(2);

    tMomResSame->Fill(tKTrue,tKRec);

    //-------------------------------

    tSum2 = tFitSameProjX->GetRandom();
    tDiff2 =  SameProjY->GetRandom();

    tKTrue2 = tSum2 - tDiff2/sqrt(2);
    tKRec2 = tSum2 + tDiff2/sqrt(2);

    tMomResSame2->Fill(tKTrue2,tKRec2);

  }

  TCanvas* tCanMomResSame = new TCanvas("tCanMomResSame","tCanMomResSame");
  tCanMomResSame->Divide(2,1);

  tCanMomResSame->cd(1);
  gPad->SetLogz();
  //tMomResSame->RebinX(5);
  //tMomResSame->RebinY(5);
  //NormalizeEachRow(tMomResSame);
  tMomResSame->Draw("colz");

  tCanMomResSame->cd(2);
  gPad->SetLogz();
  tMomResSame2->Draw("colz");


  cout << "DONE!" << endl;

//-------------------------------------------------------------------------------
  TCanvas* tCanCorrected = new TCanvas("tCanCorrected","tCanCorrected");
  tCanCorrected->cd();

//  LamKchPMCTot->BuildModelCfTrueIdealCfTrueRatio(/*0.32,0.40,2*/);
//  TH1* LamKchPTrueIdealOG = LamKchPMCTot->GetModelKStarHeavyCfTrueIdeal()->GetHeavyCfClone();

  LamKchPMCTot->BuildModelCfFakeIdealCfFakeRatio(/*0.32,0.40,2*/);
  TH1* LamKchPTrueIdealOG = LamKchPMCTot->GetModelKStarHeavyCfFakeIdeal()->GetHeavyCfClone();

  TH1D* LamKchPTrueIdeal = new TH1D("LamKchPTrueIdeal","LamKchPTrueIdeal",20,0.,0.1);
  LamKchPTrueIdeal->Sumw2();
  for(int i=1; i<=LamKchPTrueIdeal->GetNbinsX(); i++) 
  {
    LamKchPTrueIdeal->SetBinContent(i,LamKchPTrueIdealOG->GetBinContent(i));
    LamKchPTrueIdeal->SetBinError(i,LamKchPTrueIdealOG->GetBinError(i));
  }

    LamKchPTrueIdeal->SetLineColor(1);
    LamKchPTrueIdeal->SetMarkerColor(1);
    LamKchPTrueIdeal->SetMarkerStyle(20);

  TH1D* LamKchPRec = new TH1D("LamKchPRec","LamKchPRec",20,0.,0.1);
  LamKchPRec->Sumw2();
    LamKchPRec->SetLineColor(2);
    LamKchPRec->SetMarkerColor(2);
    LamKchPRec->SetMarkerStyle(20);

  tMomResSame->RebinY(5);
  tMomResSame->RebinX(5);

  for(int j=1; j<=LamKchPTrueIdeal->GetNbinsX(); j++)
  {
    double tValue = 0.;
    for(int i=1; i<=tMomResSame->GetNbinsX(); i++)
    {
      tValue += LamKchPTrueIdeal->GetBinContent(i)*tMomResSame->GetBinContent(i,j);
    }
    tValue /= tMomResSame->Integral(1,tMomResSame->GetNbinsX(),j,j);
    LamKchPRec->SetBinContent(j,tValue);
  }
  LamKchPRec->Draw("ep");
  LamKchPTrueIdeal->Draw("epsame");

//-------------------------------------------------------------------------------

  TCanvas* tCanCorrected2 = new TCanvas("tCanCorrected2","tCanCorrected2");
  tCanCorrected2->cd();

  TH1D* tLamKchPCorrected = (TH1D*)LamKchPTrueIdeal->Clone("tLamKchPCorrected");
  tLamKchPCorrected->Divide(LamKchPRec);
  tLamKchPCorrected->Draw();
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
