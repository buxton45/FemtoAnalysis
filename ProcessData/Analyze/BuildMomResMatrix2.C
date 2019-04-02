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

#include "TMatrixD.h"

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
*/
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
  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
    Analysis* LamKchP = new Analysis(FileLocationBase,kLamKchP,k0010,kGrid,5);

  TString FileLocationBaseMC = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229";
    Analysis* LamKchPMCTot = new Analysis(FileLocationBaseMC,kLamKchP,k0010,kGrid,5);
//-----------------------------------------------------------------------------

  LamKchP->BuildKStarHeavyCf();
  LamKchPMCTot->BuildAllModelKStarTrueVsRecTotal();

  TH1* LamKchPUncorrected = LamKchP->GetKStarHeavyCf()->GetHeavyCfClone();
  TH2* LamKchPTrueVsRecSame = LamKchPMCTot->GetModelKStarTrueVsRecTotal(kSame);
  //NormalizeEachRow(LamKchPTrueVsRecSame);

  //TMatrixD *tMatrixInitial = new TMatrixD(LamKchPTrueVsRecSame->GetNbinsX(),LamKchPTrueVsRecSame->GetNbinsY());
  TMatrixD tMatrixInitial(LamKchPTrueVsRecSame->GetNbinsX(),LamKchPTrueVsRecSame->GetNbinsY());
  TMatrixD tMatrixInitialClone(LamKchPTrueVsRecSame->GetNbinsX(),LamKchPTrueVsRecSame->GetNbinsY());

  for(int i=0; i<LamKchPTrueVsRecSame->GetNbinsX(); i++)
  {
    for(int j=0; j<LamKchPTrueVsRecSame->GetNbinsY(); j++)
    {
      //tMatrixInitial[i][j] = LamKchPTrueVsRecSame->GetBinContent(i+1,j+1);
      tMatrixInitial(i,j) = LamKchPTrueVsRecSame->GetBinContent(i+1,j+1);
      tMatrixInitialClone(i,j) = LamKchPTrueVsRecSame->GetBinContent(i+1,j+1);
    }
  }


  TMatrixD tMatrixFinal = tMatrixInitialClone.Invert();

  TH2D* tMomResFinal = new TH2D("tMomResFinal","tMomResFinal",200,0.,1.,200,0.,1.);

  for(int i=0; i<tMatrixFinal.GetNcols(); i++)
  {
    for(int j=0; j<tMatrixFinal.GetNrows(); j++)
    {
      tMomResFinal->SetBinContent(i+1,j+1,tMatrixFinal(i,j));
    }
  }

/*
  TMatrixD tUnity = tMatrixInitial*tMatrixFinal;
  //TMatrixD tUnity;
  //tUnity.AMultB(tMatrixInitial,tMatrixFinal);
  for(int i=0; i<tUnity.GetNcols(); i++)
  {
    for(int j=0; j<tUnity.GetNrows(); j++)
    {
      if(i==j) cout << "\t\t\t tUnity(" << i << "," << j << ") = " << tUnity(i,j) << endl;
      else cout << "tUnity(" << i << "," << j << ") = " << tUnity(i,j) << endl;
    }
  }
*/

  TH1D* LamKchPCorrected = new TH1D("LamKchPCorrected","LamKchPCorrected",200,0.,1.);

  for(int iRow=1; iRow<=tMomResFinal->GetNbinsY(); iRow++)
  {
    double tValue = 0.;
    for(int iCol=1; iCol<=tMomResFinal->GetNbinsX(); iCol++)
    {
      tValue += LamKchPUncorrected->GetBinContent(iCol)*tMomResFinal->GetBinContent(iCol,iRow);
    }
    tValue /= tMomResFinal->Integral(1,tMomResFinal->GetNbinsX(),iRow,iRow);
    LamKchPCorrected->SetBinContent(iRow,tValue);
    LamKchPCorrected->SetBinError(iRow,LamKchPUncorrected->GetBinError(iRow));
  }

/*
  for(int iRow=0; iRow<tMatrixFinal.GetNrows(); iRow++)
  {
    double tValue = 0.;
    double tIntegral = 0.;
    for(int iCol=0; iCol<tMatrixFinal.GetNcols(); iCol++)
    {
      tValue += LamKchPUncorrected->GetBinContent(iCol+1)*tMatrixFinal(iCol,iRow);
      tIntegral += tMatrixFinal(iCol,iRow);
    }
    tValue /= tIntegral;
    LamKchPCorrected->SetBinContent(iRow+1,tValue);
    LamKchPCorrected->SetBinError(iRow+1,LamKchPUncorrected->GetBinError(iRow+1));
  }
*/
  LamKchPCorrected->SetLineColor(1);
  LamKchPCorrected->SetMarkerColor(1);
  LamKchPCorrected->SetMarkerStyle(20);

  LamKchPUncorrected->SetLineColor(2);
  LamKchPUncorrected->SetMarkerColor(2);
  LamKchPUncorrected->SetMarkerStyle(21);

  TCanvas* aCan = new TCanvas("aCan","aCan");
  aCan->cd();
  LamKchPCorrected->Draw("ep");
  LamKchPUncorrected->Draw("same");

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
