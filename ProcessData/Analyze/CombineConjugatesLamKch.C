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

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //-----Data
  //TString FileLocationBase = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";
  TString FileLocationBase = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  Analysis* LamKchP = new Analysis(FileLocationBase,kLamKchP,k0010);
  Analysis* ALamKchP = new Analysis(FileLocationBase,kALamKchP,k0010);
  Analysis* LamKchM = new Analysis(FileLocationBase,kLamKchM,k0010);
  Analysis* ALamKchM = new Analysis(FileLocationBase,kALamKchM,k0010);

  vector<PartialAnalysis*> tLamKchPwConjVec;
  vector<PartialAnalysis*> tLamKchPVec = LamKchP->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tALamKchMVec = ALamKchM->GetPartialAnalysisCollection();
  assert(tLamKchPVec.size() == tALamKchMVec.size());
  for(unsigned int i=0; i<tLamKchPVec.size(); i++) {tLamKchPwConjVec.push_back(tLamKchPVec[i]);}
  for(unsigned int i=0; i<tALamKchMVec.size(); i++) {tLamKchPwConjVec.push_back(tALamKchMVec[i]);}
  Analysis* LamKchPwConj = new Analysis("LamKchPwConj_0010",tLamKchPwConjVec,true);


  vector<PartialAnalysis*> tLamKchMwConjVec;
  vector<PartialAnalysis*> tLamKchMVec = LamKchM->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tALamKchPVec = ALamKchP->GetPartialAnalysisCollection();
  assert(tLamKchMVec.size() == tALamKchPVec.size());
  for(unsigned int i=0; i<tLamKchMVec.size(); i++) {tLamKchMwConjVec.push_back(tLamKchMVec[i]);}
  for(unsigned int i=0; i<tALamKchPVec.size(); i++) {tLamKchMwConjVec.push_back(tALamKchPVec[i]);}
  Analysis* LamKchMwConj = new Analysis("LamKchMwConj_0010",tLamKchMwConjVec,true);


//-----------------------------------------------------------------------------

  bool bContainsKStarCfs = true;
  bool bContainsKStar2dCfs = false;

  bool bSaveFigures = false;
//  TString tSaveFiguresLocation = "~/Analysis/K0Lam/Results_cLamcKch_AsRc_20151007/0010/";
  TString tSaveFiguresLocation = "~/Poster2016/";
  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    LamKchPwConj->BuildKStarHeavyCf();
    LamKchMwConj->BuildKStarHeavyCf();

//    LamKchPwConj->GetKStarHeavyCf()->Rebin(2);
//    LamKchMwConj->GetKStarHeavyCf()->Rebin(2);

    TLegend* leg1 = new TLegend(0.60,0.15,0.85,0.45);
      leg1->SetFillColor(0);
      leg1->SetHeader("0-10% Centrality");
      leg1->AddEntry(LamKchPwConj->GetKStarHeavyCf()->GetHeavyCf(),"#LambdaK+ & #bar{#Lambda}K-","lp");
      leg1->AddEntry(LamKchMwConj->GetKStarHeavyCf()->GetHeavyCf(),"#LambdaK- & #bar{#Lambda}K+","lp");

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");
    gStyle->SetOptTitle(0);

    LamKchPwConj->GetKStarHeavyCf()->GetHeavyCf()->SetTitle("LamKchwConj");

    LamKchPwConj->DrawKStarHeavyCf((TPad*)canKStar->cd(1),2);
    LamKchMwConj->DrawKStarHeavyCf((TPad*)canKStar->cd(1),4,"same");
    canKStar->cd(1);
    leg1->Draw();

    if(bSaveFigures)
    {
      TString aName = "LamKchwConjKStarCf.eps";
      canKStar->SaveAs(tSaveFiguresLocation+aName);
    }


  }


  if(bContainsKStar2dCfs)
  {
    LamKchPwConj->BuildKStar2dHeavyCfs();
    LamKchMwConj->BuildKStar2dHeavyCfs();


    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    canKStarRatios->Divide(1,2);

    LamKchPwConj->RebinKStar2dHeavyCfs(2);
    LamKchMwConj->RebinKStar2dHeavyCfs(2);


    LamKchPwConj->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(1));
    LamKchMwConj->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(2));


    if(bSaveFigures)
    {
      TString aName = "LamKchwConjKStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveFiguresLocation+aName);
    }

  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
