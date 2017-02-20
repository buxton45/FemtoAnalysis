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
  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170217/Results_cXicKch_20170217";
  Analysis* XiKchP = new Analysis(FileLocationBase,kXiKchP,k0010);
  Analysis* AXiKchP = new Analysis(FileLocationBase,kAXiKchP,k0010);
  Analysis* XiKchM = new Analysis(FileLocationBase,kXiKchM,k0010);
  Analysis* AXiKchM = new Analysis(FileLocationBase,kAXiKchM,k0010);

  vector<PartialAnalysis*> tXiKchPwConjVec;
  vector<PartialAnalysis*> tXiKchPVec = XiKchP->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tAXiKchMVec = AXiKchM->GetPartialAnalysisCollection();
  assert(tXiKchPVec.size() == tAXiKchMVec.size());
  for(unsigned int i=0; i<tXiKchPVec.size(); i++) {tXiKchPwConjVec.push_back(tXiKchPVec[i]);}
  for(unsigned int i=0; i<tAXiKchMVec.size(); i++) {tXiKchPwConjVec.push_back(tAXiKchMVec[i]);}
  Analysis* XiKchPwConj = new Analysis("XiKchPwConj_0010",tXiKchPwConjVec,true);


  vector<PartialAnalysis*> tXiKchMwConjVec;
  vector<PartialAnalysis*> tXiKchMVec = XiKchM->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tAXiKchPVec = AXiKchP->GetPartialAnalysisCollection();
  assert(tXiKchMVec.size() == tAXiKchPVec.size());
  for(unsigned int i=0; i<tXiKchMVec.size(); i++) {tXiKchMwConjVec.push_back(tXiKchMVec[i]);}
  for(unsigned int i=0; i<tAXiKchPVec.size(); i++) {tXiKchMwConjVec.push_back(tAXiKchPVec[i]);}
  Analysis* XiKchMwConj = new Analysis("XiKchMwConj_0010",tXiKchMwConjVec,true);


//-----------------------------------------------------------------------------

  bool bContainsKStarCfs = true;
  bool bContainsKStar2dCfs = false;

  bool bSaveFigures = true;
  TString tSaveFiguresLocation = "~/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170217/0010/";
  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    XiKchPwConj->BuildKStarHeavyCf();
    XiKchMwConj->BuildKStarHeavyCf();

    XiKchPwConj->GetKStarHeavyCf()->Rebin(2);
    XiKchMwConj->GetKStarHeavyCf()->Rebin(2);

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(XiKchPwConj->GetKStarHeavyCf()->GetHeavyCf(),"#Xi-K+ and #bar{#Xi}-K-","lp");
      leg1->AddEntry(XiKchMwConj->GetKStarHeavyCf()->GetHeavyCf(),"#Xi-K- and #bar{#Xi}-K+","lp");

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");

    XiKchPwConj->GetKStarHeavyCf()->GetHeavyCf()->SetTitle("XiKchwConj");

    XiKchPwConj->DrawKStarHeavyCf((TPad*)canKStar->cd(1),2);
    XiKchMwConj->DrawKStarHeavyCf((TPad*)canKStar->cd(1),4,"same");
    canKStar->cd(1);
    leg1->Draw();

    if(bSaveFigures)
    {
      TString aName = "XiKchwConjKStarCf_0010.eps";
      canKStar->SaveAs(tSaveFiguresLocation+aName);
    }



  }

  if(bContainsKStar2dCfs)
  {
    XiKchPwConj->BuildKStar2dHeavyCfs();
    XiKchMwConj->BuildKStar2dHeavyCfs();


    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    canKStarRatios->Divide(1,2);

    XiKchPwConj->RebinKStar2dHeavyCfs(2);
    XiKchMwConj->RebinKStar2dHeavyCfs(2);


    XiKchPwConj->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(1));
    XiKchMwConj->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(2));


    if(bSaveFigures)
    {
      TString aName = "XiKchwConjKStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveFiguresLocation+aName);
    }

  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
