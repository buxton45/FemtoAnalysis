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
  CentralityType tCentType = k0010;
  //-----Data


  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cXiK0_20170615/Results_cXiK0_20170615";
  Analysis* XiK0 = new Analysis(FileLocationBase,kXiK0,tCentType);
  Analysis* AXiK0 = new Analysis(FileLocationBase,kAXiK0,tCentType);


  TString SaveFileName = "~/Analysis/FemtoAnalysis/Results/Results_cXiK0_20170615/0010/Results_cXiK0_20170615_0010.root";

//-----------------------------------------------------------------------------


  bool bSaveFile = false;
  TFile *mySaveFile;
  if(bSaveFile) {mySaveFile = new TFile(SaveFileName, "RECREATE");}

  bool bContainsPurity = false;
  bool bContainsKStarCfs = true;
  bool bContainsAvgSepCfs = false;

  bool bDrawMC = false;

  bool bSaveFigures = false;
  TString tSaveFiguresLocation = "~/Analysis/FemtoAnalysis/Results/Results_cXiK0_20170615/0010/";
  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    XiK0->BuildKStarHeavyCf();
    AXiK0->BuildKStarHeavyCf();

    XiK0->GetKStarHeavyCf()->Rebin(4);
    AXiK0->GetKStarHeavyCf()->Rebin(4);

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(XiK0->GetKStarHeavyCf()->GetHeavyCf(),XiK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(AXiK0->GetKStarHeavyCf()->GetHeavyCf(),AXiK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TCanvas *canKStar = new TCanvas("canKStar","canKStar", 1400, 500);
    canKStar->Divide(2,1);
    gStyle->SetOptTitle(0);

    double tXMin = 0.0;
    double tXMax = 1.0;

    double tYMin = 0.38;
    double tYMax = 1.7;

    XiK0->DrawKStarHeavyCf((TPad*)canKStar->cd(1),1,"",20,tXMin,tXMax,tYMin,tYMax);
    canKStar->cd(1);
    leg1->Draw();


    AXiK0->DrawKStarHeavyCf((TPad*)canKStar->cd(2),1,"",20,tXMin,tXMax,tYMin,tYMax);
    canKStar->cd(2);
    leg2->Draw();

    if(bSaveFigures)
    {
      TString aName = "cXiK0KStarCfs.eps";
      canKStar->SaveAs(tSaveFiguresLocation+aName);
    }

    //----------------------------------
    if(bSaveFile)
    {
      XiK0->SaveAllKStarHeavyCf(mySaveFile);
      AXiK0->SaveAllKStarHeavyCf(mySaveFile);
    }

  }


  if(bContainsAvgSepCfs)
  {
/*
    double aMinNorm = 14.99;
    double aMaxNorm = 19.99;
    int aRebin = 2;

    XiKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
    AXiKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);

    TCanvas *canAvgSepXiKchP = new TCanvas("canAvgSepXiKchP","canAvgSepXiKchP", 2100, 500);
    TCanvas *canAvgSepAXiKchP = new TCanvas("canAvgSepAXiKchP","canAvgSepAXiKchP", 2100, 500);

    gStyle->SetOptTitle(0);
    double tXMargin = 0.01;  //default = 0.01
    double tYMargin = 0.001;  //default = 0.01
    canAvgSepXiKchP->Divide(3,1, tXMargin,tYMargin);
    canAvgSepAXiKchP->Divide(3,1, tXMargin,tYMargin);

    XiKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepXiKchP->cd(1));
    XiKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepXiKchP->cd(2));
    XiKchP->DrawAvgSepHeavyCf(kTrackBac,(TPad*)canAvgSepXiKchP->cd(3),true);

    AXiKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepAXiKchP->cd(1));
    AXiKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepAXiKchP->cd(2));
    AXiKchP->DrawAvgSepHeavyCf(kTrackBac,(TPad*)canAvgSepAXiKchP->cd(3),true);

    //----------------------------------
    if(bSaveFile)
    {
      XiKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
      AXiKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
    }


    if(bSaveFigures)
    {
      TString aName = "cXiK0AvgSepCfs";
      TString aNameXiKchP = aName + TString("_XiKchP.eps");
      TString aNameAXiKchP = aName + TString("_AXiKchP.eps");

      canAvgSepXiKchP->SaveAs(tSaveFiguresLocation+aNameXiKchP);
      canAvgSepAXiKchM->SaveAs(tSaveFiguresLocation+aNameAXiKchM);
    }
*/
  }


  if(bContainsPurity)
  {
    XiK0->BuildPurityCollection();
    AXiK0->BuildPurityCollection();

    TCanvas* canPurity = new TCanvas("canPurity","canPurity");
    canPurity->Divide(2,1);

    XiK0->DrawAllPurityHistos((TPad*)canPurity->cd(1));
    AXiK0->DrawAllPurityHistos((TPad*)canPurity->cd(2));

    if(bSaveFigures)
    {
      TString aName = "cXiK0Purity.eps";
      canPurity->SaveAs(tSaveFiguresLocation+aName);
    }

  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  if(bSaveFile) {mySaveFile->Close();}

  return 0;
}
