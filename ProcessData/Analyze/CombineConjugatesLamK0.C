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
  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/Results_cLamK0_AsRc_20150923";
  Analysis* LamK0 = new Analysis(FileLocationBase,kLamK0,k0010);
  Analysis* ALamK0 = new Analysis(FileLocationBase,kALamK0,k0010);

  vector<PartialAnalysis*> tLamK0wConjVec;
  vector<PartialAnalysis*> tLamK0Vec = LamK0->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tALamK0Vec = ALamK0->GetPartialAnalysisCollection();
  assert(tLamK0Vec.size() == tALamK0Vec.size());
  for(unsigned int i=0; i<tLamK0Vec.size(); i++) {tLamK0wConjVec.push_back(tLamK0Vec[i]);}
  for(unsigned int i=0; i<tALamK0Vec.size(); i++) {tLamK0wConjVec.push_back(tALamK0Vec[i]);}
  Analysis* LamK0wConj = new Analysis("LamK0wConj_0010",tLamK0wConjVec,true);




//-----------------------------------------------------------------------------


  bool bContainsKStar2dCfs = true;

  bool bSaveFigures = true;
  TString tSaveFiguresLocation = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20150923/0010/";
  //-------------------------------------------------------------------

  if(bContainsKStar2dCfs)
  {
    LamK0wConj->BuildKStar2dHeavyCfs();
    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    LamK0wConj->RebinKStar2dHeavyCfs(4);
    LamK0wConj->DrawKStar2dHeavyCfRatios(canKStarRatios);


    if(bSaveFigures)
    {
      TString aName = "LamK0wConjKStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveFiguresLocation+aName);
    }

  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
