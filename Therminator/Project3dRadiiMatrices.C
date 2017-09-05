#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

//________________________________________________________________________________________________________________
void DrawProject3dMatrixTo1d(TPad* aPad, TH3D* a3dMatrix)
{
  aPad->cd();
//  gStyle->SetOptStat(111111);

  TString tName = TString(a3dMatrix->GetName()) + TString("_pz");
  TH1D* t1dHisto = a3dMatrix->ProjectionZ(tName, 1, a3dMatrix->GetNbinsX(), 1, a3dMatrix->GetNbinsY());

  t1dHisto->GetXaxis()->SetTitle("Radius (fm)");
  t1dHisto->GetYaxis()->SetTitle("Counts");

  t1dHisto->DrawCopy();

cout << "t1dHisto->GetMean() = " << t1dHisto->GetMean() << endl;
}

//________________________________________________________________________________________________________________
void Draw2dRadiiVsBeta(TPad* aPad, TH3D* a3dMatrix)
{
  aPad->cd();
//  gStyle->SetOptStat(111111);

  a3dMatrix->GetYaxis()->SetRange(1, a3dMatrix->GetNbinsY());
  a3dMatrix->GetZaxis()->SetRange(1, a3dMatrix->GetNbinsZ());

  TH2D* t2dHisto = (TH2D*)a3dMatrix->Project3D("zy");

  t2dHisto->GetXaxis()->SetTitle("#Beta");
  t2dHisto->GetYaxis()->SetTitle("Radius");

  t2dHisto->DrawCopy("colz");
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

/*
  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";
*/

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocationPairFractions = tDirectory + "testPairFractions.root";

  TH3D* t3dProtonRadii = Get3dHisto(tFileLocationPairFractions, "f3dProtonRadii");

  //X-axis = tBin = parents bin
  //Y-axis = tBeta
  //Z-axis = tDecayLength


  TCanvas* tCan_1dProtonRadii = new TCanvas("tCan_1dProtonRadii", "tCan_1dProtonRadii");
  DrawProject3dMatrixTo1d((TPad*)tCan_1dProtonRadii, t3dProtonRadii);



  TCanvas* tCan_2dBetaVsRadii = new TCanvas("tCan_2dBetaVsRadii", "tCan_2dBetaVsRadii");
  Draw2dRadiiVsBeta((TPad*)tCan_2dBetaVsRadii, t3dProtonRadii);




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




