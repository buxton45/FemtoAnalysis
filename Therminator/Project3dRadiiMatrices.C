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
TH2D* Get2dRadiiVsPid(TH3D* a3dMatrix)
{
  a3dMatrix->GetYaxis()->SetRange(1, a3dMatrix->GetNbinsY());
  a3dMatrix->GetZaxis()->SetRange(1, a3dMatrix->GetNbinsZ());

  TH2D* t2dHisto = (TH2D*)a3dMatrix->Project3D("zx");

  t2dHisto->GetXaxis()->SetTitle("PID");
  t2dHisto->GetYaxis()->SetTitle("Radius");

  return t2dHisto;
}

//________________________________________________________________________________________________________________
void Draw2dRadiiVsPid(TPad* aPad, TH3D* a3dMatrix)
{
  aPad->cd();
//  gStyle->SetOptStat(111111);

  TH2D* t2dHisto = Get2dRadiiVsPid(a3dMatrix);

  t2dHisto->DrawCopy("colz");
}


//________________________________________________________________________________________________________________
TH1D* Get1dRadiiForParticularParent(TH3D* a3dMatrix, int aType, int aParentType)
{
  vector<int> tFathers = GetParentsPidVector(static_cast<ParticlePDGType>(aType));
  int tBin = -1;
  for(unsigned int i=0; i<tFathers.size(); i++) if(tFathers[i] == aParentType) tBin = i+1;
  assert(tBin > -1);

  TString tReturnName = TString::Format("%s from %s Radii (fm)", GetPDGRootName(static_cast<ParticlePDGType>(aType)), GetParticleName(aParentType).Data());
  TH1D* tReturnHist = a3dMatrix->ProjectionZ(tReturnName.Data(), tBin, tBin, 1, a3dMatrix->GetNbinsY());
    tReturnHist->SetName(tReturnName);
    tReturnHist->SetTitle(tReturnName);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
void DrawRadiiForParticularParent(TPad* aPad, TH3D* a3dMatrix, int aType, int aParentType)
{
  gStyle->SetOptStat(1111);

  TH1D* tHistToDraw = Get1dRadiiForParticularParent(a3dMatrix, aType, aParentType);
  aPad->cd();
  tHistToDraw->DrawCopy();
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


  TCanvas* tCan_RadiiVsPid = new TCanvas("tCan_RadiiVsPid", "tCan_RadiiVsPid");
  Draw2dRadiiVsPid((TPad*)tCan_RadiiVsPid, t3dProtonRadii);

  TCanvas* tCan_CondensedRadiiVsPid = new TCanvas("tCan_CondensedRadiiVsPid", "tCan_CondensedRadiiVsPid");
  TH2D* tCondensedRadiiVsPid = Get2dRadiiVsPid(t3dProtonRadii);
  DrawCondensed2dRadiiVsPid(kPDGProt, (TPad*)tCan_CondensedRadiiVsPid, tCondensedRadiiVsPid);


  TCanvas* tCan_RadiiForParent = new TCanvas("tCan_RadiiForParent", "tCan_RadiiForParent");
  DrawRadiiForParticularParent((TPad*)tCan_RadiiForParent, t3dProtonRadii, kPDGProt, 2224);

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}




