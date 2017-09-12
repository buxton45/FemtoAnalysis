#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TH1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TFile.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "THistPainter.h"
#include <TStyle.h>

#include "Types.h"

//________________________________________________________________________________________________________________
TH1D* Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle)
{
  assert(aCfVec.size() == aKStarBinCenters.size());

  double tBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];
  int tNbins = aKStarBinCenters.size();
  double tKStarMin = aKStarBinCenters[0]-tBinWidth/2.0;
  tKStarMin=0.;
  double tKStarMax = aKStarBinCenters[tNbins-1] + tBinWidth/2.0;

  TH1D* tReturnHist = new TH1D(aTitle, aTitle, tNbins, tKStarMin, tKStarMax);
  for(int i=0; i<tNbins; i++) {tReturnHist->SetBinContent(i+1,aCfVec[i]); tReturnHist->SetBinError(i+1,0.00000000001);}
  //NOTE: Set errors to very small, because if set to zero, just drawing histogram points seems to not work with CanvasPartition package

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH2D* LoadCoulombOnlyInterpCfs(AnalysisType aResType, TString aFileDirectory="/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/")
{
  TString aFileName = aFileDirectory + TString::Format("2dCoulombOnlyInterpCfs_%s.root", cAnalysisBaseTags[aResType]);
  TFile aFile(aFileName);
  TH2D* t2dCoulombOnlyInterpCfs = (TH2D*)aFile.Get(TString::Format("t2dCoulombOnlyInterpCfs_%s", cAnalysisBaseTags[aResType]));
  assert(t2dCoulombOnlyInterpCfs);
  TH2D* tReturnHist = (TH2D*)t2dCoulombOnlyInterpCfs->Clone();
  tReturnHist->SetDirectory(0);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* GetCfFrom2dInterpCfs(AnalysisType aResType, TH2D* a2dCoulombOnlyInterpCfs, td1dVec aKStarBinCenters, double aRadius, double aLambda)
{
  assert(aRadius>0.);
  assert(a2dCoulombOnlyInterpCfs);  //TODO does this really check that 2dhist is loaded?

  assert(a2dCoulombOnlyInterpCfs->GetNbinsX() >= (int)aKStarBinCenters.size());
  assert( abs(a2dCoulombOnlyInterpCfs->GetXaxis()->GetBinWidth(1) - (aKStarBinCenters[1]-aKStarBinCenters[0])) < 0.0000000001 );

  td1dVec tReturnVec(0);
  double tCfValue = -1.;
  for(unsigned int i=0; i<aKStarBinCenters.size(); i++)
  {
    tCfValue = a2dCoulombOnlyInterpCfs->Interpolate(aKStarBinCenters[i], aRadius);
    tReturnVec.push_back(tCfValue);
  }
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] = /*aLambda*(tReturnVec[i]-1.)*/aLambda*tReturnVec[i] + (1.-aLambda);
  TString tReturnName = TString::Format("%sCf_R%0.3f", cAnalysisBaseTags[aResType], aRadius);
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, aKStarBinCenters, tReturnName);
  tReturnHist->SetDirectory(0);
  return tReturnHist;
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

  AnalysisType tAnType = kXiKchP;

  TH2D* t2dCoulombOnlyInterpCfs = LoadCoulombOnlyInterpCfs(tAnType);

  double tRadius = 3.0;
  double tLambda = 0.5;

  td1dVec tKStarBinCenters;
  int tNBinsK = 30;
  double tKLow = 0.0;
  double tKHigh = 0.3;
  double tBinWidth = (tKHigh-tKLow)/tNBinsK;
  for(int i=0; i<tNBinsK; i++) tKStarBinCenters.push_back(tKLow + (i+0.5)*tBinWidth);

  TCanvas *tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  TH1D* tCf = GetCfFrom2dInterpCfs(tAnType, t2dCoulombOnlyInterpCfs, tKStarBinCenters, tRadius, tLambda);
  tCf->SetMarkerStyle(20);
  tCf->SetMarkerColor(1);
  tCf->SetLineColor(1);
  tCf->SetLineStyle(1);
  tCf->SetLineWidth(2);

  tCf->GetXaxis()->SetRangeUser(0.,0.15);

  tCf->Draw("l");


//-------------------------------------------------------------------------------

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}








