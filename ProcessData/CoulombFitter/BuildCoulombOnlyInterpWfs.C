/* BuildCoulombOnlyInterWfs.C
Interpolation Wave Functions (Wfs) */

#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"


#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"
#include <TLatex.h>
#include "TGraphAsymmErrors.h"
#include "TFile.h"
/*
CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
//  myFitter->CalculateChi2PMLwMomResCorrection(npar,f,par);
//  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}
*/



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  bool bVerboseL1 = true;
  bool bVerboseL2 = false;
  bool bVerboseL3 = false; 

  AnalysisType tAnType = kXiKchP;

  double tLambda = 1.0;
  double tNorm = 1.;
  double tReF0=0., tImF0=0., tD0=0.;

  int tNbinsKStar = 200;
  double tKStarMin = 0.;
  double tKStarMax = 1.;

  int tNbinsRStar = 400;
  double tRStarMin = 0.;
  double tRStarMax = 100.;

  double tBinWidthTheta = M_PI/180;
  int tNbinsTheta = 182;  //one bin under 0 and one bin over pi, to be safe
  double tThetaMin = 0. - 1.*tBinWidthTheta;
  double tThetaMax = 1.*M_PI + 1.*tBinWidthTheta;


  TH3D* t3dCoulombOnlyInterpWfs = new TH3D(TString::Format("t3dCoulombOnlyInterpWfs_%s", cAnalysisBaseTags[tAnType]),
                                           TString::Format("t3dCoulombOnlyInterpWfs_%s", cAnalysisBaseTags[tAnType]), 
                                           tNbinsKStar, tKStarMin, tKStarMax,
                                           tNbinsRStar, tRStarMin, tRStarMax,
                                           tNbinsTheta, tThetaMin, tThetaMax);

  CoulombFitter* tFitter = new CoulombFitter(tAnType, 1.0);

  TString tFileLocationInterpHistos = "InterpHists";
    tFileLocationInterpHistos += TString::Format("_%s", cAnalysisBaseTags[tAnType]);
  TString tFileLocationLednickyHFile = "LednickyHFunction";
    tFileLocationLednickyHFile += TString::Format("_%s", cAnalysisBaseTags[tAnType]);
  TString tSaveName = TString::Format("3dCoulombOnlyInterpWfs_%s.root", cAnalysisBaseTags[tAnType]);

  tFitter->LoadInterpHistFile(tFileLocationInterpHistos, tFileLocationLednickyHFile);

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,50000);
  tFitter->SetIncludeSingletAndTriplet(false);
/*
  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;
*/

  TAxis* tKaxis = t3dCoulombOnlyInterpWfs->GetXaxis();
  TAxis* tRaxis = t3dCoulombOnlyInterpWfs->GetYaxis();
  TAxis* tThetaaxis = t3dCoulombOnlyInterpWfs->GetZaxis();

  double tWfSq = -1.;
  double tKStar=-1., tRStar=-1., tTheta=-1.;
  for(int iK=0; iK<tNbinsKStar; iK++)
  {
    tKStar = tKaxis->GetBinCenter(iK+1); //histograms start at 1, not 0
    if(bVerboseL1) cout << "\tiK+1 = " << iK+1 << "  |  tKStar = " << tKStar << endl;
    for(int iR=0; iR<tNbinsRStar; iR++)
    {
      tRStar = tRaxis->GetBinCenter(iR+1);
      if(bVerboseL2) cout << "\t\tiR+1 = " << iR+1 << "  |  tRStar = " << tRStar << endl;
      for(int iTheta=0; iTheta<tNbinsTheta; iTheta++)
      {
        tTheta = tThetaaxis->GetBinCenter(iTheta+1);
        if(bVerboseL3) cout << "\t\t\tiTheta+1 = " << iTheta+1 << "  |  tTheta = " << tTheta << endl;

        tWfSq = tFitter->InterpolateWfSquared(tKStar, tRStar, tTheta, tReF0, tImF0, tD0);
        t3dCoulombOnlyInterpWfs->SetBinContent(iK+1, iR+1, iTheta+1, tWfSq);
      }
    }
  }


  TCanvas* tTestCan = new TCanvas("tTestCan", "tTestCan");
  tTestCan->cd();
  t3dCoulombOnlyInterpWfs->Draw(); 



  TFile* tSaveFile = new TFile(tSaveName, "RECREATE");
  t3dCoulombOnlyInterpWfs->Write();
  tSaveFile->Close();

  delete tFitter;
//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}





