#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TH3.h"

#include "ThermCf.h"
#include "FitPartialAnalysis.h"


//________________________________________________________________________________________________________________
TH3* GetThermHist3d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH3 *tReturnHist = (TH3*)tFile->Get(aHistName);
  TH3 *tReturnHistClone = (TH3*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
double FitFunctionGaussian(double *x, double *par)
{
  //4 parameters
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/(sqrt(2)*par[2]),2.0))) + par[3];
}

//________________________________________________________________________________________________________________
TF1* FitwGauss(TH1* aHist, double aMinFit=0., double aMaxFit=50.)
{
  TString tFitName = TString::Format("%s_FitGauss", aHist->GetName());
//  TF1* tReturnFunction = new TF1(tFitName, BackgroundFitter::FitFunctionGaussian, aMinFit, aMaxFit, 4);  //No sqrt(2) with sigma
  TF1* tReturnFunction = new TF1(tFitName, FitFunctionGaussian, aMinFit, aMaxFit, 4);

  double tMaxVal = aHist->GetMaximum();
  double tMaxPos = aHist->GetBinCenter(aHist->GetMaximumBin());
  int tApproxSigBin = aHist->FindLastBinAbove(tMaxVal/2.);
  double tApproxSig = aHist->GetBinCenter(tApproxSigBin);

  tReturnFunction->SetParameter(0, tMaxVal);

  tReturnFunction->SetParameter(1, tMaxPos);
//  tReturnFunction->SetParLimits(1, 0., 50.);
//  tReturnFunction->FixParameter(1, 0.);

  tReturnFunction->SetParameter(2, tApproxSig);

  tReturnFunction->FixParameter(3, 0.);

  aHist->Fit(tFitName, "0", "", aMinFit, aMaxFit);
  return tReturnFunction;
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
  AnalysisType tAnType = kLamKchP;

  bool bCombineConjugates = true;
  bool bSaveFigures = false;

  int tRebin=2;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 2;


  TString tFilaNameBase = "CorrelationFunctions_wOtherPairs";
  TString tFileNameModifier = "";
//  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_OnlyWeightLongDecayParents";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";


  TString tFileName = TString::Format("%s%s.root", tFilaNameBase.Data(), tFileNameModifier.Data());



  //--------------------------------------------

  //ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
  TH1* tThermCf = ThermCf::GetThermCf(tFileName, "PrimaryOnly", tAnType, tImpactParam, true, kMe, tRebin, tMinNorm, tMaxNorm);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  tThermCf->Draw();

  //--------------------------------------------
  double tRef0, tImf0, td0;
  if(tAnType == kLamKchP)
  {
    tRef0 = -1.16;
    tImf0 = 0.51;
    td0 = 1.08;
  }
  else if(tAnType == kLamKchM)
  {
    tRef0 = 0.41;
    tImf0 = 0.47;
    td0 = -4.89;
  }
  else if(tAnType == kLamK0)
  {
    tRef0 = -0.41;
    tImf0 = 0.20;
    td0 = 2.08;
  }
  else assert(0);

  int tNFitParams = 5;
  TString tFitName = "tFitFcn";
  TF1* tFitFcn = new TF1(tFitName, FitPartialAnalysis::LednickyEqWithNorm,0.,0.5,tNFitParams+1);

    tFitFcn->SetParameter(0, 1.);
    tFitFcn->SetParameter(1, 5.);

    tFitFcn->FixParameter(2, tRef0);
    tFitFcn->FixParameter(3, tImf0);
    tFitFcn->FixParameter(4, td0);

    tFitFcn->SetParameter(5, 1.);

  tThermCf->Fit(tFitName, "0", "", 0.0, 0.3);

  tFitFcn->Draw("same");

//-------------------------------------------------------------------------------

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
  TString tFileLocation = TString::Format("%s%s", tFileDir.Data(), tFileName.Data());

  TString tHistName3d = TString::Format("PairSource3d_oslPrimaryOnly%s", cAnalysisBaseTags[tAnType]);

  TH3* tTest3d = GetThermHist3d(tFileLocation, tHistName3d);

  //----------

  TH2D* tSourceSO = (TH2D*)tTest3d->Project3D("yx");
    tSourceSO->SetTitle("Side(y) vs. Out(x)");
    tSourceSO->GetYaxis()->SetTitle("Side");
    tSourceSO->GetXaxis()->SetTitle("Out");

  TH2D* tSourceLO = (TH2D*)tTest3d->Project3D("zx");
    tSourceLO->SetTitle("Long(y) vs. Out(x)");
    tSourceLO->GetYaxis()->SetTitle("Long");
    tSourceLO->GetXaxis()->SetTitle("Out");


  TH2D* tSourceLS = (TH2D*)tTest3d->Project3D("zy");
    tSourceLS->SetTitle("Long(y) vs. Side(x)");
    tSourceLS->GetYaxis()->SetTitle("Long");
    tSourceLS->GetXaxis()->SetTitle("Side");

  TCanvas* tCanSO = new TCanvas("tCanSO", "tCanSO");
  TCanvas* tCanLO = new TCanvas("tCanLO", "tCanLO");
  TCanvas* tCanLS = new TCanvas("tCanLS", "tCanLS");

  tCanSO->cd();
  tSourceSO->Draw("colz");

  tCanLO->cd();
  tSourceLO->Draw("colz");

  tCanLS->cd();
  tSourceLS->Draw("colz");

  //----------
  int tBinProjLow, tBinProjHigh;

  tBinProjLow=-1;
  tBinProjHigh=-1;

//  tBinProjLow = tTest3d->GetXaxis()->FindBin(-1.);
//  tBinProjHigh = tTest3d->GetXaxis()->FindBin(1.);

  TH1D* tSourceO = tTest3d->ProjectionX("Out", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
  TH1D* tSourceS = tTest3d->ProjectionY("Side", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
  TH1D* tSourceL = tTest3d->ProjectionZ("Long", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);

  TCanvas* tCanO = new TCanvas("tCanO", "tCanO");
  TCanvas* tCanS = new TCanvas("tCanS", "tCanS");
  TCanvas* tCanL = new TCanvas("tCanL", "tCanL");


  double tGaussFitMin = -20.;
  double tGaussFitMax = 20.;
  TF1* tGaussFitO = FitwGauss(tSourceO, tGaussFitMin, tGaussFitMax);
  TF1* tGaussFitS = FitwGauss(tSourceS, tGaussFitMin, tGaussFitMax);
  TF1* tGaussFitL = FitwGauss(tSourceL, tGaussFitMin, tGaussFitMax);

  tCanO->cd();
  tSourceO->Draw();
  tGaussFitO->Draw("same");

  tCanS->cd();
  tSourceS->Draw();
  tGaussFitS->Draw("same");

  tCanL->cd();
  tSourceL->Draw();
  tGaussFitL->Draw("same");

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
