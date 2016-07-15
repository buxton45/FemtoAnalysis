#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"

CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
//  myFitter->CalculateChi2PML(npar,f,par);
  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}


//________________________________________________________________________
double GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double LednickyEq(double *x, double *par)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first

  std::complex<double> f0 (par[2],par[3]);
  double Alpha = 0.; // alpha = 0 for non-identical
  double z = 2.*(x[0]/hbarc)*par[1];  //z = 2k*R, to be fed to GetLednickyF1(2)

  double C_QuantumStat = Alpha*exp(-z*z);  // will be zero for my analysis

  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*par[4]*(x[0]/hbarc)*(x[0]/hbarc) - ImI*(x[0]/hbarc),-1);

  double C_FSI = (1+Alpha)*( 0.5*norm(ScattAmp)/(par[1]*par[1])*(1.-1./(2*sqrt(TMath::Pi()))*(par[4]/par[1])) + 2.*real(ScattAmp)/(par[1]*sqrt(TMath::Pi()))*GetLednickyF1(z) - (imag(ScattAmp)/par[1])*GetLednickyF2(z));

  double Cf = 1. + par[0]*(C_QuantumStat + C_FSI);
  Cf *= par[5];

  return Cf;
  
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

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();

  bool bDoFit = true;
  bool bDrawFit = false;
  bool bFakeFit = false;

  double tKStarMin = 0.0;
  double tKStarMax = 0.50;
  double tBinSize = 0.01;
  int tNBinsK = (tKStarMax-tKStarMin)/tBinSize;

//-----------------------------------------------------------------------------

  TString tFileLocationBase = "/home/jesse/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";

  AnalysisType tAnType = kLamKchP;
  AnalysisType tConjType = kALamKchM;

//  AnalysisType tAnType = kLamKchM;
//  AnalysisType tConjType = kALamKchP;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tAnType,k0010);
  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase,tConjType,k0010);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairConjAn0010);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  if(tAnType==kLamKchP || tAnType==kALamKchM)
  {

    tSharedAn->SetSharedParameter(kLambda,0.1923,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,5.031,2.,12.);
    tSharedAn->SetSharedParameter(kRef0,-1.694,-10.,10.);
    tSharedAn->SetSharedParameter(kImf0,1.123,-10.,10.);
    tSharedAn->SetSharedParameter(kd0,3.195,-10.,10.);

/*
    tSharedAn->SetSharedParameter(kLambda,0.4793);
    tSharedAn->SetSharedParameter(kRadius,4.926);
    tSharedAn->SetSharedParameter(kRef0,-0.5223);
    tSharedAn->SetSharedParameter(kImf0,0.4042);
    tSharedAn->SetSharedParameter(kd0,-1.084);
*/
/*
    tSharedAn->SetSharedAndFixedParameter(kLambda,0.479305);
    tSharedAn->SetSharedParameter(kRadius,4.92579);
    tSharedAn->SetSharedParameter(kRef0,-0.522311);
    tSharedAn->SetSharedAndFixedParameter(kImf0,0.404234);
    tSharedAn->SetSharedAndFixedParameter(kd0,-1.08440);
*/
  }

  if(tAnType==kLamKchM || tAnType==kALamKchP)
  {

    tSharedAn->SetSharedParameter(kLambda,0.312);
    tSharedAn->SetSharedParameter(kRadius,3.895);
    tSharedAn->SetSharedParameter(kRef0,0.1146);
    tSharedAn->SetSharedParameter(kImf0,0.4182);
    tSharedAn->SetSharedParameter(kd0,7.277);
  }

  tSharedAn->RebinAnalyses(2);

  tSharedAn->SetFitType(kChi2);

  tSharedAn->SetFixNormParams(true);
  tSharedAn->CreateMinuitParameters();

//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.15);
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,tKStarMax);
//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.02);
    tFitter->SetTurnOffCoulomb(true);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetUseRandomKStarVectors(true);
    tFitter->SetUseStaticPairs(true,100000);

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  if(bDoFit)
  {
    tFitter->DoFit();
    TString tSaveHistName = "Chi2HistogramsMinuit_" + TString(cAnalysisBaseTags[tAnType]) + TString(".root");
    tSharedAn->GetFitChi2Histograms()->SaveHistograms(tSaveHistName);
  }

//_______________________________________________________________________________________________________________________
  if(bDrawFit)
  {
    TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();
    gStyle->SetOptStat(0);

    TString tName = cAnalysisRootTags[tAnType];
    TString tSaveName = cAnalysisBaseTags[tAnType] + TString(".eps");

    double tLambda, tR1, tReF0, tImF0, tD0, tNorm;
    tNorm = 1.;

    if(tAnType==kLamKchP || tAnType==kALamKchM)
    {
      tLambda = 0.1923;
      tR1 = 5.031;
      tReF0 = -1.694;
      tImF0 = 1.123;
      tD0 = 3.195;
    }

    else if (tAnType==kLamKchM || tAnType==kALamKchP)
    {
      tLambda = 0.312;
      tR1 = 3.895;
      tReF0 = 0.1146;
      tImF0 = 0.4182;
      tD0 = 7.277;
    }

    TH1* tSampleHist1 = tFitter->CreateFitHistogramSample("SampleHist1", tAnType, tNBinsK, tKStarMin, tKStarMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
    tSampleHist1->SetDirectory(0);
      tSampleHist1->SetTitle(tName);
      tSampleHist1->SetName(tName);
      tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
      tSampleHist1->GetYaxis()->SetTitle("C(k*)");
      tSampleHist1->SetMarkerStyle(22);
      if(tAnType==kLamKchP || tAnType==kALamKchM)
      {
        tSampleHist1->GetYaxis()->SetRangeUser(0.88,1.02);
        tSampleHist1->SetMarkerColor(2);
        tSampleHist1->SetLineColor(2);
      }
      else if(tAnType==kLamKchM || tAnType==kALamKchP)
      {
        tSampleHist1->GetYaxis()->SetRangeUser(0.88,1.02);
        tSampleHist1->SetMarkerColor(4);
        tSampleHist1->SetLineColor(4);
      }

      tSampleHist1->Draw("p");

      tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
      tPairAn0010->GetKStarCf()->Draw("same");


    TF1* tLednickyFit = new TF1("LednickyEqFit",LednickyEq,0.,0.5,6);
      tLednickyFit->SetParameter(0,0.1923);
      tLednickyFit->SetParameter(1,5.031);
      tLednickyFit->SetParameter(2,-1.694);
      tLednickyFit->SetParameter(3,1.123);
      tLednickyFit->SetParameter(4,3.195);
      tLednickyFit->SetParameter(5,1.);

      tLednickyFit->SetLineColor(2);

    tLednickyFit->Draw("psame");

    TLegend *tLeg = new TLegend(0.65,0.60,0.85,0.75);
      tLeg->AddEntry(tPairAn0010->GetKStarCf(),tName,"p");
      tLeg->AddEntry(tSampleHist1,"NewMethod","p");
      tLeg->AddEntry(tLednickyFit, "OldFit", "l");
      tLeg->Draw();

    tCan->SaveAs(tSaveName);

  }


//-------------------------------------------------------------------------------

  if(bFakeFit)
  {
    double tLambda, tR1, tReF0, tImF0, tD0, tNorm;
    tNorm = 1.;

    if(tAnType==kLamKchP || tAnType==kALamKchM)
    {
      tLambda = 0.1923;
      tR1 = 5.031;
      tReF0 = -1.694;
      tImF0 = 1.123;
      tD0 = 3.195;
    }

    else if (tAnType==kLamKchM || tAnType==kALamKchP)
    {
      tLambda = 0.312;
      tR1 = 3.895;
      tReF0 = 0.1146;
      tImF0 = 0.4182;
      tD0 = 7.277;
    }

    TH1* tFakeCf = tFitter->CreateFitHistogramSample("FakeCf", tAnType, tNBinsK, tKStarMin, tKStarMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
    tFitter->DoFit();
  }

  delete tFitter;

  delete tSharedAn;
  delete tPairAn0010;

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
