#include "GeneralFitter.h"
#include "TLegend.h"

GeneralFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateFitFunction(npar,f,par);
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

  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161027/Results_cLamcKch_20161027";
  TString tFileLocationBaseMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161027/Results_cLamcKchMC_20161027";

  AnalysisType tAnType = kLamKchP;
  AnalysisType tConjType = kALamKchM;

//  AnalysisType tAnType = kLamKchM;
//  AnalysisType tConjType = kALamKchP;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tAnType,k0010);
  FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tAnType,k1030);
  FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tAnType,k3050);

  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tConjType,k0010);
  FitPairAnalysis* tPairConjAn1030 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tConjType,k1030);
  FitPairAnalysis* tPairConjAn3050 = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tConjType,k3050);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairConjAn0010);
  tVecOfPairAn.push_back(tPairAn1030);
  tVecOfPairAn.push_back(tPairConjAn1030);
  tVecOfPairAn.push_back(tPairAn3050);
  tVecOfPairAn.push_back(tPairConjAn3050);


  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  if(tAnType==kLamKchP || tAnType==kALamKchM)
  {

    tSharedAn->SetSharedParameter(kRadius,{0,1},5.0,2.,12.);
    tSharedAn->SetSharedParameter(kRadius,{2,3},4.5,2.,12.);
    tSharedAn->SetSharedParameter(kRadius,{4,5},4.0,2.,12.);

    tSharedAn->SetSharedParameter(kRef0,-1.694,-10.,10.);
    tSharedAn->SetSharedParameter(kImf0,1.123,-10.,10.);
    tSharedAn->SetSharedParameter(kd0,3.195,-10.,10.);
  }

  if(tAnType==kLamKchM || tAnType==kALamKchP)
  {

    tSharedAn->SetSharedParameter(kLambda,0.312);
    tSharedAn->SetSharedParameter(kRadius,3.895);
    tSharedAn->SetSharedParameter(kRef0,0.1146);
    tSharedAn->SetSharedParameter(kImf0,0.4182);
    tSharedAn->SetSharedParameter(kd0,7.277);
  }

//  tSharedAn->RebinAnalyses(2);

  tSharedAn->SetFitType(kChi2PML);

  tSharedAn->SetFixNormParams(false);
  tSharedAn->CreateMinuitParameters();

  GeneralFitter* tFitter = new GeneralFitter(tSharedAn,tKStarMax);
    tFitter->SetTurnOffCoulomb(true);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetUseRandomKStarVectors(true);
    tFitter->SetUseStaticPairs(true,100000);

    tFitter->SetApplyMomResCorrection(true);
    tFitter->SetApplyNonFlatBackgroundCorrection(true);
    tFitter->SetIncludeResidualCorrelations(true);

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  if(bDoFit)
  {
    tFitter->DoFit();
    TString tSaveHistName = "Chi2HistogramsMinuit_" + TString(cAnalysisBaseTags[tAnType]) + TString(".root");
    tSharedAn->GetFitChi2Histograms()->SaveHistograms(tSaveHistName);
  }

//_______________________________________________________________________________________________________________________


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
