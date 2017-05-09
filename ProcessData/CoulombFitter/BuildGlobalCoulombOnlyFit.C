#include "FitSharedAnalyses.h"
#include "GlobalCoulombFitter.h"
#include "TLegend.h"

GlobalCoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
//  myFitter->CalculateChi2PMLwMomResCorrection(npar,f,par);
//  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //!!!!!!!!!!!!!!!! NOTE:  must set myFitter = to whichever LednickyFitter object I want to use

//-----------------------------------------------------------------------------
//Be sure to set the following...

  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170423/Results_cXicKch_20170423";

  AnalysisType tAnType, tConjType;
  AnalysisType tAnTypeOppSign, tConjTypeOppSign;

  tAnType = kXiKchP;
  tConjType = kAXiKchM;

  tAnTypeOppSign = kXiKchM;
  tConjTypeOppSign = kAXiKchP;



  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 2;

  bool bIncludeSingletAndTriplet=false;
  bool bSharedLambdas=true;

//-----------------------------------------------------------------------------

  FitPairAnalysis* tPairAn = new FitPairAnalysis(tFileLocationBase, tAnType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairConjAn = new FitPairAnalysis(tFileLocationBase, tConjType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);

//-----------------------------------------------------------------------------

  FitPairAnalysis* tPairAnOppSign = new FitPairAnalysis(tFileLocationBase, tAnTypeOppSign, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairConjAnOppSign = new FitPairAnalysis(tFileLocationBase, tConjTypeOppSign, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);

//-----------------------------------------------------------------------------

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn);
  tVecOfPairAn.push_back(tPairConjAn);
  tVecOfPairAn.push_back(tPairAnOppSign);
  tVecOfPairAn.push_back(tPairConjAnOppSign);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

//-----------------------------------------------------------------------------
  tSharedAn->SetSharedAndFixedParameter(kRef0, 0.);
  tSharedAn->SetSharedAndFixedParameter(kImf0, 0.);
  tSharedAn->SetSharedAndFixedParameter(kd0, 0.);

  tSharedAn->SetSharedParameter(kRadius, {0,1,2,3}, 3.0, 1.0, 10.0);

  if(bSharedLambdas)
  {
    tSharedAn->SetSharedParameter(kLambda,{0,1},0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kLambda,{2,3},0.5,0.1,1.);
  }


  tSharedAn->RebinAnalyses(2);
  tSharedAn->CreateMinuitParameters();

  GlobalCoulombFitter* tFitter = new GlobalCoulombFitter(tSharedAn,0.30);
    tFitter->SetIncludeSingletAndTriplet(bIncludeSingletAndTriplet);
    tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  TString tFileLocationInterpHistosRepulsive = tFileLocationInterpHistos = "InterpHistsRepulsive";
  TString tFileLocationInterpHistosAttractive = tFileLocationInterpHistos = "InterpHistsAttractive";

  tFitter->LoadInterpHistFile(tFileLocationInterpHistosRepulsive);
  tFitter->LoadInterpHistFileOppSign(tFileLocationInterpHistosAttractive);

  //-------------------------------------------

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,16384);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  tFitter->DoFit();

  delete tFitter;

  delete tSharedAn;
  delete tPairAn;
  delete tPairConjAn;
  delete tPairAnOppSign;
  delete tPairConjAnOppSign;

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE);
  return 0;
}
