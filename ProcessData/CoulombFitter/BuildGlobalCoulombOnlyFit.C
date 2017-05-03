#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"

CoulombFitter *myFitter = NULL;

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
  //tAnType = kXiKchP;
  tAnType = kXiKchM;

  if(tAnType==kXiKchP) tConjType = kAXiKchM;
  else if(tAnType==kXiKchM) tConjType = kAXiKchP;
  else assert(0);

  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 5;
  if(tAnalysisRunType==kTrain || tAnalysisRunType==kTrainSys) tNPartialAnalysis = 2;

  bool bIncludeSingletAndTriplet=false;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);


//-----------------------------------------------------------------------------

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase, tAnType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase, tConjType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairConjAn0010);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {

    tSharedAn->SetSharedParameter(kLambda,{0,1},0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kLambda,{2,3},0.5,0.1,1.);
//    tSharedAn->SetSharedParameter(kLambda,{4,5},0.5,0.1,1.);

    tSharedAn->SetSharedParameter(kRadius,{0,1},4.0,1.,6.);
    tSharedAn->SetSharedParameter(kRadius,{2,3},3.0,1.,6.);
//    tSharedAn->SetSharedParameter(kRadius,{4,5},2.0,1.,6.);

    tSharedAn->SetSharedParameter(kRef0,1.02,-3.,3.);
    tSharedAn->SetSharedParameter(kImf0,0.14,-3.,3.);
    tSharedAn->SetSharedParameter(kd0,0.,-5.,5.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,0.48,-3.,3.);
      tSharedAn->SetSharedParameter(kImf02,0.17,-3.,3.);
      tSharedAn->SetSharedParameter(kd02,0.,-3.,3.);
    }
  }

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {


    tSharedAn->SetSharedParameter(kLambda,{0,1},0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kLambda,{2,3},0.5,0.1,1.);
//    tSharedAn->SetSharedParameter(kLambda,{4,5},0.5,0.1,1.);

    tSharedAn->SetSharedParameter(kRadius,{0,1},4.0,1.,6.);
    tSharedAn->SetSharedParameter(kRadius,{2,3},3.0,1.,6.);
//    tSharedAn->SetSharedParameter(kRadius,{4,5},2.0,1.,6.);

    tSharedAn->SetSharedParameter(kRef0,-0.2,-3.,3.);
    tSharedAn->SetSharedParameter(kImf0,0.2,-3.,3.);
    tSharedAn->SetSharedParameter(kd0,0.,-5.,5.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,-0.2,-3.,3.);
      tSharedAn->SetSharedParameter(kImf02,0.2,-3.,3.);
      tSharedAn->SetSharedParameter(kd02,0.,-5.,5.);
    }
  }




  tSharedAn->RebinAnalyses(2);

//  tSharedAn->SetFitType(kChi2);

  tSharedAn->CreateMinuitParameters();

//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.15);
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.30);
//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.02);
    tFitter->SetIncludeSingletAndTriplet(bIncludeSingletAndTriplet);
    tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,16384);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;


  tFitter->DoFit();







  delete tFitter;

  delete tSharedAn;
  delete tPairAn0010;

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE);
  return 0;
}
