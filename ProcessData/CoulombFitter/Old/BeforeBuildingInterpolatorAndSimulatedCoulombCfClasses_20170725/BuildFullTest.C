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
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //!!!!!!!!!!!!!!!! NOTE:  must set myFitter = to whichever LednickyFitter object I want to use

  vector<int> Share01(2);
    Share01[0] = 0;
    Share01[1] = 1;

  vector<int> Share23(2);
    Share23[0] = 2;
    Share23[1] = 3;

  vector<int> Share45(2);
    Share45[0] = 4;
    Share45[1] = 5;
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

  bool bDoFit = true;
  bool bDrawFit = false;
  bool bDrawLam = false;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);


//-----------------------------------------------------------------------------

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase, tAnType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase, tConjType, k0010, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase, tAnType, k1030, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
  FitPairAnalysis* tPairConjAn1030 = new FitPairAnalysis(tFileLocationBase, tConjType, k1030, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
//  FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase, tAnType, k3050, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
//  FitPairAnalysis* tPairConjAn3050 = new FitPairAnalysis(tFileLocationBase, tConjType, k3050, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairConjAn0010);
  tVecOfPairAn.push_back(tPairAn1030);
  tVecOfPairAn.push_back(tPairConjAn1030);
//  tVecOfPairAn.push_back(tPairAn3050);
//  tVecOfPairAn.push_back(tPairConjAn3050);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
/*
    //0607100.pdf
    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,5.,1.,10.);
    tSharedAn->SetSharedParameter(kRef0,1.46,-5.,5.);
    tSharedAn->SetSharedParameter(kImf0,0.24,-5.,5.);
    tSharedAn->SetSharedParameter(kd0,0.,-5.,5.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,0.74,-5.,5.);
      tSharedAn->SetSharedParameter(kImf02,0.40,-5.,5.);
      tSharedAn->SetSharedParameter(kd02,0.,-5.,5.);
    }
*/

    //PhysRevD.80.094006.pdf
/*
    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    //tSharedAn->SetSharedParameter(kRadius,5.,1.,10.);
    tSharedAn->SetSharedParameter(kRadius,{0,1},5.0,2.,12.);
    tSharedAn->SetSharedParameter(kRadius,{2,3},4.5,2.,12.);
    tSharedAn->SetSharedParameter(kRadius,{4,5},4.0,2.,12.);


    tSharedAn->SetSharedParameter(kRef0,1.02,-5.,5.);
    tSharedAn->SetSharedParameter(kImf0,0.14,-5.,5.);
    tSharedAn->SetSharedParameter(kd0,0.,-5.,5.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,0.48,-5.,5.);
      tSharedAn->SetSharedParameter(kImf02,0.17,-5.,5.);
      tSharedAn->SetSharedParameter(kd02,0.,-5.,5.);
    }
*/
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
/*
    //0607100.pdf
    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,5.,1.,10.);
    tSharedAn->SetSharedParameter(kRef0,0.57,-5.,5.);
    tSharedAn->SetSharedParameter(kImf0,0.,-5.,5.);
    tSharedAn->SetSharedParameter(kd0,0.,-10.,10.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,-0.32,-5.,5.);
      tSharedAn->SetSharedParameter(kImf02,0.,-5.,5.);
      tSharedAn->SetSharedParameter(kd02,0.,-10.,10.);
    }
*/

    //PhysRevD.80.094006.pdf
/*
    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,5.,1.,10.);
    tSharedAn->SetSharedParameter(kRef0,0.,-5.,5.);
    tSharedAn->SetSharedParameter(kImf0,0.,-5.,5.);
    tSharedAn->SetSharedParameter(kd0,0.,-10.,10.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,-0.26,-5.,5.);
      tSharedAn->SetSharedParameter(kImf02,0.,-5.,5.);
      tSharedAn->SetSharedParameter(kd02,0.,-10.,10.);
    }
*/

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
  TString tPairKStarNtupleDirName = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/NTuples/Roman";
  TString tFileBaseName = "Results_cXicKch_20160610";
  TString tOutputName = "PairKStar3dVec_20160610_";
  int tNFiles = 27;


//  tFitter->BuildPairKStar3dVecFull(tPairKStarNtupleDirName,tFileBaseName,tNFiles,kAXiKchP,k0010,16,0.,0.16);
//  tFitter->WritePairKStar3dVecFile(tOutputName,tPairKStarNtupleDirName,tFileBaseName,tNFiles,kAXiKchP,k0010,16,0.,0.16);
//  tFitter->WriteAllPairKStar3dVecFiles(tOutputName,tPairKStarNtupleDirName,tFileBaseName,tNFiles,16,0.,0.16);
//  tFitter->WriteAllPairKStar3dVecFiles(tOutputName,tPairKStarNtupleDirName,tFileBaseName,tNFiles,32,0.,0.32);

//  tFitter->BuildPairKStar3dVecFromTxt(tOutputName);

//  tFitter->BuildPairKStar4dVecFromTxt(tOutputName);
  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,16384);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  if(bDoFit)
  {
    tFitter->DoFit();
    TString tSaveHistName = "Chi2HistogramsMinuit_" + TString(cAnalysisBaseTags[tAnType]) + TString(".root");
    tSharedAn->GetFitChi2Histograms()->SaveHistograms(tSaveHistName);
  }

/*
  cout << "Norm values for analyses of type: " << tAnBaseName << endl;
  for(unsigned int iPairAn=0; iPairAn<tVecOfPairAn.size(); iPairAn++)
    {
      for(int iPartAn=0; iPartAn<tSharedAn->GetFitPairAnalysis(iPairAn)->GetNFitPartialAnalysis(); iPartAn++)
        {
          cout << "\tiPairAn = " << iPairAn << "\tiPartAn = " << iPartAn << "\tNorm = " << tSharedAn->GetFitPairAnalysis(iPairAn)->GetFitPartialAnalysis(iPartAn)->GetFitParameter(kNorm)->GetFitValue() << endl;
        }
    }

  TCanvas* canPairAn = new TCanvas("canPairAn","canPairAn");

  canPairAn->cd(1);
  tSharedAn->DrawFit(0,tAnName0010);
*/
/*
  int tNbinsK = 30;
  double tKMin = 0.;
  double tKMax = 0.30;

  double tLambda = 0.5;

  double tR1 = 5.25;
  double tR2 = 7.0;
  double tR3 = 8.0;

  double tReF0 = 1.46;
  double tImF0 = 0.24;
  double tD0 = -1.;
  double tNorm = 1.;

  //-----------From (non-converged) fitter 1
//  tLambda = 0.5319;
//  tR1 = 10.81;
//  tReF0 = 2.11242;
//  tImF0 = 13.8741;
//  tD0 = 0.;

  //-----------From (non-converged) fitter 2
//  tLambda = 0.4985;  //+- 0.4
//  tR1 = 11.24;  //+- 5
//  tReF0 = 1.65627; //+- 1
//  tImF0 = 8.55026; //+- 1
//  tD0 = 0.;

  //-----------From (converged simplex) fitter
  tLambda = 0.1923;
  tR1 = 5.031;
  tReF0 = -1.694;
  tImF0 = 1.123;
  tD0 = 3.195;

  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();
  gStyle->SetOptStat(0);

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSample("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
  tSampleHist1->SetDirectory(0);
cout << "tSampleHist1: " << tSampleHist1->GetNbinsX() << endl;
  tSampleHist1->SetTitle("#bar{#Xi}K+");
  tSampleHist1->SetName("#bar{#Xi}K+");
  tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
  tSampleHist1->GetYaxis()->SetTitle("C(k*)");
  tSampleHist1->GetYaxis()->SetRangeUser(0.46,1.04);
  tSampleHist1->Draw();

  tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
  tPairAn0010->GetKStarCf()->Draw("same");

//  TString tFileLocationBaseLamKchP = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";
//  FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(tFileLocationBaseLamKchP,kLamKchP,k0010);
//  tLamKchP0010->GetKStarCf()->SetMarkerStyle(20);
//  tLamKchP0010->GetKStarCf()->Draw("same");

//  tCan->SaveAs("FirstFits20160516.eps");
*/

/*
  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

  //--Values averaged from Furnstahl reference
//  double tLambda = 0.72;
//  double tR1 = 5.25;
//  double tReF0s = 1.02;
//  double tImF0s = 0.14;
//  double tD0s = 0.;
//  double tReF0t = 0.11;
//  double tImF0t = 0.17;
//  double tD0t = 0.;
//  double tNorm = 1.;


  //--From non-converged simplex (chi2 = 381.625, 2009 calls)
//  double tLambda = 0.52;
//  double tR1 = 5.05;
//  double tReF0s = -1.241;
//  double tImF0s = 0.99;
//  double tD0s = 8.16;
//  double tReF0t = 0.1136;
//  double tImF0t = 0.1725;
//  double tD0t = 0.00443;
//  double tNorm = 1.;

  //--From non-converged migrad (chi2 = 393.867, 326 calls)
//  double tLambda = 0.74;
//  double tR1 = 5.24;
//  double tReF0s = 1.036;
//  double tImF0s = 0.507;
//  double tD0s = 0.00483;
//  double tReF0t = 0.1136;
//  double tImF0t = 0.1753;
//  double tD0t = 0.000019;
//  double tNorm = 1.;

  //--Random (chi2 = 393.601, call #905)
  double tLambda = 0.818916;
  double tR1 = 7.77189;
  double tReF0s = 2.96129;
  double tImF0s = 1.4245;
  double tD0s = -6.73696;
  double tReF0t = 0.113408;
  double tImF0t = 0.17163;
  double tD0t = 0.00150092;
  double tNorm = 1.;

  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();
  gStyle->SetOptStat(0);

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
  tSampleHist1->SetDirectory(0);
  tSampleHist1->SetTitle("#bar{#Xi}K+");
  tSampleHist1->SetName("#bar{#Xi}K+");
  tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
  tSampleHist1->GetYaxis()->SetTitle("C(k*)");

  if(tAnType == kAXiKchP) tSampleHist1->GetYaxis()->SetRangeUser(0.46,1.04);
  else tSampleHist1->GetYaxis()->SetRangeUser(0.95,3.5);

  tSampleHist1->Draw();

  tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
  tPairAn0010->GetKStarCf()->Draw("same");
*/

//_______________________________________________________________________________________________________________________
  if(bDrawFit)
  {
    int tNbinsK = 15;
    double tKMin = 0.;
    double tKMax = 0.15;

    double tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm;

    if(tAnType==kXiKchP || tAnType==kAXiKchM)
    {
      //18 April 2017
      tLambda = 0.398942;
      tRadius = 2.33678;

      tReF0s = -1.13176;
      tImF0s = 0.988682;
      tD0s = -5.;

      tReF0t = -0.17914;
      tImF0t = 0.0205195;
      tD0t = 2.48178;
      tNorm = 1.;
/*
      tLambda = 0.555232;
      tRadius = 4.60717;

      tReF0s = -0.00976133;
      tImF0s = 0.0409787;
      tD0s = -0.33091;

      tReF0t = -0.484049;
      tImF0t = 0.523492;
      tD0t = 1.53176;
      tNorm = 1.;
*/
/*
      tLambda = 0.712651;
      tRadius = 8.46941;

      tReF0s = -1.09173;
      tImF0s = 4.42899;
      tD0s = 0.81589;

      tReF0t = 0.;
      tImF0t = 0.;
      tD0t = 0.;
      tNorm = 1.;
*/
    }

    if(tAnType==kAXiKchP || tAnType==kXiKchM)
    {
      //18 April 2017
      tLambda = 0.691397;
      tRadius = 4.11372;

      tReF0s = -0.437582;
      tImF0s = -1.03858;
      tD0s = -2.19830;

      tReF0t = 0.126213;
      tImF0t = 0.743508;
      tD0t = -3.72636;
      tNorm = 1.;
/*
      tLambda = 0.973083;
      tRadius = 6.97767;

      tReF0s = -1.94078;
      tImF0s = -1.21309;
      tD0s = 0.160156;

      tReF0t = 1.38324;
      tImF0t = 2.02133;
      tD0t = 4.07520;
      tNorm = 1.;
*/
/*
      tLambda = 0.720526;
      tRadius = 6.0;

      tReF0s = -1.5;
      tImF0s = -1.5;
      tD0s = 0.;

      tReF0t = 0.5;
      tImF0t = 1.5;
      tD0t = 2.;
      tNorm = 1.;
*/
    }




    TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tName = cAnalysisRootTags[tAnType];
    TString tNameConj = cAnalysisRootTags[tConjType];
    TString tSaveName = cAnalysisBaseTags[tAnType] + TString(".eps"); 

    TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
      tSampleHist1->SetDirectory(0);
      tSampleHist1->SetTitle(tName);
      tSampleHist1->SetName(tName);
      tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
      tSampleHist1->GetYaxis()->SetTitle("C(k*)");
      tSampleHist1->SetMarkerStyle(22);
      tSampleHist1->SetMarkerColor(1);
      tSampleHist1->SetLineColor(1);
      tSampleHist1->SetLineStyle(1);
/*
    TH1* tSampleHist2 = tFitter->CreateFitHistogramSampleComplete("SampleHist2", tConjType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
      tSampleHist1->SetDirectory(0);
      tSampleHist1->SetTitle(tNameConj);
      tSampleHist1->SetName(tNameConj);
      tSampleHist1->SetMarkerStyle(29);
      tSampleHist1->SetMarkerColor(4);
      tSampleHist1->SetLineColor(4);
*/
    TH1* tCoulombOnlyHist = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHist", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
      tCoulombOnlyHist->SetDirectory(0);
      tCoulombOnlyHist->SetMarkerStyle(21);
      tCoulombOnlyHist->SetMarkerColor(1);
      tCoulombOnlyHist->SetLineColor(1);
      tCoulombOnlyHist->SetLineStyle(7);


    if(tAnType == kAXiKchP) tSampleHist1->GetYaxis()->SetRangeUser(0.38,1.04);
    else tSampleHist1->GetYaxis()->SetRangeUser(0.95,1.7);

    tSampleHist1->Draw("l");
//    tSampleHist2->Draw("psame");
    tCoulombOnlyHist->Draw("lsame");

    tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
    if(tAnType == kAXiKchP)
    {
      tPairAn0010->GetKStarCf()->SetMarkerColor(4);
      tPairAn0010->GetKStarCf()->SetLineColor(4);
    }
    else
    {
      tPairAn0010->GetKStarCf()->SetMarkerColor(2);
      tPairAn0010->GetKStarCf()->SetLineColor(2);
    }
    tPairAn0010->GetKStarCf()->Draw("psame");

    TLegend *tLeg = new TLegend(0.55,0.50,0.85,0.75);
      tLeg->AddEntry(tPairAn0010->GetKStarCf(),tName,"p");
//      tLeg->AddEntry(tPairConjAn0010->GetKStarCf(),tNameConj,"p");
      tLeg->AddEntry(tSampleHist1,"Full Fit","l");
      tLeg->AddEntry(tCoulombOnlyHist, "Coulomb Only", "l");
      tLeg->Draw();

    TLine *line = new TLine(0,1,0.15,1);
    line->SetLineColor(14);
    line->Draw();

    tCan->SaveAs(tSaveName);
  }

//_______________________________________________________________________________________________________________________
  if(bDrawLam)
  {
    tFitter->SetTurnOffCoulomb(true);

    int tNbinsK = 30;
    double tKMin = 0.;
    double tKMax = 0.30;

    TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();
    gStyle->SetOptStat(0);

//    AnalysisType tLamType = kLamKchP;
    AnalysisType tLamType = kALamKchP;
    TString tName = cAnalysisRootTags[tLamType];
    TString tSaveName = cAnalysisBaseTags[tLamType] + TString(".eps");

    double tLambda, tR1, tReF0, tImF0, tD0, tNorm;
    tNorm = 1.;

    if(tLamType == kLamKchP)
    {
      tLambda = 0.1923;
      tR1 = 5.031;
      tReF0 = -1.694;
      tImF0 = 1.123;
      tD0 = 3.195;
    }

    else
    {
      tLambda = 0.312;
      tR1 = 3.895;
      tReF0 = 0.1146;
      tImF0 = 0.4182;
      tD0 = 7.277;
    }

    TH1* tSampleHist1 = tFitter->CreateFitHistogramSample("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
    tSampleHist1->SetDirectory(0);
      tSampleHist1->SetTitle(tName);
      tSampleHist1->SetName(tName);
      tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
      tSampleHist1->GetYaxis()->SetTitle("C(k*)");
      if(tLamType == kLamKchP) tSampleHist1->GetYaxis()->SetRangeUser(0.88,1.02);
      if(tLamType == kALamKchP) tSampleHist1->GetYaxis()->SetRangeUser(0.88,1.02);
      tSampleHist1->SetMarkerStyle(22);
      if(tLamType == kLamKchP)
      {
        tSampleHist1->SetMarkerColor(2);
        tSampleHist1->SetLineColor(2);
      }
      if(tLamType == kALamKchP)
      {
        tSampleHist1->SetMarkerColor(4);
        tSampleHist1->SetLineColor(4);
      }
      tSampleHist1->Draw("p");

      TString tFileLocationBaseLamKchP = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";
      FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(tFileLocationBaseLamKchP,tLamType,k0010);
      tLamKchP0010->GetKStarCf()->SetMarkerStyle(20);
      tLamKchP0010->GetKStarCf()->Draw("same");

    tCan->SaveAs(tSaveName);

  }

/*
  TH1* tSampleHist2 = tFitter->CreateFitHistogramSample("SampleHist2", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR2, tReF0, tImF0, tD0, tNorm);
  tSampleHist2->SetLineColor(2);
  tSampleHist2->Draw("same");

  TH1* tSampleHist3 = tFitter->CreateFitHistogramSample("SampleHist2", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR3, tReF0, tImF0, tD0, tNorm);
  tSampleHist3->SetLineColor(3);
  tSampleHist3->Draw("same");
*/


/*
  double tRVec[3] = {4.,5.,6.};
  double tReF0Vec[3] = {-1.,1.,9.};
  double tImF0Vec[3] = {0.1,1.,9.};

  TCanvas* aSplitCan = new TCanvas("aSplitCan","aSplitCan");
  aSplitCan->Divide(9,3);

  TH1* tSampleHist;
  TString tSampleHistName = "SampleHist";

  int aSplitCanCounter = 1;

  for(int iR=0; iR<3; iR++)
  {
    for(int iReF0=0; iReF0<3; iReF0++)
    {
      for(int iImF0=0; iImF0<3; iImF0++)
      {
        TString tName = tSampleHistName;
        tName+=iR;
        tName+=iReF0;
        tName+=iImF0;
        tSampleHist = tFitter->CreateFitHistogramSample(tName, kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tRVec[iR], tReF0Vec[iReF0], tImF0Vec[iImF0], tD0, tNorm);
        aSplitCan->cd(aSplitCanCounter);
        tSampleHist->SetLineColor(2);
        tSampleHist->DrawCopy();
        tPairAn0010->GetKStarCf()->Draw("same");
        aSplitCanCounter++;
        aSplitCan->Update();
      }
    }
  }
*/


//-----------------Try to fit a sample CF------------------------
/*
  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

  double tLambda = 0.72;
  double tR1 = 5.25;

  double tReF0 = 1.46;
  double tImF0 = 0.24;
  double tD0 = 0.;
  double tNorm = 1.;

  TH1* tFakeCf = tFitter->CreateFitHistogramSample("FakeCf", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
  tFitter->DoFit();
*/


//---------------Calculate chi2 by hand-------------------------
/*
  int tNbinsK = 62;
  double tKMin = 0.;
  double tKMax = 0.31;

  double tLambda = 0.5;

  double tR1 = 5.0;

  double tReF0 = 1.46;
  double tImF0 = 0.24;
  double tD0 = 0.;
  double tNorm = 0.113436;

  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSample("SampleHist1", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
  tSampleHist1->Draw();

  FitPartialAnalysis* tPartAn0010_0= tPairAn0010->GetFitPartialAnalysis(0);
  TH1* tNum = tPartAn0010_0->GetNumKStarCf();
  TH1* tDen = tPartAn0010_0->GetDenKStarCf();
  TH1* tCf = tPartAn0010_0->GetKStarCf();
  tCf->SetLineColor(2);
  tCf->SetMarkerColor(2);
  tCf->Draw("same");

cout << "tNum->GetBinWidth(1) = " << tNum->GetBinWidth(1) << endl;

  double tChi2 = 0.;

  for(int i=1; i<=tNum->FindBin(0.3); i++)
  {
    double tKStar = tNum->GetBinCenter(i);

    double tNumContent = tNum->GetBinContent(i);
    double tDenContent = tDen->GetBinContent(i);

    double tCfContent = tSampleHist1->GetBinContent(tSampleHist1->FindBin(tKStar));

cout << "i = " << i << endl;
cout << "tNumContent = " << tNumContent << endl;
cout << "tDenContent = " << tDenContent << endl;
cout << "tCfContent = " << tCfContent << endl;

    if(tNumContent==0 || tDenContent==0 || tCfContent==0) cout << "UH OH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    else
    {
      double tTerm1 = tNumContent*log(  (tCfContent*(tNumContent+tDenContent)) / (tNumContent*(tCfContent+1))  );
      double tTerm2 = tDenContent*log(  (tNumContent+tDenContent) / (tDenContent*(tCfContent+1))  );
      double tmp = -2.0*(tTerm1+tTerm2);

cout << "tTerm1 = " << tTerm1 << endl;
cout << "tTerm2 = " << tTerm2 << endl;
cout << "tmp = " << tmp << endl;

      tChi2+=tmp;

cout << "tChi2 = " << tChi2 << endl << endl;

    }
  }
cout << "tChi2(FINAL) = " << tChi2 << endl;
*/

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
