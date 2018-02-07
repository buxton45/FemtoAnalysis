#include "ParallelWaveFunction.h"
#include "CoulombFitterParallel.h"

#include <complex>
#include <math.h>

CoulombFitterParallel *myFitter = NULL;

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

//  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  std::clock_t start = std::clock();

//-----------------------------------------------------------------------------

/*
  const int ARRAY_SIZE = 64;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  //generate the input array on the host
  float h_in[ARRAY_SIZE];
  for(int i=0; i<ARRAY_SIZE; i++)
  {
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];

  Square *tSq = new Square();
  tSq->RunSquare(h_out,h_in,ARRAY_SIZE);
*/
/*
  TString tFileLocationNtupleBase = "~/Analysis/K0Lam/Results_cXicKch_20160414/Results_cXicKch_20160414";

  Interpolate *tInt = new Interpolate();
  InterpolateGPU *tIntGPU = new InterpolateGPU();

  tInt->BuildPairKStar3dVec(tFileLocationNtupleBase,kAXiKchP,k0010,kBp2,16,0.,0.16);
  tInt->MakeOtherArrays("~/Analysis/MathematicaNumericalIntegration/InterpHistsRepulsive");
  vector<vector<double> > myPairs2d = tInt->BuildPairs();
  vector<vector<double> > myGTildeReal = tInt->ReturnGTildeReal();

  //---------------------

  int tNThreadsPerBlock = 1000;
  int tNBlocks = 10;
  int tNbinsK = 160;
  int tNbinsR = 100;
  int tElementsPerPair = 2;


  int sizeOut = tNThreadsPerBlock*tNBlocks*sizeof(double);
  int sizePairs = tNThreadsPerBlock*tNBlocks*tElementsPerPair*sizeof(double);
  int size2dVec = tNbinsK*tNbinsR*sizeof(double);

  double * host_out;
  double* myPairs1d;
  double* myGTildeReal1d;

  checkCudaErrors(cudaHostAlloc((void**) &host_out, sizeOut, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &myPairs1d, sizePairs, cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void**) &myGTildeReal1d, size2dVec, cudaHostAllocMapped));

  for(int i=0; i<tNbinsK; i++)
  {
    for(int j=0; j<tNbinsR; j++)
    {
      myGTildeReal1d[j+tNbinsR*i] = myGTildeReal[i][j];
    }
  }

  for(int i=0; i<tNThreadsPerBlock*tNBlocks; i++)
  {
    for(int j=0; j<tElementsPerPair; j++)
    {
      myPairs1d[j+tElementsPerPair*i] = myPairs2d[i][j];
    }
  }


  //--------------------------
  auto t1 = std::chrono::high_resolution_clock::now();

//  vector<double> myResults = tIntGPU->RunBilinearInterpolate(myPairs2d,myGTildeReal);
  double* myResults = tIntGPU->RunBilinearInterpolate(host_out,myPairs1d,myGTildeReal1d);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto int12 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
  cout << "Total time parallel = " << int12.count() << " microseconds" << endl;

//--------------------------
  auto ta = std::chrono::high_resolution_clock::now();

  vector<double> mySerialResults = tInt->RunBilinearInterpolateSerial(myPairs2d,myGTildeReal);

  auto tb = std::chrono::high_resolution_clock::now();
  auto intab = std::chrono::duration_cast<std::chrono::microseconds>(tb-ta);
  cout << "Total time serial = " << intab.count() << " microseconds" << endl;

//--------------------------
//  for(int i=0; i<10000; i++) cout << "i=" << i << " : myResults[i] = " << myResults[i] << " : " << mySerialResults[i] << " = mySerialResults[i]" << endl << endl << endl;


  for(int i=0; i<10000; i++) 
  {
//    cout << "i=" << i << " : myResults[i] = " << myResults[i] << " : " << mySerialResults[i] << " = mySerialResults[i]" << endl;
    if(abs(myResults[i]-mySerialResults[i]) > 0.0000001) 
    {
      cout << "DISCREPANCY in bin " << i << "!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << "myResults = " << myResults[i] << endl;
      cout << "mySerialResults = " << mySerialResults[i] << endl << endl;
    }
  }
*/


  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170423/Results_cXicKch_20170423";

  AnalysisType tAnType, tConjType;
  tAnType = kXiKchP;
  //tAnType = kXiKchM;

  int tNPairsPerKStarBin = 16384;
  int tNThreadsPerBlock = 512;
  int tNBlocks = 32;

/*
  int tNPairsPerKStarBin = 65536;
  int tNThreadsPerBlock = 512;
  int tNBlocks = 128;
*/
  assert(tNThreadsPerBlock*tNBlocks == tNPairsPerKStarBin);

  if(tAnType==kXiKchP) tConjType = kAXiKchM;
  else if(tAnType==kXiKchM) tConjType = kAXiKchP;
  else assert(0);

  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 5;
  if(tAnalysisRunType==kTrain || tAnalysisRunType==kTrainSys) tNPartialAnalysis = 2;

  bool bIncludeSingletAndTriplet=false;
  bool bRunMB=true;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);

//-----------------------------------------------------------------------------
  vector<FitPairAnalysis*> tVecOfPairAn;

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase, tAnType, k0010, tAnalysisRunType, tNPartialAnalysis, "",bIncludeSingletAndTriplet);
    tVecOfPairAn.push_back(tPairAn0010);
  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase, tConjType, k0010, tAnalysisRunType, tNPartialAnalysis, "", bIncludeSingletAndTriplet);
    tVecOfPairAn.push_back(tPairConjAn0010);

  if(bRunMB)
  {
    FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase, tAnType, k1030, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
      tVecOfPairAn.push_back(tPairAn1030);
    FitPairAnalysis* tPairConjAn1030 = new FitPairAnalysis(tFileLocationBase, tConjType, k1030, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
      tVecOfPairAn.push_back(tPairConjAn1030);
    FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase, tAnType, k3050, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
      tVecOfPairAn.push_back(tPairAn3050);
    FitPairAnalysis* tPairConjAn3050 = new FitPairAnalysis(tFileLocationBase, tConjType, k3050, tAnalysisRunType, tNPartialAnalysis, TString(""), bIncludeSingletAndTriplet);
      tVecOfPairAn.push_back(tPairConjAn3050);
  }

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);



  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    tSharedAn->SetSharedParameter(kLambda,{0,1},0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,{0,1},4.0,1.,6.);
    tSharedAn->SetSharedParameter(kRef0,1.02,-3.,3.);
    tSharedAn->SetSharedParameter(kImf0,0.14,-3.,3.);
    tSharedAn->SetSharedParameter(kd0,0.,-5.,5.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,0.48,-3.,3.);
      tSharedAn->SetSharedParameter(kImf02,0.17,-3.,3.);
      tSharedAn->SetSharedParameter(kd02,0.,-3.,3.);
    }
    if(bRunMB)
    {
      tSharedAn->SetSharedParameter(kLambda,{2,3},0.5,0.1,1.);
      tSharedAn->SetSharedParameter(kLambda,{4,5},0.5,0.1,1.);

      tSharedAn->SetSharedParameter(kRadius,{2,3},3.0,1.,6.);
      tSharedAn->SetSharedParameter(kRadius,{4,5},2.0,1.,6.);
    }
  }

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    tSharedAn->SetSharedParameter(kLambda,{0,1},0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,{0,1},4.0,1.,10.);
    tSharedAn->SetSharedParameter(kRef0,-0.2,0.,0.);
    tSharedAn->SetSharedParameter(kImf0,0.2,0.,0.);
    tSharedAn->SetSharedParameter(kd0,0.,-0,0.);
    if(bIncludeSingletAndTriplet)
    {
      tSharedAn->SetSharedParameter(kRef02,-0.2,-3.,3.);
      tSharedAn->SetSharedParameter(kImf02,0.2,-3.,3.);
      tSharedAn->SetSharedParameter(kd02,0.,-5.,5.);
    }
    if(bRunMB)
    {
      tSharedAn->SetSharedParameter(kLambda,{2,3},0.5,0.1,1.);
      tSharedAn->SetSharedParameter(kLambda,{4,5},0.5,0.1,1.);
      tSharedAn->SetSharedParameter(kRadius,{2,3},3.0,1.,10.);
      tSharedAn->SetSharedParameter(kRadius,{4,5},2.0,1.,10.);
    }
  }





  tSharedAn->RebinAnalyses(2);

//  tSharedAn->SetFitType(kChi2);

  tSharedAn->CreateMinuitParameters();

  CoulombFitterParallel* tFitter = new CoulombFitterParallel(tSharedAn,0.30);
//  CoulombFitterParallel* tFitter = new CoulombFitterParallel(tSharedAn,0.02);
  tFitter->SetIncludeSingletAndTriplet(bIncludeSingletAndTriplet);
  tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------
  TString tPairKStarNtupleBaseName = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/NTuples/Roman/Results_cXicKch_20160610";
  TString tOutputName = "PairKStar3dVec_20160610_";
  int tNFiles = 27;


//  tFitter->BuildPairKStar3dVecFull(tPairKStarNtupleBaseName,tNFiles,kAXiKchP,k0010,16,0.,0.16);
//  tFitter->WritePairKStar3dVecFile(tOutputName,tPairKStarNtupleBaseName,tNFiles,kAXiKchP,k0010,16,0.,0.16);
//  tFitter->WriteAllPairKStar3dVecFiles(tOutputName,tPairKStarNtupleBaseName,tNFiles,16,0.,0.16);

//  tFitter->BuildPairKStar3dVecFromTxt(tOutputName);

//  tFitter->BuildPairKStar4dVecFromTxt(tOutputName);
  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true);
  tFitter->SetNPairsPerKStarBin(tNPairsPerKStarBin);
  tFitter->SetNThreadsPerBlockAndNBlocks(tNThreadsPerBlock, tNBlocks);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  tFitter->DoFit();
//TODO binOffset member of 3dKStarVecInfo?

/*
//-------------------
  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

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
//  tLambda = 0.791284;
//  tR1 = 6.91894;
//  tReF0 = -1.51338;
//  tImF0 = 1.22166;
//  tD0 = -695.747;

  //---Everthing safely within interpolater (more or less)
  //Of course, R can still be outside, but majority will be inside
  tLambda = 0.791284;
  tR1 = 3.0;
  tReF0 = -1.51338;
  tImF0 = 1.22166;
  tD0 = 0.;

  TCanvas* tCan = new TCanvas("tCan","tCan");
  tCan->cd();
  gStyle->SetOptStat(0);

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleParallel("SampleHist1", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0, tImF0, tD0, tNorm);
  tSampleHist1->SetTitle("#bar{#Xi}K+");
  tSampleHist1->SetName("#bar{#Xi}K+");
  tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
  tSampleHist1->GetYaxis()->SetTitle("C(k*)");
  tSampleHist1->GetYaxis()->SetRangeUser(0.46,1.04);
  tSampleHist1->Draw();

  tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
  tPairAn0010->GetKStarCf()->Draw("same");
//  tCan->SaveAs("FirstFits20160516.eps");
*/

/*
//-------------------
  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

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

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleCompleteParallel("SampleHist1", kAXiKchP, tNbinsK, tKMin, tKMax, tLambda, tR1, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
  tSampleHist1->SetTitle("#bar{#Xi}K+");
  tSampleHist1->SetName("#bar{#Xi}K+");
  tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
  tSampleHist1->GetYaxis()->SetTitle("C(k*)");
  tSampleHist1->GetYaxis()->SetRangeUser(0.46,1.04);
  tSampleHist1->Draw();

  tPairAn0010->GetKStarCf()->SetMarkerStyle(20);
  tPairAn0010->GetKStarCf()->Draw("same");
//  tCan->SaveAs("FirstFits20160516.eps");
*/


  delete tFitter;
  delete tSharedAn;
  delete tPairAn0010;
  delete tPairConjAn0010;



//-------------------------------------------------------------------------------

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Finished program in " << duration << " seconds" << endl;


//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
