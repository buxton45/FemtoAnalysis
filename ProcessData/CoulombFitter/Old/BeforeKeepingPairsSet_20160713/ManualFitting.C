#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"

CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
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

  TString tFileLocationBase = "~/Analysis/K0Lam/Results_cXicKch_20160202/Results_cXicKch_20160202";

//  AnalysisType tAnType = kAXiKchP;
//  AnalysisType tConjType = kXiKchM;

  AnalysisType tAnType = kXiKchP;
  AnalysisType tConjType = kAXiKchM;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tAnType,k0010);
  FitPairAnalysis* tPairConjAn0010 = new FitPairAnalysis(tFileLocationBase,tConjType,k0010);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairConjAn0010);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    tSharedAn->SetSharedParameter(kLambda,0.72);
    tSharedAn->SetSharedParameter(kRadius,5.25);
    tSharedAn->SetSharedParameter(kRef0,1.02);
    tSharedAn->SetSharedParameter(kImf0,0.14);
    tSharedAn->SetSharedParameter(kd0,0.);
    tSharedAn->SetSharedParameter(kRef02,0.11);
    tSharedAn->SetSharedParameter(kImf02,0.17);
    tSharedAn->SetSharedParameter(kd02,0.);
  }

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    tSharedAn->SetSharedParameter(kLambda,0.5,0.3,1.);
    tSharedAn->SetSharedParameter(kRadius,5.,3.,10.);
    tSharedAn->SetSharedParameter(kRef0,0.);
    tSharedAn->SetSharedParameter(kImf0,0.);
    tSharedAn->SetSharedParameter(kd0,0.);
    tSharedAn->SetSharedParameter(kRef02,0.);
    tSharedAn->SetSharedParameter(kImf02,0.0);
    tSharedAn->SetSharedParameter(kd02,0.);
  }

  tSharedAn->RebinAnalyses(2);

//  tSharedAn->SetFitType(kChi2);

  tSharedAn->CreateMinuitParameters();

//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.15);
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.30);
//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.02);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------
  TString tPairKStarNtupleDirName = "~/Analysis/MathematicaNumericalIntegration/NTuples/Roman";
  TString tFileBaseName = "Results_cXicKch_20160610";
  TString tOutputName = "PairKStar3dVec_20160610_";
  int tNFiles = 27;

//  tFitter->WriteAllPairKStar3dVecFiles(tOutputName,tPairKStarNtupleDirName,tFileBaseName,tNFiles,32,0.,0.32);
  tFitter->BuildPairKStar4dVecFromTxt(tOutputName);
  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

//----- XiKchP and Conjugate
//Winner 25 June:  Lam=0.5, R=6.0, ReF0s=-0.5, ImF0s=0.0, D0s=-2., ReF0t=-0.5, ImF0t=0.5, D0t=2.
//Winner 30 June:  Lam=0.2, R=3.0, ReF0s= 1.5, ImF0s=1.5, D0s= 2., ReF0t= 0.0, ImF0t=1.5, D0t=2.

//----- AXiKchP and Conjugate
//Winner 25 June:  Lam=0.8, R=8.0, ReF0s=-1.5, ImF0s=-1.5, D0s=0., ReF0t=0.5, ImF0t=1.5, D0t=2.


  //-------------------------------------------
  double tGlobalChi2 = 1000000000;
  double tChi2;

  int tNFitParPerAnalysis = 8;
  double *tParamsForHistograms = new double[tNFitParPerAnalysis];

  int tNbinsK = 30;
  double tKMin = 0.;
  double tKMax = 0.30;
  double tNorm = 1.;

/*
  int tNLambda = 1;
  double tLambdaVec[tNLambda] = {0.2};

  int tNR = 1;
  double tRVec[tNR] = {3.};

  int tNReF0s = 2;
  int tNImF0s = 2;
  int tND0s = 3;
  double tReF0sVec[tNReF0s] = {-1.5,-0.5};
  double tImF0sVec[tNImF0s] = {-1.5,-0.5};
  double tD0sVec[tND0s] = {-2.0,0.0,2.0};

  int tNReF0t = 2;
  int tNImF0t = 2;
  int tND0t = 3;
  double tReF0tVec[tNReF0t] = {-1.5,-0.5};
  double tImF0tVec[tNImF0t] = {-1.5,-0.5};
  double tD0tVec[tND0t] = {-2.0,0.0,2.0};
*/


  int tNLambda = 3;
  double tLambdaVec[tNLambda] = {0.2,0.5,0.8};

  int tNR = 4;
  double tRVec[tNR] = {3.,5.,6.,8.};

  int tNReF0s = 5;
  int tNImF0s = 5;
  int tND0s = 3;
  double tReF0sVec[tNReF0s] = {-1.5,-0.5,0.0,0.5,1.5};
  double tImF0sVec[tNImF0s] = {-1.5,-0.5,0.0,0.5,1.5};
  double tD0sVec[tND0s] = {-2.0,0.0,2.0};

  int tNReF0t = 5;
  int tNImF0t = 5;
  int tND0t = 3;
  double tReF0tVec[tNReF0t] = {-1.5,-0.5,0.0,0.5,1.5};
  double tImF0tVec[tNImF0t] = {-1.5,-0.5,0.0,0.5,1.5};
  double tD0tVec[tND0t] = {-2.0,0.0,2.0};

  bool bDoManualFitting = true;
  bool bFindChi2Spread = false;



//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------

  if(bDoManualFitting)
  {
    TString tOutputName = TString("GoodValues") + TString(cAnalysisBaseTags[tAnType]) + TString(".txt");
    ofstream tFileOut(tOutputName);
    double tMaxWriteValue = 375.;

    TH1D* tChi2Distribution = new TH1D("tChi2Distribution","tChi2Distribution",500,0,5000);
    int iLam, iR, iReF0s, iImF0s, iD0s, iReF0t, iImF0t, iD0t;

    for(iLam=0; iLam<tNLambda; iLam++)
    {
      for(iR=0; iR<tNR; iR++)
      {
        for(iReF0s=0; iReF0s<tNReF0s; iReF0s++)
        {
          for(iImF0s=0; iImF0s<tNImF0s; iImF0s++)
          {
            for(iD0s=0; iD0s<tND0s; iD0s++)
            {
              for(iReF0t=0; iReF0t<tNReF0t; iReF0t++)
              {
                for(iImF0t=0; iImF0t<tNImF0t; iImF0t++)
                {
                  for(iD0t=0; iD0t<tND0t; iD0t++)
                  {
                    TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambdaVec[iLam], tRVec[iR], tReF0sVec[iReF0s], tImF0sVec[iImF0s], tD0sVec[iD0s], tReF0tVec[iReF0t], tImF0tVec[iImF0t], tD0tVec[iD0t], tNorm);
                    tSampleHist1->SetDirectory(0);
                    tChi2 = tFitter->GetChi2(tSampleHist1);

                    if(tChi2 < tGlobalChi2) tGlobalChi2 = tChi2;

                    cout << "iLam = " << iLam << " | iR = " << iR << " | iReF0s = " << iReF0s << " | iImF0s = " << iImF0s << " | iD0s = " << iD0s << " | iReF0t = " << iReF0t << " | iImF0t = " << iImF0t << " | iD0t = " << iD0t << endl; 
                    cout << "tChi2 = " << tChi2 << endl;
                    cout << "tGlobalChi2 = " << tGlobalChi2 << endl << endl;

                    tChi2Distribution->Fill(tChi2);

                    tParamsForHistograms[0] = tLambdaVec[iLam];
                    tParamsForHistograms[1] = tRVec[iR];
                    tParamsForHistograms[2] = tReF0sVec[iReF0s];
                    tParamsForHistograms[3] = tImF0sVec[iImF0s];
                    tParamsForHistograms[4] = tD0sVec[iD0s];
                    tParamsForHistograms[5] = tReF0tVec[iReF0t];
                    tParamsForHistograms[6] = tImF0tVec[iImF0t];
                    tParamsForHistograms[7] = tD0tVec[iD0t];

                    tSharedAn->GetFitChi2Histograms()->FillHistograms(tChi2,tParamsForHistograms);

                    if(tChi2 < tMaxWriteValue) tFileOut << tChi2 << " : " << tLambdaVec[iLam] << "," << tRVec[iR] << "," << tReF0sVec[iReF0s] << "," << tImF0sVec[iImF0s] << "," << tD0sVec[iD0s] << "," << tReF0tVec[iReF0t] << "," << tImF0tVec[iImF0t] << "," << tD0tVec[iD0t] << endl; 
                  }
                }
              }
            }
          }
        }
      }
    }
    TString tSaveHistName = "Chi2HistogramsPersonal_" + TString(cAnalysisBaseTags[tAnType]) + TString(".root");
    tSharedAn->GetFitChi2Histograms()->SaveHistograms(tSaveHistName);

    tChi2Distribution->Draw();
  }


//-------------------------------------------------------------------------------
  if(bFindChi2Spread)
  {
    int tNCalls = 1000;
    TH1D* tChi2Distribution = new TH1D("tChi2Distribution","tChi2Distribution",500,0,500);

    int tNbinsK = 15;
    double tKMin = 0.;
    double tKMax = 0.15;

    double tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm;

    if(tAnType==kAXiKchP || tAnType==kXiKchM)
    {

      tLambda = 0.973083;
      tRadius = 6.97767;

      tReF0s = -1.94078;
      tImF0s = -1.21309;
      tD0s = 0.160156;

      tReF0t = 1.38324;
      tImF0t = 2.02133;
      tD0t = 4.07520;
      tNorm = 1.;
    }

    if(tAnType==kXiKchP || tAnType==kAXiKchM)
    {

      tLambda = 0.555232;
      tRadius = 4.60717;

      tReF0s = -0.00976133;
      tImF0s = 0.0409787;
      tD0s = -0.33091;

      tReF0t = -0.484049;
      tImF0t = 0.523492;
      tD0t = 1.53176;
      tNorm = 1.;
    }


    for(int i=0; i<tNCalls; i++)
    {
cout << i+1 << "/" << tNCalls << endl;
      TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
      tSampleHist1->SetDirectory(0);
      tChi2 = tFitter->GetChi2(tSampleHist1);
      if(tChi2 < tGlobalChi2) tGlobalChi2 = tChi2;
      tChi2Distribution->Fill(tChi2);
    }

    tChi2Distribution->Draw();
    cout << "tChi2Distribution->GetMean() = " << tChi2Distribution->GetMean() << endl;
    cout << "tChi2Distribution->GetMeanError() = " << tChi2Distribution->GetMeanError() << endl;
    cout << "tChi2Distribution->GetStdDev() = " << tChi2Distribution->GetStdDev() << endl;
    cout << "tChi2Distribution->GetStdDevError() = " << tChi2Distribution->GetStdDevError() << endl;
  }


//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  delete tFitter;

  delete tSharedAn;
  delete tPairAn0010;

  return 0;
}
