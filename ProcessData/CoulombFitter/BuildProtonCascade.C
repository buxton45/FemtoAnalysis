#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "TGraph.h"

CoulombFitter *myFitter = NULL;

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
  bool bDrawFit = true;


//-----------------------------------------------------------------------------

  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20160202/Results_cXicKch_20160202";

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
    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,6.,0.,20.);
    tSharedAn->SetSharedParameter(kRef0,-1.5);
    tSharedAn->SetSharedParameter(kImf0,-1.5);
    tSharedAn->SetSharedParameter(kd0,0.);
    tSharedAn->SetSharedParameter(kRef02,0.5);
    tSharedAn->SetSharedParameter(kImf02,1.5);
    tSharedAn->SetSharedParameter(kd02,2.);
  }

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {

    tSharedAn->SetSharedParameter(kLambda,0.5,0.1,1.);
    tSharedAn->SetSharedParameter(kRadius,6.,0.,20.);
    tSharedAn->SetSharedParameter(kRef0,-0.5,-5.,5.);
    tSharedAn->SetSharedParameter(kImf0,0.,-5.,5.);
    tSharedAn->SetSharedParameter(kd0,-2.,-5.,5.);
    tSharedAn->SetSharedParameter(kRef02,-0.5,-5.,5.);
    tSharedAn->SetSharedParameter(kImf02,0.5,-5.,5.);
    tSharedAn->SetSharedParameter(kd02,2.,-5.,5.);
  }


  tSharedAn->RebinAnalyses(2);

  tSharedAn->SetFitType(kChi2);

  tSharedAn->CreateMinuitParameters();

  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.10);
    tFitter->SetTurnOffCoulomb(false);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetUseRandomKStarVectors(true);

  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractiveProtonCascade";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;


//_______________________________________________________________________________________________________________________
  if(bDrawFit)
  {
    int tNbinsK = 20;
    double tKMin = 0.;
    double tKMax = 0.05;

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
      tLambda = 1.0;
      tRadius = 3.1;

      tReF0s = 2.88;
      tImF0s = 0.;
      tD0s = 2.92;

      tNorm = 1.;
    }


    TCanvas* tCan = new TCanvas("tCan","tCan");
    tCan->cd();
    gStyle->SetOptStat(0);

    TString tName = TString("p#Xi-");
    TString tSaveName = TString("p#Xi-") + TString(".eps"); 

    TH1* tSampleHist1 = tFitter->CreateFitHistogramSample("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0s, tImF0s, tD0s, tNorm);
      tSampleHist1->SetDirectory(0);
      tSampleHist1->SetTitle(tName);
      tSampleHist1->SetName(tName);
      tSampleHist1->GetXaxis()->SetTitle("k* (GeV/c)");
      tSampleHist1->GetYaxis()->SetTitle("C(k*)");
      tSampleHist1->SetMarkerStyle(22);
      tSampleHist1->SetMarkerColor(1);
      tSampleHist1->SetLineColor(1);

    tSampleHist1->Draw("p");

    //--------------------------------------------------------
    int tNMaria = 18;
    double tX[tNMaria], tY[tNMaria];

    tX[0] = 0.000478011;
    tX[1] = 0.00105163;
    tX[2] = 0.00152964;
    tX[3] = 0.00200765;
    tX[4] = 0.00248566;
    tX[5] = 0.00305927;
    tX[6] = 0.00344168;
    tX[7] = 0.0040153;
    tX[8] = 0.00449331;
    tX[9] = 0.00650096;
    tX[10] = 0.00984704;
    tX[11] = 0.0159656;
    tX[12] = 0.0198853;
    tX[13] = 0.0249522;
    tX[14] = 0.0298279;
    tX[15] = 0.0359465;
    tX[16] = 0.0398662;
    tX[17] = 0.0478011;

    tY[0] = 105.808;
    tY[1] = 52.2727;
    tY[2] = 34.596;
    tY[3] = 26.2626;
    tY[4] = 20.7071;
    tY[5] = 16.9192;
    tY[6] = 14.6465;
    tY[7] = 12.6263;
    tY[8] = 11.3636;
    tY[9] = 7.57576;
    tY[10] = 5.05051;
    tY[11] = 3.0303;
    tY[12] = 2.52525;
    tY[13] = 2.0202;
    tY[14] = 1.76768;
    tY[15] = 1.51515;
    tY[16] = 1.51515;
    tY[17] = 1.26263;

    TGraph *tGraphMaria = new TGraph(tNMaria,tX,tY);
      tGraphMaria->SetMarkerStyle(20);
      tGraphMaria->SetMarkerColor(4);
      tGraphMaria->SetLineColor(4);
    tGraphMaria->Draw("psame");

    //--------------------------------------------------------
    
    TLegend *tLeg = new TLegend(0.65,0.70,0.85,0.85);
      tLeg->AddEntry(tSampleHist1,"My code","p");
      tLeg->AddEntry(tGraphMaria, "Lednicky code (from Maria)", "p");
      tLeg->Draw();

    tCan->SaveAs(tSaveName);
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
