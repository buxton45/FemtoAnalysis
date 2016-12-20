#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

LednickyFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
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

  ChronoTimer tFullTimer(kSec);
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

  TString FileLocationBase_cLamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_AsRc_20151228_Old/Results_cLamK0_AsRc_20151228_Old";
//  TString FileLocationBase_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";

  TString FileLocationBase_cLamcKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKch_20161007";

  //Choose one
  //tConjPairType is automatically set
  AnalysisType tPairType;
    //tPairType = kLamK0;
    tPairType = kLamKchP;
    //tPairType = kLamKchM;

  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 5;
  if(tAnalysisRunType==kTrain || tAnalysisRunType==kTrainSys) tNPartialAnalysis = 2;

  bool bRunPair = false;
  bool bRunConjPair = false;
  bool bRunPairwConjAll = true;
  bool bRunPairwConj0010 = false;

  bool bDoFit = true;
  bool bDrawCfsOnly = false;


//-----------------------------------------------------------------------------
  AnalysisType tConjPairType;
  if(tPairType==kLamK0) {tConjPairType=kALamK0;}
  else if(tPairType==kLamKchP) {tConjPairType=kALamKchM;}
  else if(tPairType==kLamKchM) {tConjPairType=kALamKchP;}

  TString tFileLocationBase;
  if(tPairType==kLamK0) {tFileLocationBase=FileLocationBase_cLamK0;}
  else if(tPairType==kLamKchP || tPairType==kLamKchM) {tFileLocationBase=FileLocationBase_cLamcKch;}
//-----------------------------------------------------------------------------

  if(bRunPair || bRunConjPair)
  {
    //Don't want both to be true!
    assert(!(bRunPair && bRunConjPair)); 

    AnalysisType tAnType;
    if(bRunPair) {tAnType=tPairType;}
    else {tAnType=tConjPairType;}
   
    TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
    TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);
    TString tAnName1030 = tAnBaseName + TString(cCentralityTags[1]);
    TString tAnName3050 = tAnBaseName + TString(cCentralityTags[2]);

    FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tAnType,k0010,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase,tAnType,k1030,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase,tAnType,k3050,tAnalysisRunType,tNPartialAnalysis);

    vector<FitPairAnalysis*> tVecOfPairAn;
    tVecOfPairAn.push_back(tPairAn0010);
    tVecOfPairAn.push_back(tPairAn1030);
    tVecOfPairAn.push_back(tPairAn3050);

    FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

    if(bDoFit)
    {
      tSharedAn->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tSharedAn->SetSharedParameter(kRef0,-1.7);
      tSharedAn->SetSharedParameter(kImf0,1.1);
      tSharedAn->SetSharedParameter(kd0,3.);

      tSharedAn->CreateMinuitParameters();

      LednickyFitter* tFitter = new LednickyFitter(tSharedAn);
      tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tFitter;

      tFitter->DoFit();

      cout << "Norm values for analyses of type: " << tAnBaseName << endl;
      for(unsigned int iPairAn=0; iPairAn<tVecOfPairAn.size(); iPairAn++)
      {
        for(int iPartAn=0; iPartAn<tSharedAn->GetFitPairAnalysis(iPairAn)->GetNFitPartialAnalysis(); iPartAn++)
        {
          cout << "\tiPairAn = " << iPairAn << "\tiPartAn = " << iPartAn << "\tNorm = " << tSharedAn->GetFitPairAnalysis(iPairAn)->GetFitPartialAnalysis(iPartAn)->GetFitParameter(kNorm)->GetFitValue() << endl;
        }
      }

      TCanvas* canPairAn = new TCanvas("canPairAn","canPairAn");
        canPairAn->Divide(1,3);

      canPairAn->cd(1);
      tSharedAn->DrawFit(0,tAnName0010);

      canPairAn->cd(2);
      tSharedAn->DrawFit(1,tAnName1030);

      canPairAn->cd(3);
      tSharedAn->DrawFit(2,tAnName3050);

      delete tFitter;
    }

    if(bDrawCfsOnly)
    {
      TH1* tCf0010 = tSharedAn->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCf1030 = tSharedAn->GetKStarCfHeavy(1)->GetHeavyCf();
      TH1* tCf3050 = tSharedAn->GetKStarCfHeavy(2)->GetHeavyCf();

      TCanvas* canPairAn = new TCanvas("canPairAn","canPairAn");
        canPairAn->Divide(1,3);

      canPairAn->cd(1);
        tCf0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCf0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCf0010->Draw();
      canPairAn->cd(2);
        tCf1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCf1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCf1030->Draw();
      canPairAn->cd(3);
        tCf3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCf3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCf3050->Draw();
    }

    delete tSharedAn;
    delete tPairAn0010;
    delete tPairAn1030;
    delete tPairAn3050;
    
  }

  if(bRunPairwConjAll)
  {
    TString tPairAnBaseName = TString(cAnalysisBaseTags[tPairType]);
      TString tPairAnName0010 = tPairAnBaseName + TString(cCentralityTags[0]);
      TString tPairAnName1030 = tPairAnBaseName + TString(cCentralityTags[1]);
      TString tPairAnName3050 = tPairAnBaseName + TString(cCentralityTags[2]);

    TString tConjPairAnBaseName = TString(cAnalysisBaseTags[tConjPairType]);
      TString tConjPairAnName0010 = tConjPairAnBaseName + TString(cCentralityTags[0]);
      TString tConjPairAnName1030 = tConjPairAnBaseName + TString(cCentralityTags[1]);
      TString tConjPairAnName3050 = tConjPairAnBaseName + TString(cCentralityTags[2]);


    FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tPairType,k0010,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase,tPairType,k1030,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase,tPairType,k3050,tAnalysisRunType,tNPartialAnalysis);

    FitPairAnalysis* tConjPairAn0010 = new FitPairAnalysis(tFileLocationBase,tConjPairType,k0010,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tConjPairAn1030 = new FitPairAnalysis(tFileLocationBase,tConjPairType,k1030,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tConjPairAn3050 = new FitPairAnalysis(tFileLocationBase,tConjPairType,k3050,tAnalysisRunType,tNPartialAnalysis);

    vector<FitPairAnalysis*> tVecOfPairAn;
      tVecOfPairAn.push_back(tPairAn0010);
      tVecOfPairAn.push_back(tConjPairAn0010);

      tVecOfPairAn.push_back(tPairAn1030);
      tVecOfPairAn.push_back(tConjPairAn1030);

      tVecOfPairAn.push_back(tPairAn3050);
      tVecOfPairAn.push_back(tConjPairAn3050);

    FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);


    if(bDoFit)
    {
      //tSharedAn->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tSharedAn->SetSharedParameter(kRef0,-2.0);
      tSharedAn->SetSharedParameter(kImf0,1.0);
      tSharedAn->SetSharedParameter(kd0,-1.0);

      tSharedAn->SetSharedParameter(kLambda,Share01,0.5,0.1,1.0);
      tSharedAn->SetSharedParameter(kLambda,Share23,0.5,0.1,1.0);
      tSharedAn->SetSharedParameter(kLambda,Share45,0.5,0.1,1.0);

      tSharedAn->SetSharedParameter(kRadius,Share01,5.0);
      tSharedAn->SetSharedParameter(kRadius,Share23,4.0);
      tSharedAn->SetSharedParameter(kRadius,Share45,3.0);
      
      tSharedAn->CreateMinuitParameters();

      LednickyFitter* tFitter = new LednickyFitter(tSharedAn);
      tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tFitter;

      tFitter->DoFit();

      TCanvas* canPairwConj = new TCanvas("canPairwConj","canPairwConj");
        canPairwConj->Divide(2,3);

      canPairwConj->cd(1);
      tSharedAn->DrawFit(0,tPairAnName0010);

      canPairwConj->cd(2);
      tSharedAn->DrawFit(1,tConjPairAnName0010);

      canPairwConj->cd(3);
      tSharedAn->DrawFit(2,tPairAnName1030);

      canPairwConj->cd(4);
      tSharedAn->DrawFit(3,tConjPairAnName1030);

      canPairwConj->cd(5);
      tSharedAn->DrawFit(4,tPairAnName3050);

      canPairwConj->cd(6);
      tSharedAn->DrawFit(5,tConjPairAnName3050);

      //canPairwConj->SaveAs("~/Analysis/Presentations/Group Meetings/20151030/LamK0.eps");
      //canPairwConj->SaveAs("~/Analysis/K0Lam/Results_cLamK0_AsRc_20151228_Old/Fits_cLamK0.eps");

    }


    if(bDrawCfsOnly)
    {
      TH1* tCfPair0010 = tSharedAn->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfConjPair0010 = tSharedAn->GetKStarCfHeavy(1)->GetHeavyCf();

      TH1* tCfPair1030 = tSharedAn->GetKStarCfHeavy(2)->GetHeavyCf();
      TH1* tCfConjPair1030 = tSharedAn->GetKStarCfHeavy(3)->GetHeavyCf();

      TH1* tCfPair3050 = tSharedAn->GetKStarCfHeavy(4)->GetHeavyCf();
      TH1* tCfConjPair3050 = tSharedAn->GetKStarCfHeavy(5)->GetHeavyCf();


      TCanvas* canPairwConj = new TCanvas("canPairwConj","canPairwConj");
        canPairwConj->Divide(2,3);
      canPairwConj->cd(1);
        tCfPair0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfPair0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfPair0010->Draw();
      canPairwConj->cd(2);
        tCfConjPair0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfConjPair0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfConjPair0010->Draw();
      canPairwConj->cd(3);
        tCfPair1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfPair1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfPair1030->Draw();
      canPairwConj->cd(4);
        tCfConjPair1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfConjPair1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfConjPair1030->Draw();
      canPairwConj->cd(5);
        tCfPair3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfPair3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfPair3050->Draw();
      canPairwConj->cd(6);
        tCfConjPair3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfConjPair3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfConjPair3050->Draw();
    }
    delete tSharedAn;
  }


  if(bRunPairwConj0010)
  {
    TString tPairAnBaseName = TString(cAnalysisBaseTags[tPairType]);
      TString tPairAnName0010 = tPairAnBaseName + TString(cCentralityTags[0]);

    TString tConjPairAnBaseName = TString(cAnalysisBaseTags[tConjPairType]);
      TString tConjPairAnName0010 = tConjPairAnBaseName + TString(cCentralityTags[0]);

    FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tPairType,k0010,tAnalysisRunType,tNPartialAnalysis);
    FitPairAnalysis* tConjPairAn0010 = new FitPairAnalysis(tFileLocationBase,tConjPairType,k0010,tAnalysisRunType,tNPartialAnalysis);

    vector<FitPairAnalysis*> tVecOfPairAn;
      tVecOfPairAn.push_back(tPairAn0010);
      tVecOfPairAn.push_back(tConjPairAn0010);

    FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);


    if(bDoFit)
    {
      //tSharedAn->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tSharedAn->SetSharedParameter(kRef0,-2.0);
      tSharedAn->SetSharedParameter(kImf0,1.0);
      tSharedAn->SetSharedParameter(kd0,-1.0);

      tSharedAn->SetSharedParameter(kLambda,Share01,0.5,0.1,1.0);
      tSharedAn->SetSharedParameter(kRadius,Share01,5.0);

      tSharedAn->CreateMinuitParameters();

      LednickyFitter* tFitter = new LednickyFitter(tSharedAn);
      tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tFitter;

      tFitter->DoFit();

      TCanvas* canPairwConj = new TCanvas("canPairwConj","canPairwConj");
        canPairwConj->Divide(2,1);

      canPairwConj->cd(1);
      tSharedAn->DrawFit(0,tPairAnName0010);

      canPairwConj->cd(2);
      tSharedAn->DrawFit(1,tConjPairAnName0010);

      //canPairwConj->SaveAs("~/Analysis/Presentations/Group Meetings/20151030/LamK0.eps");
      //canPairwConj->SaveAs("~/Analysis/K0Lam/Results_cLamK0_AsRc_20151228_Old/Fits_cLamK0.eps");

    }


    if(bDrawCfsOnly)
    {
      TH1* tCfPair0010 = tSharedAn->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfConjPair0010 = tSharedAn->GetKStarCfHeavy(1)->GetHeavyCf();

      TCanvas* canPairwConj = new TCanvas("canPairwConj","canPairwConj");
        canPairwConj->Divide(2,1);
      canPairwConj->cd(1);
        tCfPair0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfPair0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfPair0010->Draw();
      canPairwConj->cd(2);
        tCfConjPair0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfConjPair0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfConjPair0010->Draw();
    }
    delete tSharedAn;
  }

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
