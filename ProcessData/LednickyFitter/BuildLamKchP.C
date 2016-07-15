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
//-----------------------------------------------------------------------------
  vector<ParameterType> ShareAllButNorm(5);
    ShareAllButNorm[0] = kLambda;
    ShareAllButNorm[1] = kRadius;
    ShareAllButNorm[2] = kRef0;
    ShareAllButNorm[3] = kImf0;
    ShareAllButNorm[4] = kd0;

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

  TString FileLocationBase = "/home/jesse/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";

  bool bRunLamKchP = false;
  bool bRunALamKchM = false;
  bool bRunLamKchPwConj = true;

  bool bRunLamKchMwConj = false;

  bool bDoFit = true;
  bool bDrawCfsOnly = false;

  if(bRunLamKchP)
  {
    //-----Find good normalization values LamKchP
    FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(FileLocationBase,kLamKchP,k0010);
    FitPairAnalysis* tLamKchP1030 = new FitPairAnalysis(FileLocationBase,kLamKchP,k1030);
    FitPairAnalysis* tLamKchP3050 = new FitPairAnalysis(FileLocationBase,kLamKchP,k3050);
    
    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchP;
    tVecOfPairAnalysisLamKchP.push_back(tLamKchP0010);
    tVecOfPairAnalysisLamKchP.push_back(tLamKchP1030);
    tVecOfPairAnalysisLamKchP.push_back(tLamKchP3050);
    
    FitSharedAnalyses *tFitSharedAnalysesLamKchP = new FitSharedAnalyses(tVecOfPairAnalysisLamKchP);
    
    if(bDoFit)
    {
      tFitSharedAnalysesLamKchP->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tFitSharedAnalysesLamKchP->SetSharedParameter(kRef0,-1.7);
      tFitSharedAnalysesLamKchP->SetSharedParameter(kImf0,1.1);
      tFitSharedAnalysesLamKchP->SetSharedParameter(kd0,3.);
      
      tFitSharedAnalysesLamKchP->CreateMinuitParameters();
      
      LednickyFitter* tLamKchPFitter = new LednickyFitter(tFitSharedAnalysesLamKchP);
      tLamKchPFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tLamKchPFitter;
      
      tLamKchPFitter->DoFit();
      
      vector<double> tNormFitValuesLamKchP;
    
      for(int iPairAn=0; iPairAn<3; iPairAn++)
	{
	  for(int iPartAn=0; iPartAn<5; iPartAn++)
	    {
	      tNormFitValuesLamKchP.push_back(tFitSharedAnalysesLamKchP->GetFitPairAnalysis(iPairAn)->GetFitPartialAnalysis(iPartAn)->GetFitParameter(kNorm)->GetFitValue());
	    }
	}
      
      for(unsigned int i=0; i<tNormFitValuesLamKchP.size(); i++) {cout << tNormFitValuesLamKchP[i] << endl;}
      
      TCanvas* canLamKchP = new TCanvas("canLamKchP","canLamKchP");
        canLamKchP->Divide(1,3);
      canLamKchP->cd(1);
        tFitSharedAnalysesLamKchP->DrawFit(0,"LamKchP0010");
      canLamKchP->cd(2);
        tFitSharedAnalysesLamKchP->DrawFit(1,"LamKchP1030");
      canLamKchP->cd(3);
        tFitSharedAnalysesLamKchP->DrawFit(2,"LamKchP3050");
    }

    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchP0010 = tFitSharedAnalysesLamKchP->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfLamKchP1030 = tFitSharedAnalysesLamKchP->GetKStarCfHeavy(1)->GetHeavyCf();
      TH1* tCfLamKchP3050 = tFitSharedAnalysesLamKchP->GetKStarCfHeavy(2)->GetHeavyCf();

      TCanvas* canLamKchP = new TCanvas("canLamKchP","canLamKchP");
        canLamKchP->Divide(1,3);
      canLamKchP->cd(1);
        tCfLamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP0010->Draw();
      canLamKchP->cd(2);
        tCfLamKchP1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP1030->Draw();
      canLamKchP->cd(3);
        tCfLamKchP3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP3050->Draw();
    }
    
    //delete tFitSharedAnalysesLamKchP;
/*
    delete tLamKchP0010;
    delete tLamKchP1030;
    delete tLamKchP3050;
*/
  }
  

  if(bRunALamKchM)
  {
    //-----Find good normalization values ALamKchM
    FitPairAnalysis* tALamKchM0010 = new FitPairAnalysis(FileLocationBase,kALamKchM,k0010);
    FitPairAnalysis* tALamKchM1030 = new FitPairAnalysis(FileLocationBase,kALamKchM,k1030);
    FitPairAnalysis* tALamKchM3050 = new FitPairAnalysis(FileLocationBase,kALamKchM,k3050);
    
    vector<FitPairAnalysis*> tVecOfPairAnalysisALamKchM;
    tVecOfPairAnalysisALamKchM.push_back(tALamKchM0010);
    tVecOfPairAnalysisALamKchM.push_back(tALamKchM1030);
    tVecOfPairAnalysisALamKchM.push_back(tALamKchM3050);
    
    FitSharedAnalyses *tFitSharedAnalysesALamKchM = new FitSharedAnalyses(tVecOfPairAnalysisALamKchM);

    if(bDoFit)
    {
      tFitSharedAnalysesALamKchM->SetSharedParameter(kLambda,0.21,0.1,1.0);
      tFitSharedAnalysesALamKchM->SetSharedParameter(kRef0,-1.7);
      tFitSharedAnalysesALamKchM->SetSharedParameter(kImf0,1.1);
      tFitSharedAnalysesALamKchM->SetSharedParameter(kd0,3.);
      
      tFitSharedAnalysesALamKchM->CreateMinuitParameters();
      
      LednickyFitter* tALamKchMFitter = new LednickyFitter(tFitSharedAnalysesALamKchM);
      tALamKchMFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tALamKchMFitter;
      
      tALamKchMFitter->DoFit();
      
      vector<double> tNormFitValuesALamKchM;
      
      for(int iPairAn=0; iPairAn<3; iPairAn++)
	{
	  for(int iPartAn=0; iPartAn<5; iPartAn++)
	    {
	      tNormFitValuesALamKchM.push_back(tFitSharedAnalysesALamKchM->GetFitPairAnalysis(iPairAn)->GetFitPartialAnalysis(iPartAn)->GetFitParameter(kNorm)->GetFitValue());
	    }
	}
      
      for(unsigned int i=0; i<tNormFitValuesALamKchM.size(); i++) {cout << tNormFitValuesALamKchM[i] << endl;}
      

      TCanvas* canALamKchM = new TCanvas("canALamKchM","canALamKchM");
        canALamKchM->Divide(1,3);
      canALamKchM->cd(1);
        tFitSharedAnalysesALamKchM->DrawFit(0,"ALamKchM0010");
      canALamKchM->cd(2);
        tFitSharedAnalysesALamKchM->DrawFit(1,"ALamKchM1030");
      canALamKchM->cd(3);
        tFitSharedAnalysesALamKchM->DrawFit(2,"ALamKchM3050");
    }


    if(bDrawCfsOnly)
    {
      TH1* tCfALamKchM0010 = tFitSharedAnalysesALamKchM->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfALamKchM1030 = tFitSharedAnalysesALamKchM->GetKStarCfHeavy(1)->GetHeavyCf();
      TH1* tCfALamKchM3050 = tFitSharedAnalysesALamKchM->GetKStarCfHeavy(2)->GetHeavyCf();

      TCanvas* canALamKchM = new TCanvas("canALamKchM","canALamKchM");
        canALamKchM->Divide(1,3);
      canALamKchM->cd(1);
        tCfALamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM0010->Draw();
      canALamKchM->cd(2);
        tCfALamKchM1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM1030->Draw();
      canALamKchM->cd(3);
        tCfALamKchM3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM3050->Draw();
    }
    
    //delete tFitSharedAnalysesALamKchM;
/*
    delete tALamKchM0010;
    delete tALamKchM1030;
    delete tALamKchM3050;
*/
  }
  

  if(bRunLamKchPwConj)
  {
    FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(FileLocationBase,kLamKchP,k0010);
    FitPairAnalysis* tLamKchP1030 = new FitPairAnalysis(FileLocationBase,kLamKchP,k1030);
    FitPairAnalysis* tLamKchP3050 = new FitPairAnalysis(FileLocationBase,kLamKchP,k3050);

    FitPairAnalysis* tALamKchM0010 = new FitPairAnalysis(FileLocationBase,kALamKchM,k0010);
    FitPairAnalysis* tALamKchM1030 = new FitPairAnalysis(FileLocationBase,kALamKchM,k1030);
    FitPairAnalysis* tALamKchM3050 = new FitPairAnalysis(FileLocationBase,kALamKchM,k3050);

    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchPwConj;
      tVecOfPairAnalysisLamKchPwConj.push_back(tLamKchP0010);
      tVecOfPairAnalysisLamKchPwConj.push_back(tALamKchM0010);
      tVecOfPairAnalysisLamKchPwConj.push_back(tLamKchP1030);
      tVecOfPairAnalysisLamKchPwConj.push_back(tALamKchM1030);
      tVecOfPairAnalysisLamKchPwConj.push_back(tLamKchP3050);
      tVecOfPairAnalysisLamKchPwConj.push_back(tALamKchM3050);

    FitSharedAnalyses *tFitSharedAnalysesLamKchPwConj = new FitSharedAnalyses(tVecOfPairAnalysisLamKchPwConj);

    if(bDoFit)
    {
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRef0,-1.7);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kImf0,1.1);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kd0,3.);

      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRadius,Share01);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRadius,Share23);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRadius,Share45);
      
      tFitSharedAnalysesLamKchPwConj->CreateMinuitParameters();
      
      LednickyFitter* tLamKchPwConjFitter = new LednickyFitter(tFitSharedAnalysesLamKchPwConj);
      tLamKchPwConjFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tLamKchPwConjFitter;
      
      tLamKchPwConjFitter->DoFit();
      
      TCanvas* canLamKchPwConj = new TCanvas("canLamKchPwConj","canLamKchPwConj");
        canLamKchPwConj->Divide(2,3);
      canLamKchPwConj->cd(1);
        tFitSharedAnalysesLamKchPwConj->DrawFit(0,"LamKchP0010");
      canLamKchPwConj->cd(2);
        tFitSharedAnalysesLamKchPwConj->DrawFit(1,"ALamKchM0010");
      canLamKchPwConj->cd(3);
        tFitSharedAnalysesLamKchPwConj->DrawFit(2,"LamKchP1030");
      canLamKchPwConj->cd(4);
        tFitSharedAnalysesLamKchPwConj->DrawFit(3,"ALamKchM1030");
      canLamKchPwConj->cd(5);
        tFitSharedAnalysesLamKchPwConj->DrawFit(4,"LamKchP3050");
      canLamKchPwConj->cd(6);
        tFitSharedAnalysesLamKchPwConj->DrawFit(5,"ALamKchM3050");
    }


    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchP0010 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfALamKchM0010 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(1)->GetHeavyCf();

      TH1* tCfLamKchP1030 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(2)->GetHeavyCf();
      TH1* tCfALamKchM1030 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(3)->GetHeavyCf();

      TH1* tCfLamKchP3050 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(4)->GetHeavyCf();
      TH1* tCfALamKchM3050 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(5)->GetHeavyCf();

      TCanvas* canLamKchPwConj = new TCanvas("canLamKchPwConj","canLamKchPwConj");
        canLamKchPwConj->Divide(2,3);
      canLamKchPwConj->cd(1);
        tCfLamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP0010->Draw();
      canLamKchPwConj->cd(2);
        tCfALamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM0010->Draw();
      canLamKchPwConj->cd(3);
        tCfLamKchP1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP1030->Draw();
      canLamKchPwConj->cd(4);
        tCfALamKchM1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM1030->Draw();
      canLamKchPwConj->cd(5);
        tCfLamKchP3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP3050->Draw();
      canLamKchPwConj->cd(6);
        tCfALamKchM3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM3050->Draw();
    }

  }


  if(bRunLamKchMwConj)
  {
    FitPairAnalysis* tLamKchM0010 = new FitPairAnalysis(FileLocationBase,kLamKchM,k0010);
    FitPairAnalysis* tLamKchM1030 = new FitPairAnalysis(FileLocationBase,kLamKchM,k1030);
    FitPairAnalysis* tLamKchM3050 = new FitPairAnalysis(FileLocationBase,kLamKchM,k3050);

    FitPairAnalysis* tALamKchP0010 = new FitPairAnalysis(FileLocationBase,kALamKchP,k0010);
    FitPairAnalysis* tALamKchP1030 = new FitPairAnalysis(FileLocationBase,kALamKchP,k1030);
    FitPairAnalysis* tALamKchP3050 = new FitPairAnalysis(FileLocationBase,kALamKchP,k3050);

    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchMwConj;
      tVecOfPairAnalysisLamKchMwConj.push_back(tLamKchM0010);
      tVecOfPairAnalysisLamKchMwConj.push_back(tALamKchP0010);
      tVecOfPairAnalysisLamKchMwConj.push_back(tLamKchM1030);
      tVecOfPairAnalysisLamKchMwConj.push_back(tALamKchP1030);
      tVecOfPairAnalysisLamKchMwConj.push_back(tLamKchM3050);
      tVecOfPairAnalysisLamKchMwConj.push_back(tALamKchP3050);

    FitSharedAnalyses *tFitSharedAnalysesLamKchMwConj = new FitSharedAnalyses(tVecOfPairAnalysisLamKchMwConj);

    if(bDoFit)
    {
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kLambda,0.30,0.1,1.0);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRef0,0.10);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kImf0,0.40);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kd0,7.);

      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRadius,Share01);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRadius,Share23);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRadius,Share45);
      
      tFitSharedAnalysesLamKchMwConj->CreateMinuitParameters();
      
      LednickyFitter* tLamKchMwConjFitter = new LednickyFitter(tFitSharedAnalysesLamKchMwConj);
      tLamKchMwConjFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      myFitter = tLamKchMwConjFitter;
      
      tLamKchMwConjFitter->DoFit();
      
      TCanvas* canLamKchMwConj = new TCanvas("canLamKchMwConj","canLamKchMwConj");
        canLamKchMwConj->Divide(2,3);
      canLamKchMwConj->cd(1);
        tFitSharedAnalysesLamKchMwConj->DrawFit(0,"LamKchM0010");
      canLamKchMwConj->cd(2);
        tFitSharedAnalysesLamKchMwConj->DrawFit(1,"ALamKchP0010");
      canLamKchMwConj->cd(3);
        tFitSharedAnalysesLamKchMwConj->DrawFit(2,"LamKchM1030");
      canLamKchMwConj->cd(4);
        tFitSharedAnalysesLamKchMwConj->DrawFit(3,"ALamKchP1030");
      canLamKchMwConj->cd(5);
        tFitSharedAnalysesLamKchMwConj->DrawFit(4,"LamKchM3050");
      canLamKchMwConj->cd(6);
        tFitSharedAnalysesLamKchMwConj->DrawFit(5,"ALamKchP3050");
    }


    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchM0010 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfALamKchP0010 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(1)->GetHeavyCf();

      TH1* tCfLamKchM1030 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(2)->GetHeavyCf();
      TH1* tCfALamKchP1030 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(3)->GetHeavyCf();

      TH1* tCfLamKchM3050 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(4)->GetHeavyCf();
      TH1* tCfALamKchP3050 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(5)->GetHeavyCf();

      TCanvas* canLamKchMwConj = new TCanvas("canLamKchMwConj","canLamKchMwConj");
        canLamKchMwConj->Divide(2,3);
      canLamKchMwConj->cd(1);
        tCfLamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchM0010->Draw();
      canLamKchMwConj->cd(2);
        tCfALamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchP0010->Draw();
      canLamKchMwConj->cd(3);
        tCfLamKchM1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchM1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchM1030->Draw();
      canLamKchMwConj->cd(4);
        tCfALamKchP1030->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchP1030->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchP1030->Draw();
      canLamKchMwConj->cd(5);
        tCfLamKchM3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchM3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchM3050->Draw();
      canLamKchMwConj->cd(6);
        tCfALamKchP3050->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchP3050->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchP3050->Draw();
    }

  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
