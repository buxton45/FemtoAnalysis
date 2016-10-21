#include "FitSharedAnalyses.h"
#include "LednickyFitter.h"

LednickyFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  //myFitter->CalculateChi2PMLwMomResCorrectionv2(npar,f,par);
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

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
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


  TString FileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  TString FileLocationBaseMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229";
  TString FileLocationTransform = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices_Mix5.root";

  TString FileLocationBaseTrain = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKch_20161007";
  TString FileLocationBaseTrainMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161007/Results_cLamcKchMC_20161007";

  bool bRunLamKchP = false;
  bool bRunALamKchM = true;
  bool bRunLamKchPwConj = false;

  bool bRunLamKchMwConj = false;

  bool bDoFit = true;
  bool bDrawCfsOnly = false;

  bool bSaveFigures = false;
  TString tSaveFiguresLocation = "~/Analysis/Presentations/AliFemto/20160330/";

  bool bApplyMomResCorrection = false;

  if(bRunLamKchP)
  {
    //-----Find good normalization values LamKchP
    FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kLamKchP,k0010);
    tLamKchP0010->LoadTransformMatrices(FileLocationTransform);


    //TODO delete
    vector<TH2D*> tTransformMatrices = tLamKchP0010->GetTransformMatrices();
    cout << "tTransformMatrices.size() = " << tTransformMatrices.size() << endl;
    TCanvas* tTrashCan = new TCanvas("tTrashCan","tTrashCan");
    tTrashCan->Divide(2,2);
    for(unsigned int i=0; i<tTransformMatrices.size(); i++)
    {
      tTrashCan->cd(i+1);
      tTransformMatrices[i]->Draw("lego2");
    }

    //end TODO delete
    
    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchP;
    tVecOfPairAnalysisLamKchP.push_back(tLamKchP0010);
    
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
      tLamKchPFitter->SetApplyMomResCorrection(bApplyMomResCorrection);
      myFitter = tLamKchPFitter;
      
      tLamKchPFitter->DoFit();
      
      vector<double> tNormFitValuesLamKchP;

      for(unsigned int i=0; i<tNormFitValuesLamKchP.size(); i++) {cout << tNormFitValuesLamKchP[i] << endl;}
      
      TCanvas* canLamKchP = new TCanvas("canLamKchP","canLamKchP");
      canLamKchP->cd(1);
        tFitSharedAnalysesLamKchP->DrawFit(0,"LamKchP0010");
    }

    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchP0010 = tFitSharedAnalysesLamKchP->GetKStarCfHeavy(0)->GetHeavyCf();

      TCanvas* canLamKchP = new TCanvas("canLamKchP","canLamKchP");
      canLamKchP->cd(1);
        tCfLamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP0010->Draw();
    }
    
    //delete tFitSharedAnalysesLamKchP;
/*
    delete tLamKchP0010;
*/
  }
  

  if(bRunALamKchM)
  {
    //-----Find good normalization values ALamKchM
    FitPairAnalysis* tALamKchM0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kALamKchM,k0010);
    
    vector<FitPairAnalysis*> tVecOfPairAnalysisALamKchM;
    tVecOfPairAnalysisALamKchM.push_back(tALamKchM0010);
    
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
      tALamKchMFitter->SetApplyMomResCorrection(bApplyMomResCorrection);
      myFitter = tALamKchMFitter;
      
      tALamKchMFitter->DoFit();
/*      
      vector<double> tNormFitValuesALamKchM;
      
      for(int iPairAn=0; iPairAn<3; iPairAn++)
	{
	  for(int iPartAn=0; iPartAn<5; iPartAn++)
	    {
	      tNormFitValuesALamKchM.push_back(tFitSharedAnalysesALamKchM->GetFitPairAnalysis(iPairAn)->GetFitPartialAnalysis(iPartAn)->GetFitNormParameter()->GetFitValue());
	    }
	}
      
      for(unsigned int i=0; i<tNormFitValuesALamKchM.size(); i++) {cout << tNormFitValuesALamKchM[i] << endl;}
*/      

      TCanvas* canALamKchM = new TCanvas("canALamKchM","canALamKchM");
      canALamKchM->cd(1);
        tFitSharedAnalysesALamKchM->DrawFit(0,"ALamKchM0010");
    }


    if(bDrawCfsOnly)
    {
      TH1* tCfALamKchM0010 = tFitSharedAnalysesALamKchM->GetKStarCfHeavy(0)->GetHeavyCf();

      TCanvas* canALamKchM = new TCanvas("canALamKchM","canALamKchM");
      canALamKchM->cd(1);
        tCfALamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM0010->Draw();
    }
    
    //delete tFitSharedAnalysesALamKchM;
/*
    delete tALamKchM0010;
*/
  }
  

  if(bRunLamKchPwConj)
  {
//    FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kLamKchP,k0010);
//    FitPairAnalysis* tALamKchM0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kALamKchM,k0010);

    FitPairAnalysis* tLamKchP0010 = new FitPairAnalysis(FileLocationBaseTrain,FileLocationBaseTrainMC,kLamKchP,k0010,kTrain,2);
    FitPairAnalysis* tALamKchM0010 = new FitPairAnalysis(FileLocationBaseTrain,FileLocationBaseTrainMC,kALamKchM,k0010,kTrain,2);


    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchPwConj;
      tVecOfPairAnalysisLamKchPwConj.push_back(tLamKchP0010);
      tVecOfPairAnalysisLamKchPwConj.push_back(tALamKchM0010);

    FitSharedAnalyses *tFitSharedAnalysesLamKchPwConj = new FitSharedAnalyses(tVecOfPairAnalysisLamKchPwConj);

    if(bDoFit)
    {
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kLambda,0.19,0.1,1.0);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRef0,-1.7);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kImf0,1.1);
      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kd0,3.);

      tFitSharedAnalysesLamKchPwConj->SetSharedParameter(kRadius,Share01);
      
      tFitSharedAnalysesLamKchPwConj->CreateMinuitParameters();
      
      LednickyFitter* tLamKchPwConjFitter = new LednickyFitter(tFitSharedAnalysesLamKchPwConj);
      tLamKchPwConjFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      tLamKchPwConjFitter->SetApplyMomResCorrection(bApplyMomResCorrection);
      myFitter = tLamKchPwConjFitter;
      
      tLamKchPwConjFitter->DoFit();

      TCanvas* canLamKchPwConj = new TCanvas("canLamKchPwConj","canLamKchPwConj");
        canLamKchPwConj->Divide(2,1);
      canLamKchPwConj->cd(1);
        gStyle->SetOptTitle(0);
        tFitSharedAnalysesLamKchPwConj->DrawFit(0,"LamKchP0010");
    TPaveText* text1 = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
      text1->SetFillColor(0);
      text1->SetTextSize(0.05);
      text1->SetBorderSize(1);
      text1->AddText(TString(cAnalysisRootTags[tLamKchP0010->GetAnalysisType()]));
      text1->Draw();

      canLamKchPwConj->cd(2);
        gStyle->SetOptTitle(0);
        tFitSharedAnalysesLamKchPwConj->DrawFit(1,"ALamKchM0010");
    TPaveText* text2 = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
      text2->SetFillColor(0);
      text2->SetTextSize(0.05);
      text2->SetBorderSize(1);
      text2->AddText(TString(cAnalysisRootTags[tALamKchM0010->GetAnalysisType()]));
      text2->Draw();


      if(bSaveFigures) canLamKchPwConj->SaveAs(tSaveFiguresLocation+"LamKchPwConj_FitwRes.eps");

      TCanvas* canLamKchP = new TCanvas("canLamKchP","canLamKchP",400,600);
      canLamKchP->cd();
        gStyle->SetOptTitle(0);
      tFitSharedAnalysesLamKchPwConj->DrawFit(0,"LamKchP0010");
      text1->Draw();
      if(bSaveFigures) canLamKchP->SaveAs(tSaveFiguresLocation+"LamKchP_FitwRes.eps");

      TCanvas* canALamKchM = new TCanvas("canALamKchM","canALamKchM",400,600);
      canALamKchM->cd();
        gStyle->SetOptTitle(0);
      tFitSharedAnalysesLamKchPwConj->DrawFit(1,"ALamKchM0010");
      text2->Draw();
      if(bSaveFigures) canALamKchM->SaveAs(tSaveFiguresLocation+"ALamKchM_FitwRes.eps");
    }


    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchP0010 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfALamKchM0010 = tFitSharedAnalysesLamKchPwConj->GetKStarCfHeavy(1)->GetHeavyCf();

      TCanvas* canLamKchPwConj = new TCanvas("canLamKchPwConj","canLamKchPwConj");
        canLamKchPwConj->Divide(2,1);
      canLamKchPwConj->cd(1);
        tCfLamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchP0010->Draw();
      canLamKchPwConj->cd(2);
        tCfALamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchM0010->Draw();
    }

  }


  if(bRunLamKchMwConj)
  {
    FitPairAnalysis* tLamKchM0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kLamKchM,k0010);
    FitPairAnalysis* tALamKchP0010 = new FitPairAnalysis(FileLocationBase,FileLocationBaseMC,kALamKchP,k0010);

    vector<FitPairAnalysis*> tVecOfPairAnalysisLamKchMwConj;
      tVecOfPairAnalysisLamKchMwConj.push_back(tLamKchM0010);
      tVecOfPairAnalysisLamKchMwConj.push_back(tALamKchP0010);

    FitSharedAnalyses *tFitSharedAnalysesLamKchMwConj = new FitSharedAnalyses(tVecOfPairAnalysisLamKchMwConj);

    if(bDoFit)
    {
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kLambda,0.30,0.1,1.0);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRef0,0.10);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kImf0,0.40);
      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kd0,7.);

      tFitSharedAnalysesLamKchMwConj->SetSharedParameter(kRadius,Share01);
      
      tFitSharedAnalysesLamKchMwConj->CreateMinuitParameters();
      
      LednickyFitter* tLamKchMwConjFitter = new LednickyFitter(tFitSharedAnalysesLamKchMwConj);
      tLamKchMwConjFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
      tLamKchMwConjFitter->SetApplyMomResCorrection(bApplyMomResCorrection);
      myFitter = tLamKchMwConjFitter;
      
      tLamKchMwConjFitter->DoFit();
      
      TCanvas* canLamKchMwConj = new TCanvas("canLamKchMwConj","canLamKchMwConj");
        canLamKchMwConj->Divide(2,1);
      canLamKchMwConj->cd(1);
        gStyle->SetOptTitle(0);
        tFitSharedAnalysesLamKchMwConj->DrawFit(0,"LamKchM0010");
    TPaveText* text1 = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
      text1->SetFillColor(0);
      text1->SetTextSize(0.05);
      text1->SetBorderSize(1);
      text1->AddText(TString(cAnalysisRootTags[tLamKchM0010->GetAnalysisType()]));
      text1->Draw();


      canLamKchMwConj->cd(2);
        gStyle->SetOptTitle(0);
        tFitSharedAnalysesLamKchMwConj->DrawFit(1,"ALamKchP0010");
    TPaveText* text2 = new TPaveText(0.73,0.80,0.90,0.90,"NDC");
      text2->SetFillColor(0);
      text2->SetTextSize(0.05);
      text2->SetBorderSize(1);
      text2->AddText(TString(cAnalysisRootTags[tALamKchP0010->GetAnalysisType()]));
      text2->Draw();

      if(bSaveFigures) canLamKchMwConj->SaveAs(tSaveFiguresLocation+"LamKchMwConj_FitwRes.eps");


      TCanvas* canLamKchM = new TCanvas("canLamKchM","canLamKchM",400,600);
      canLamKchM->cd();
        gStyle->SetOptTitle(0);
      tFitSharedAnalysesLamKchMwConj->DrawFit(0,"LamKchM0010");
      text1->Draw();
      if(bSaveFigures) canLamKchM->SaveAs(tSaveFiguresLocation+"LamKchM_FitwRes.eps");

      TCanvas* canALamKchP = new TCanvas("canALamKchP","canALamKchP",400,600);
      canALamKchP->cd();
        gStyle->SetOptTitle(0);
      tFitSharedAnalysesLamKchMwConj->DrawFit(1,"ALamKchP0010");
      text2->Draw();
      if(bSaveFigures) canALamKchP->SaveAs(tSaveFiguresLocation+"ALamKchP_FitwRes.eps");

    }


    if(bDrawCfsOnly)
    {
      TH1* tCfLamKchM0010 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(0)->GetHeavyCf();
      TH1* tCfALamKchP0010 = tFitSharedAnalysesLamKchMwConj->GetKStarCfHeavy(1)->GetHeavyCf();

      TCanvas* canLamKchMwConj = new TCanvas("canLamKchMwConj","canLamKchMwConj");
        canLamKchMwConj->Divide(2,1);
      canLamKchMwConj->cd(1);
        tCfLamKchM0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfLamKchM0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfLamKchM0010->Draw();
      canLamKchMwConj->cd(2);
        tCfALamKchP0010->GetXaxis()->SetRangeUser(0.,0.5);
        tCfALamKchP0010->GetYaxis()->SetRangeUser(0.9,1.04);
        tCfALamKchP0010->Draw();
    }

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
