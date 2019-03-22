#include "FitSharedAnalyses.h"
#include "GlobalCoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"

GlobalCoulombFitter *myFitter = NULL;

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

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //!!!!!!!!!!!!!!!! NOTE:  must set myFitter = to whichever LednickyFitter object I want to use

//-----------------------------------------------------------------------------
//Be sure to set the following...

  TString tFileDirectory = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/";
  TString tFileLocationBase = tFileDirectory + TString("Results_cXicKch_20170505_ignoreOnFlyStatus");

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

  bool bDoFit = false;

  bool bDrawFit = true;
  int tParamSetToUse = 2;
  bool bSaveImage = true;

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
  tFitter->SetUseStaticPairs(true);
  tFitter->SetNPairsPerKStarBin(16384);
  tFitter->SetBinSizeKStar(0.01);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;

  if(bDoFit) tFitter->DoFit();

  if(bDrawFit)
  {
    int tNbinsK = 15;
    double tKMin = 0.;
    double tKMax = 0.15;

    double tLambdaXiKchP, tLambdaXiKchM, tRadius;

    if(tParamSetToUse==1)
    {
      //Chi2 = 351.433; Lower limit of 0.1 fm on R
      tLambdaXiKchP = 0.0261;
      tLambdaXiKchM = 0.2898;
      tRadius = 0.1;
    }
    else if(tParamSetToUse==2)
    {
      //Chi2 = 371.544; No limits
      tLambdaXiKchP = 0.0138;
      tLambdaXiKchM = 0.1617;
      tRadius = 10.84;
    }
    else assert(0);

    int tColorXiKchP, tColorTransparentXiKchP;
    tColorXiKchP = kRed+1;
    tColorTransparentXiKchP = TColor::GetColorTransparent(tColorXiKchP,0.2);

    int tColorXiKchM, tColorTransparentXiKchM;
    tColorXiKchM = kBlue+1;
    tColorTransparentXiKchM = TColor::GetColorTransparent(tColorXiKchM,0.2);


    TH1* tFitXiKchP = tFitter->CreateFitHistogramSampleComplete("tFitXiKchP", kXiKchP, tNbinsK, tKMin, tKMax, tLambdaXiKchP, tRadius,  0., 0., 0., 0., 0., 0., 1.0);
    TH1* tFitXiKchM = tFitter->CreateFitHistogramSampleComplete("tFitXiKchM", kXiKchM, tNbinsK, tKMin, tKMax, tLambdaXiKchM, tRadius,  0., 0., 0., 0., 0., 0., 1.0);

    TH1* tDataXiKchP = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetKStarCf();
    TH1* tDataXiKchPwSysErrors = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCfwSysErrors();
      tDataXiKchPwSysErrors->SetFillColor(tColorTransparentXiKchP);
      tDataXiKchPwSysErrors->SetFillStyle(1000);
      tDataXiKchPwSysErrors->SetLineColor(0);
      tDataXiKchPwSysErrors->SetLineWidth(0);

    TH1* tDataAXiKchM = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetKStarCf();
    TH1* tDataAXiKchMwSysErrors = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetCfwSysErrors();
      tDataAXiKchMwSysErrors->SetFillColor(tColorTransparentXiKchP);
      tDataAXiKchMwSysErrors->SetFillStyle(1000);
      tDataAXiKchMwSysErrors->SetLineColor(0);
      tDataAXiKchMwSysErrors->SetLineWidth(0);


    TH1* tDataXiKchM = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(2)->GetKStarCf();
    TH1* tDataXiKchMwSysErrors = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(2)->GetCfwSysErrors();
      tDataXiKchMwSysErrors->SetFillColor(tColorTransparentXiKchM);
      tDataXiKchMwSysErrors->SetFillStyle(1000);
      tDataXiKchMwSysErrors->SetLineColor(0);
      tDataXiKchMwSysErrors->SetLineWidth(0);

    TH1* tDataAXiKchP = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(3)->GetKStarCf();
    TH1* tDataAXiKchPwSysErrors = tFitter->GetFitSharedAnalyses()->GetFitPairAnalysis(3)->GetCfwSysErrors();
      tDataAXiKchPwSysErrors->SetFillColor(tColorTransparentXiKchM);
      tDataAXiKchPwSysErrors->SetFillStyle(1000);
      tDataAXiKchPwSysErrors->SetLineColor(0);
      tDataAXiKchPwSysErrors->SetLineWidth(0);
    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int tNx=2, tNy=2;
    float tMarginLeft = 0.10;
    float tMarginRight = 0.0025;
    float tMarginBottom = 0.15;
    float tMarginTop = 0.0025;
    double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
    tXLow = -0.02;
    tXHigh = 0.16;
    tYLow = 0.28;
    tYHigh = 1.72;
    CanvasPartition* tCanPart = new CanvasPartition("tCanPart", tNx, tNy, tXLow, tXHigh, tYLow, tYHigh, tMarginLeft, tMarginRight, tMarginBottom, tMarginTop);
    tCanPart->SetDrawOptStat(false);
    tCanPart->GetCanvas()->SetCanvasSize(1400,1000);

    //-------------------------------
    double tMarkerSize = 1.0;
    float tLabelSize = 0.2625, tLabelOffset=0.005;
    double tSystemInfoSize = 40;

    tCanPart->AddGraph(0, 0, tDataXiKchP, "", 20, tColorXiKchP, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tDataXiKchPwSysErrors, "", 20, tColorTransparentXiKchP, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tFitXiKchP, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tDataXiKchP, "", 20, tColorXiKchP, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

    TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
    TString tTextCentrality = TString(cPrettyCentralityTags[k0010]);
    TString tCombinedTextAnType = tTextAnType + TString("  ") +  tTextCentrality;
    TPaveText* tCombinedAnType = tCanPart->SetupTPaveText(tCombinedTextAnType,0,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
    tCanPart->AddPadPaveText(tCombinedAnType,0,0);

    TString tFitParameters = "Fit Parameters";
    TString tLambdaInfoXiKchP = TString::Format("#lambda = %0.2f", tLambdaXiKchP);
    TString tRadiusInfo = TString::Format("R = %0.2f fm", tRadius);
    TPaveText* tFitInfoXiKchP = tCanPart->SetupTPaveText(tFitParameters,0,0,0.60,0.15,0.15,0.20,63,(tSystemInfoSize/2));
    tFitInfoXiKchP->AddText(tLambdaInfoXiKchP);
    tFitInfoXiKchP->AddText(tRadiusInfo);
    tFitInfoXiKchP->SetTextAlign(12);
    tCanPart->AddPadPaveText(tFitInfoXiKchP,0,0);

    //-------------------------------
    tCanPart->AddGraph(1, 0, tDataAXiKchM, "", 20, tColorXiKchP, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tDataAXiKchMwSysErrors, "", 20, tColorTransparentXiKchP, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tFitXiKchP, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tDataAXiKchM, "", 20, tColorXiKchP, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);    

    TString tTextConjType = TString(cAnalysisRootTags[tConjType]);
    TString tCombinedTextConjType = tTextConjType + TString("  ") +  tTextCentrality;
    TPaveText* tCombinedConjType = tCanPart->SetupTPaveText(tCombinedTextConjType,1,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
    tCanPart->AddPadPaveText(tCombinedConjType,1,0);
    //-------------------------------
    tCanPart->AddGraph(0, 1, tDataXiKchM, "", 20, tColorXiKchM, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 1, tDataXiKchMwSysErrors, "", 20, tColorTransparentXiKchM, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 1, tFitXiKchM, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 1, tDataXiKchM, "", 20, tColorXiKchM, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

    TString tTextAnTypeOppSign = TString(cAnalysisRootTags[tAnTypeOppSign]);
    TString tCombinedTextAnTypeOppSign = tTextAnTypeOppSign + TString("  ") +  tTextCentrality;
    TPaveText* tCombinedAnTypeOppSign = tCanPart->SetupTPaveText(tCombinedTextAnTypeOppSign,0,1,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
    tCanPart->AddPadPaveText(tCombinedAnTypeOppSign,0,1);

    TString tLambdaInfoXiKchM = TString::Format("#lambda = %0.2f", tLambdaXiKchM);
    TPaveText* tFitInfoXiKchM = tCanPart->SetupTPaveText(tFitParameters,0,1,0.60,0.15,0.15,0.20,63,(tSystemInfoSize/2));
    tFitInfoXiKchM->AddText(tLambdaInfoXiKchM);
    tFitInfoXiKchM->AddText(tRadiusInfo);
    tFitInfoXiKchM->SetTextAlign(12);
    tCanPart->AddPadPaveText(tFitInfoXiKchM,0,1);
    //-------------------------------
    tCanPart->AddGraph(1, 1, tDataAXiKchP, "", 20, tColorXiKchM, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 1, tDataAXiKchPwSysErrors, "", 20, tColorTransparentXiKchM, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 1, tFitXiKchM, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 1, tDataAXiKchP, "", 20, tColorXiKchM, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);    

    TString tTextConjTypeOppSign = TString(cAnalysisRootTags[tConjTypeOppSign]);
    TString tCombinedTextConjTypeOppSign = tTextConjTypeOppSign + TString("  ") +  tTextCentrality;
    TPaveText* tCombinedConjTypeOppSign = tCanPart->SetupTPaveText(tCombinedTextConjTypeOppSign,1,1,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
    tCanPart->AddPadPaveText(tCombinedConjTypeOppSign,1,1);

    //------------------------------------------------------------
    tCanPart->SetDrawUnityLine(true);
    tCanPart->DrawAll();

    tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 50);
    tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43, 50, 0.05, 0.75);

    if(bSaveImage)
    {
      TString tSaveName = tFileDirectory + TString("GlobalCoulombOnlyFit");
      if(tParamSetToUse==1) tSaveName += TString("_Set1");
      else if(tParamSetToUse==2) tSaveName += TString("_Set2");
      tSaveName += TString(".pdf");
      tCanPart->GetCanvas()->SaveAs(tSaveName);
    }
  }

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
