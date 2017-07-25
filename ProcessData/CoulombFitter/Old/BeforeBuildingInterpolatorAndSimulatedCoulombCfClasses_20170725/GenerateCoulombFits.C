#include "CoulombFitGenerator.h"
class CoulombFitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20170501";

  AnalysisType tAnType = kXiKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  bool tShareLambdaParams = true;
  bool tAllShareSingleLambdaParam = false;

  bool SaveImages = false;
  bool ApplyMomResCorrection = false;
  bool ApplyNonFlatBackgroundCorrection = false;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;
  bool IncludeResiduals = false;
  bool IncludeSingletAndTriplet = false;
  bool Fixd0 = true;
  bool CoulombOnlyFit = false;

  bool bDoFit = true;

  bool bDrawFit = true;
  bool bSaveImage = false;

  double tMaxKStarFit = 0.2;
  int tNPairsPerKStarBin = 50000;

  TString tGeneralAnTypeName = "cXicKch";


  TString tDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/";
  TString tFileLocationBase = tDirectoryBase + TString("Results_cXicKch_20170505_ignoreOnFlyStatus");

  TString tDirectoryBaseMC = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBaseMC.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());


  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  if(tAllShareSingleLambdaParam) tSaveNameModifier += TString("_SingleLamParam");
  CoulombFitGenerator* tXiKchP = new CoulombFitGenerator(tFileLocationBase,tFileLocationBaseMC, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, TString(""), IncludeSingletAndTriplet);
//  CoulombFitGenerator* tXiKchP = new CoulombFitGenerator(tFileLocationBase, tFileLocationBaseMC, tAnType,{k0010,k1030}, tAnRunType, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, TString(""), IncludeSingletAndTriplet);
//  tXiKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tXiKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tXiKchP->SetSaveLocationBase(tDirectoryBase,tSaveNameModifier);
  //tXiKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tXiKchP->DrawKStarCfs();

  //tXiKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tXiKchP->SetFixd0(Fixd0);
  tXiKchP->SetFixAllScattParams(CoulombOnlyFit);
  if(bDoFit) tXiKchP->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, IncludeResiduals, IncludeSingletAndTriplet, tNonFlatBgdFitType, tMaxKStarFit, tNPairsPerKStarBin);

  if(bDrawFit)
  {

    tXiKchP->DoFit(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, IncludeResiduals, IncludeSingletAndTriplet, tNonFlatBgdFitType, tMaxKStarFit, tNPairsPerKStarBin, true);
    int tNbinsK = 15;
    double tKMin = 0.;
    double tKMax = 0.15;

    double tLambda, tRadius;
    AnalysisType tConjType;
    if(tAnType==kXiKchP)
    {
      //chi2 = 89.9652
      tLambda = 0.427;
      tRadius = 8.43;

      tConjType = kAXiKchM;
    }
    else if(tAnType==kXiKchM)
    {
      //chi2 = 76.9154
      tLambda = 0.785;
      tRadius = 3.73;

      tConjType = kAXiKchP;
    }
    else assert(0);

    TH1* tFit = tXiKchP->GetCoulombFitter()->CreateFitHistogramSampleComplete("tFit", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius,  0., 0., 0., 0., 0., 0., 1.0);

    int tColor, tColorTransparent;
    if(tAnType==kXiKchP) tColor=kRed+1;
    else if(tAnType==kXiKchM) tColor=kBlue+1;
    else tColor=1;

    tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
  
    TH1* tData = tXiKchP->GetCoulombFitter()->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetKStarCf();
    TH1* tDatawSysErrors = tXiKchP->GetCoulombFitter()->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCfwSysErrors();
      tDatawSysErrors->SetFillColor(tColorTransparent);
      tDatawSysErrors->SetFillStyle(1000);
      tDatawSysErrors->SetLineColor(0);
      tDatawSysErrors->SetLineWidth(0);

    TH1* tConjData = tXiKchP->GetCoulombFitter()->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetKStarCf();
    TH1* tConjDatawSysErrors = tXiKchP->GetCoulombFitter()->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetCfwSysErrors();
      tConjDatawSysErrors->SetFillColor(tColorTransparent);
      tConjDatawSysErrors->SetFillStyle(1000);
      tConjDatawSysErrors->SetLineColor(0);
      tConjDatawSysErrors->SetLineWidth(0);


    double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
    tXLow = -0.02;
    tXHigh = 0.16;
    if(tAnType==kXiKchP)
    {
      tYLow = 0.92;
      tYHigh = 1.72;
    }
    else if(tAnType==kXiKchM)
    {
      tYLow = 0.28;
      tYHigh = 1.14;
    }
    else assert(0);

    int tNx=2, tNy=1;
    float tMarginLeft = 0.10;
    float tMarginRight = 0.0025;
    float tMarginBottom = 0.15;
    float tMarginTop = 0.0025;

    CanvasPartition* tCanPart = new CanvasPartition("tCanPart", tNx, tNy, tXLow, tXHigh, tYLow, tYHigh, tMarginLeft, tMarginRight, tMarginBottom, tMarginTop);
    tCanPart->SetDrawOptStat(false);
    tCanPart->GetCanvas()->SetCanvasSize(1400,500);

    double tMarkerSize = 1.0;
    float tLabelSize = 0.2625, tLabelOffset=0.005;
    double tSystemInfoSize = 40;

    //------------------------------------------------------------
    tCanPart->AddGraph(0, 0, tData, "", 20, tColor, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tDatawSysErrors, "", 20, tColorTransparent, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tFit, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(0, 0, tData, "", 20, tColor, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

    TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
    TString tTextCentrality = TString(cPrettyCentralityTags[k0010]);
    TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
    TPaveText* tCombined;
    if(tAnType==kXiKchP) tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
    else if(tAnType==kXiKchM) tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
    else assert(0);
    tCanPart->AddPadPaveText(tCombined,0,0);

    TString tFitParameters = "Fit Parameters";
    TString tLambdaInfo = TString::Format("#lambda = %0.2f", tLambda);
    TString tRadiusInfo = TString::Format("R = %0.2f fm", tRadius);
    TPaveText* tFitInfo = tCanPart->SetupTPaveText(tFitParameters,0,0,0.60,0.35,0.15,0.20,63,(tSystemInfoSize/2));
    tFitInfo->AddText(tLambdaInfo);
    tFitInfo->AddText(tRadiusInfo);
    tFitInfo->SetTextAlign(12);
    tCanPart->AddPadPaveText(tFitInfo,0,0);
    //------------------------------------------------------------
    tCanPart->AddGraph(1, 0, tConjData, "", 20, tColor, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tConjDatawSysErrors, "", 20, tColorTransparent, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tFit, "", 20, kBlack, tMarkerSize, "lsame", tLabelSize, tLabelOffset);
    tCanPart->AddGraph(1, 0, tConjData, "", 20, tColor, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

    TString tTextConjType = TString(cAnalysisRootTags[tConjType]);
    TString tCombinedTextConj = tTextConjType + TString("  ") +  tTextCentrality;
    TPaveText* tCombinedConj;
    if(tAnType==kXiKchP) tCombinedConj = tCanPart->SetupTPaveText(tCombinedTextConj,1,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
    else if(tAnType==kXiKchM) tCombinedConj = tCanPart->SetupTPaveText(tCombinedTextConj,1,0,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
    else assert(0);
    tCanPart->AddPadPaveText(tCombinedConj,1,0);

    //------------------------------------------------------------
    tCanPart->SetDrawUnityLine(true);
    tCanPart->DrawAll();

    tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 50, 0.315, 0.03);
    tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43, 50, 0.05, 0.75);

    if(bSaveImage)
    {
      TString tSaveName = tDirectoryBase;
      tSaveName += TString("CoulombOnlyFit") + cAnalysisBaseTags[tAnType] + TString(cCentralityTags[k0010]);
      tSaveName += TString(".pdf");
      tCanPart->GetCanvas()->SaveAs(tSaveName);
    }
  }
//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
