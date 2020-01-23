#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

#include "FitValuesWriterwSysErrs.h"
class FitValuesWriterwSysErrs;

#include "CanvasPartition.h"
class CanvasPartition;

//________________________________________________________________________________________________________________
void CreateParamText(AnalysisType aAnType, IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const td1dVec &aSysErrors, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize, bool aDrawAll, double aChi2, double aNDF)
{
  int tNx=2, tNy=3;
  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  tLambda = aFit->GetParameter(0);
  tRadius = aFit->GetParameter(1);
  tReF0 = aFit->GetParameter(2);
  tImF0 = aFit->GetParameter(3);
  tD0 = aFit->GetParameter(4);

  tLambdaErr = aFit->GetParError(0);
  tRadiusErr = aFit->GetParError(1);
  tReF0Err = aFit->GetParError(2);
  tImF0Err = aFit->GetParError(3);
  tD0Err = aFit->GetParError(4);

  if(aIncResType != kIncludeNoResiduals)
  {
    double tScale = cAnalysisLambdaFactorsArr[aIncResType][aResPrimMaxDecayType][aAnType];
    tLambda /= tScale;
    tLambdaErr /= tScale;
  }

  double tChi2 = aChi2;
  int tNDF = aNDF;

  if(!aDrawAll) {aTextHeight /= 7; aTextHeight *= 3; aTextYmin += 1.25*aTextHeight; if(aNy==2) aTextYmin -= 0.25*aTextHeight;}
  TPaveText *tText = aCanPart->SetupTPaveText(/*"       stat.     sys."*/"",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f #pm %0.2f",tLambda,tLambdaErr,aSysErrors[0]));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f #pm %0.2f",tRadius,tRadiusErr,aSysErrors[1]));
  if(aDrawAll)
  {
    tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f #pm %0.2f",tReF0,tReF0Err,aSysErrors[2]));
    tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f #pm %0.2f",tImF0,tImF0Err,aSysErrors[3]));
    tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f #pm %0.2f",tD0,tD0Err,aSysErrors[4]));
  }

  tText->SetTextAlign(33);

  aCanPart->AddPadPaveText(tText,aNx,aNy);

  //--------------------------------
  if(aNx==0)
  {
//    TPaveText *tText2 = aCanPart->SetupTPaveText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF),aNx,aNy,0.125,0.05,aTextWidth,0.10,aTextFont,0.9*aTextSize);
//    aCanPart->AddPadPaveText(tText2,aNx,aNy);

    TPaveText *tText3 = aCanPart->SetupTPaveText("val. #pm stat. #pm sys.",aNx,aNy,0.255,0.48,aTextWidth,0.10,aTextFont,aTextSize);
    aCanPart->AddPadPaveText(tText3,aNx,aNy);
  }
}



//________________________________________________________________________________________________________________
TCanvas* DrawAll(vector<LednickyFitter*> &aFitters, td2dVec &aParamsSysErrs, IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, FitType aFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, double* aChi2All, double* aNDFAll)
{
  TString tCanvasName = TString("canKStarCfwFits");
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

  unsigned int tNAnalyses = 2*aFitters.size();
  assert(tNAnalyses==6);
  int tNx=2, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(1400,1500);

  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
//    FitSharedAnalyses* tSharedAn = aFitters[j]->GetFitSharedAnalyses();
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      int tColor, tColorTransparent;

      AnalysisType tAnType = aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetAnalysisType();

      if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
      else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
      else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
      else tColor=1;

      tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

      int tColorCorrectFit = kMagenta+1;
      int tColorNonFlatBgd = kGreen+2;

      TH1F* tCorrectedFitHisto = aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetCorrectedFitHistv2();
        tCorrectedFitHisto->SetLineWidth(2);

      //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
      TH1* tHistToPlot;
      if(aDrawSysErrors)
      {
        tHistToPlot = (TH1*)aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetCfwSysErrors();
          tHistToPlot->SetFillColor(tColorTransparent);
          tHistToPlot->SetFillStyle(1000);
          tHistToPlot->SetLineColor(0);
          tHistToPlot->SetLineWidth(0);
      }


      if(aDrawSysErrors) assert(tHistToPlot->GetBinWidth(1) == ((TH1*)aFitters[j]->GetFitSharedAnalyses()->GetKStarCfHeavy(i)->GetHeavyCfClone())->GetBinWidth(1));
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)aFitters[j]->GetFitSharedAnalyses()->GetKStarCfHeavy(i)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0");  //ex0 suppresses the error along x
      tCanPart->AddGraph(i,j,(TF1*)aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetNonFlatBackground(aNonFlatBgdFitType, aFitType, true, true),"",20,tColorNonFlatBgd);
      tCanPart->AddGraph(i,j,(TF1*)aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetPrimaryFit(),"");
      tCanPart->AddGraph(i,j,tCorrectedFitHisto,"",20,tColorCorrectFit,0.5,"lsame");
      if(aDrawSysErrors) tCanPart->AddGraph(i,j,tHistToPlot,"",20,tColorTransparent,0.5,"e2psame");
      tCanPart->AddGraph(i,j,(TH1*)aFitters[j]->GetFitSharedAnalyses()->GetKStarCfHeavy(i)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top

      TString tTextAnType = TString(cAnalysisRootTags[aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetAnalysisType()]);
      TString tTextCentrality = TString(cPrettyCentralityTags[aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetCentralityType()]);

      TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.70,0.825,0.15,0.10,43,20);
      tCanPart->AddPadPaveText(tCombined,i,j);
/*
      if(i==0 && j==0)
      {
        TString tTextAlicePrelim = TString("ALICE Preliminary");
        //TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.30,0.85,0.40,0.10,43,15);
        TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.175,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tAlicePrelim,i,j);
      }
*/
      if(i==1 && j==0)
      {
        TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        //TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.30,0.85,0.40,0.10,43,15);
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.125,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }

      if(i==1 && j==2)
      {
        TString tTextkTInfo = TString("All #it{k}_{T}");
        TPaveText* tkTInfo = tCanPart->SetupTPaveText(tTextkTInfo,i,j,0.79,0.05,0.20,0.15,43,17.5);
        tCanPart->AddPadPaveText(tkTInfo,i,j);
      }

      td1dVec tSysErrors = aParamsSysErrs[tAnalysisNumber];

      bool bDrawAll = false;
      if(i==0) bDrawAll = true;
      CreateParamText(tAnType, aFitters[j]->GetIncludeResidualsType(), aResPrimMaxDecayType, tCanPart, i, j,(TF1*)aFitters[j]->GetFitSharedAnalyses()->GetFitPairAnalysis(i)->GetPrimaryFit(), tSysErrors, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll, aChi2All[j], aNDFAll[j]);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
td2dVec GetParamValuesAndErrors(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  td1dVec tParVals(6), tParStatErr(6), tParSystErr(6);
  td2dVec tReturnVec;

  ParameterType tParamType;
  FitParameter* tFitParam;
  for(unsigned int iParam=0; iParam<5; iParam++)
  {
    tParamType = static_cast<ParameterType>(iParam);
    tFitParam = FitValuesWriterwSysErrs::GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, tParamType);

    tParVals[iParam] = tFitParam->GetFitValue();
    tParStatErr[iParam] = tFitParam->GetFitValueError();
    tParSystErr[iParam] = tFitParam->GetFitValueSysError();
  }
  //Normalization
  tParVals[5] = 1.;
  tParStatErr[5] = 0.;
  tParSystErr[5] = 0.;

  tReturnVec.push_back(tParVals);
  tReturnVec.push_back(tParStatErr);
  tReturnVec.push_back(tParSystErr);

  return tReturnVec;
}





//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20180505";


  CentralityType tCentType = k0010;  //TODO
  FitType tFitType = kChi2PML;

  bool SaveImages = false;
  bool bZoomROP=true;
  TString tSaveFileType = "pdf";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/Figures/";

  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kPolynomial;
  vector<NonFlatBgdFitType> tNonFlatBgdFitTypesStd{kLinear, kLinear,
                                                   kPolynomial, kPolynomial, kPolynomial, kPolynomial};
  IncludeResidualsType tIncResType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  TString tFitInfoTString = 
                                                                 FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypesStd, 
                                                                                                      tIncResType, tResPrimMaxDecayType, 
                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                      false, false, false, false, false, 
                                                                                                      true, false, false, true, 
                                                                                                      true, true);

  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tFitInfoTString.Data(), tFitInfoTString.Data());

  TString tMasterFileLocation_LamKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tMasterFileLocation_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20180505/MasterFitResults_20180505.txt";




  td2dVec tParsAndErrs_LamK0 = GetParamValuesAndErrors(tMasterFileLocation_LamK0, tSystematicsFileLocation, tFitInfoTString, kLamK0, tCentType);
  td2dVec tParsAndErrs_ALamK0 = GetParamValuesAndErrors(tMasterFileLocation_LamK0, tSystematicsFileLocation, tFitInfoTString, kALamK0, tCentType);

  td2dVec tParsAndErrs_LamKchP = GetParamValuesAndErrors(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchP, tCentType);
  td2dVec tParsAndErrs_ALamKchM = GetParamValuesAndErrors(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kALamKchM, tCentType);

  td2dVec tParsAndErrs_LamKchM = GetParamValuesAndErrors(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchM, tCentType);
  td2dVec tParsAndErrs_ALamKchP = GetParamValuesAndErrors(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kALamKchP, tCentType);

  td2dVec tParsAll{tParsAndErrs_LamK0[0], tParsAndErrs_ALamK0[0], tParsAndErrs_LamKchP[0], tParsAndErrs_ALamKchM[0], tParsAndErrs_LamKchM[0], tParsAndErrs_ALamKchP[0]};
  td2dVec tParErrsAll{tParsAndErrs_LamK0[1], tParsAndErrs_ALamK0[1], tParsAndErrs_LamKchP[1], tParsAndErrs_ALamKchM[1], tParsAndErrs_LamKchM[1], tParsAndErrs_ALamKchP[1]};
  td2dVec tParSysErrsAll{tParsAndErrs_LamK0[2], tParsAndErrs_ALamK0[2], tParsAndErrs_LamKchP[2], tParsAndErrs_ALamKchM[2], tParsAndErrs_LamKchM[2], tParsAndErrs_ALamKchP[2]};

  double tChi2All[6] = {357.0, 357.0, 425.8, 425.8, 284.0, 284.0};
  double tNDFAll[6] = {341, 341, 336, 336, 288, 288};


  TString tGeneralAnTypeName;
  TString tDirectoryBase, tFileLocationBase, tFileLocationBaseMC;
  vector<FitPairAnalysis*> tVecOfPairAn;
//  vector<FitSharedAnalyses*> tVecOfSharedAn;
  vector<LednickyFitter*> tVecOfLednickyFit;
  vector<int> tOrder{2, 4, 0}; //default = {0, 2, 4} = LamK0, LamKchP, LamKchM
//  for(int iAnType=0; iAnType<kXiKchP; iAnType+=2)
  double tOrderedChi2All[3];
  double tOrderedNDFAll[3];
  AnalysisType tAnType, tConjAnType;
  for(unsigned int i=0; i<tOrder.size(); i++)
  {
    int iAnType = tOrder[i];

    tVecOfPairAn.clear();

    tAnType = static_cast<AnalysisType>(iAnType);
    if(tAnType==kLamK0) tConjAnType = kALamK0;
    else if(tAnType==kLamKchP) tConjAnType = kALamKchM;
    else if(tAnType==kLamKchM) tConjAnType = kALamKchP;
    else assert(0);

    if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
    else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
    else assert(0);

    tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
    tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
    tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

    tVecOfPairAn.push_back(new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tAnType,tCentType,kTrain,2));
    tVecOfPairAn.push_back(new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,tConjAnType,tCentType,kTrain,2));


    FitSharedAnalyses* tFitSharedAn = new FitSharedAnalyses(tVecOfPairAn);
    tFitSharedAn->GetFitPairAnalysis(0)->GetFitParameter(kLambda)->SetStartValue(tParsAll[iAnType][0]);
    tFitSharedAn->GetFitPairAnalysis(1)->GetFitParameter(kLambda)->SetStartValue(tParsAll[iAnType+1][0]);
//    tFitSharedAn->SetSharedParameter(kLambda, {0}, tParsAll[iAnType][0]);
//    tFitSharedAn->SetSharedParameter(kLambda, {1}, tParsAll[iAnType+1][0]);

    tFitSharedAn->SetSharedParameter(kRadius, {0,1}, tParsAll[iAnType][1]);
    tFitSharedAn->SetSharedParameter(kRef0, {0,1}, tParsAll[iAnType][2]);
    tFitSharedAn->SetSharedParameter(kImf0, {0,1}, tParsAll[iAnType][3]);
    tFitSharedAn->SetSharedParameter(kd0, {0,1}, tParsAll[iAnType][4]);

    tFitSharedAn->CreateMinuitParameters();
    tFitSharedAn->SetFitType(tFitType);

    LednickyFitter* tLednickyFitter = new LednickyFitter(tFitSharedAn, 0.3);
    tLednickyFitter->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
    tLednickyFitter->SetNonFlatBgdFitType(tNonFlatBgdFitType);
    tLednickyFitter->SetApplyMomResCorrection(ApplyMomResCorrection);
    tLednickyFitter->SetIncludeResidualCorrelationsType(tIncResType);
    tLednickyFitter->SetChargedResidualsType(tChargedResidualsType);
    tLednickyFitter->SetResPrimMaxDecayType(tResPrimMaxDecayType);

    int tNpar = 10;
    double tChi2 = 0.;

    double tParams[tNpar] = {tParsAll[iAnType][0], tParsAll[iAnType+1][0], tParsAll[iAnType][1], tParsAll[iAnType][2], tParsAll[iAnType][3], tParsAll[iAnType][4], 1., 1., 1., 1.};
    double tParamErrs[tNpar] = {tParErrsAll[iAnType][0], tParErrsAll[iAnType+1][0], tParErrsAll[iAnType][1], tParErrsAll[iAnType][2], tParErrsAll[iAnType][3], tParErrsAll[iAnType][4], 0., 0., 0., 0.};
    tLednickyFitter->CalculateFitFunctionOnce(tNpar, tChi2, tParams, tParamErrs, tChi2All[iAnType], tNDFAll[iAnType]);

    tVecOfLednickyFit.push_back(tLednickyFitter);
    tOrderedChi2All[i] = tChi2All[iAnType];
    tOrderedNDFAll[i] = tNDFAll[iAnType];
  }


  TCanvas* tCan = DrawAll(tVecOfLednickyFit, tParSysErrsAll, tIncResType, tResPrimMaxDecayType, ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitType, tFitType, false, true, bZoomROP, tOrderedChi2All, tOrderedNDFAll);
  if(SaveImages)
  {
    vector<TString> tZoomROPText{"UnZoomed", ""};
    TString tSaveName = TString::Format("%sAllFitsCentral%s%s.%s", tSaveDir.Data(), tZoomROPText[bZoomROP].Data(), cCentralityTags[tCentType], tSaveFileType.Data());
    tCan->SaveAs(tSaveName);
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
