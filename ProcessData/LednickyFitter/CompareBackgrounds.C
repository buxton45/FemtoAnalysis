#include "FitGenerator.h"
class FitGenerator;

#include "DrawAMPTCfs.h"


//_________________________________________________________________________________________
TH1D* Get1dTHERMHist(TString FileName, TString HistName)
{
  TFile f1(FileName);
  TH1D *ReturnHist = (TH1D*)f1.Get(HistName);

  TH1D *ReturnHistClone = (TH1D*)ReturnHist->Clone();
  ReturnHistClone->SetDirectory(0);

  return ReturnHistClone;
}


//________________________________________________________________________________________________________________
TH1* GetTHERMCf(AnalysisType aAnType, int aImpactParam=8, bool aCombineConj = true, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  //--------------------------------

  TString tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", aImpactParam);
  TString tFileName = "CorrelationFunctions_RandomEPs.root";
  TString tFileLocation = TString::Format("%s%s", tDirectory.Data(), tFileName.Data());
  //--------------------------------
  TH1D* tNum1 = Get1dTHERMHist(tFileLocation, TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]));
  TH1D* tDen1 = Get1dTHERMHist(tFileLocation, TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]));
  CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                tNum1, tDen1, aMinNorm, aMaxNorm);
  tCfLite1->Rebin(aRebin);

  TH1D* tNum2 = Get1dTHERMHist(tFileLocation, TString::Format("NumFull%s", cAnalysisBaseTags[tConjAnType]));
  TH1D* tDen2 = Get1dTHERMHist(tFileLocation, TString::Format("DenFull%s", cAnalysisBaseTags[tConjAnType]));
  CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]), 
                                tNum2, tDen2, aMinNorm, aMaxNorm);
  tCfLite2->Rebin(aRebin);

  if(!aCombineConj) return tCfLite1->Cf();
  else
  {
    vector<CfLite*> tCfLiteVec {tCfLite1, tCfLite2};
    CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                                    TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                                    tCfLiteVec, aMinNorm, aMaxNorm);
    return tCfHeavy->GetHeavyCf();
  }
}



//________________________________________________________________________________________________________________
TH1* GetAMPTCf(AnalysisType aAnType, CentralityType aCentType, bool aCombineAllcLamcKch, bool aCombineConjugates)
{
  bool tCombineAllcLamcKch = aCombineAllcLamcKch;
  bool tCombineConjugates = aCombineConjugates;
  if( (aAnType==kLamK0 || aAnType==kALamK0) && tCombineAllcLamcKch) {tCombineAllcLamcKch=false; tCombineConjugates=true;}


  TString tResultsDate = "20180312";
  CentralityType tAnDirCentType = kMB;    //If data stored in separate directories for each centrality, 0010, 1030, 3050
                                          //    choose corresponding CentralityType
                                          //If data stored in one, large, 0100 directory, choose kMB
  if(tResultsDate.EqualTo("20180312")) tAnDirCentType = kMB;
  else tAnDirCentType = aCentType;

  bool tCombineCentFiles = true;
  TString tCentralityFile;
  if     (aCentType==k0010) tCentralityFile = "0005";
  else if(aCentType==k1030) tCentralityFile = "1020";
  else if(aCentType==k3050) tCentralityFile = "3040";
  else assert(0);
  //----------
  int tRebin = 4;
  double tMinNorm=0.32, tMaxNorm=0.40;

  //-----------------------------------------------------------------------------
  TH1* tCf = TypicalGetAMPTCf(tResultsDate, tAnDirCentType, aAnType, tCentralityFile, tCombineAllcLamcKch, tCombineConjugates, tCombineCentFiles, tRebin, tMinNorm, tMaxNorm);

  return tCf;
}

//________________________________________________________________________________________________________________
TCanvas* CompareDataToAMPT(FitGenerator* aGen, bool aDrawTHERM, bool aZoom0010=false)
{
  bool tCombineAllcLamcKch = false; 
  bool tCombineConjugates = true;

  AnalysisType tAnType = aGen->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  vector<CentralityType> tCentralityTypes = aGen->GetCentralityTypes();
  int tNAnalyses = aGen->GetNAnalyses();

  TString tCanvasName = TString::Format("CompareDataToAMPT_%s", cAnalysisBaseTags[tAnType]);
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[tCentralityTypes[i]]);
  if(aZoom0010) tCanvasName += TString("_Zoom0010");

  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 1.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyleData = 20;
  int tMarkerStyleAMPT = 25;
  int tMarkerStyleTHERM = 26;
  double tMarkerSize = 0.5;

  int tColorData, tColorAMPT, tColorTHERM;
  if     (tAnType==kLamK0 || tAnType==kALamK0)     tColorData = kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColorData = kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColorData = kBlue+1;
  else assert(0);

  tColorAMPT = kMagenta;
  tColorTHERM = kGreen;

  int tNx_Leg=0, tNy_Leg=0;

  TH1 *tDataCf, *tAMPTCf, *tTHERMCf;
  CentralityType tCentType;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      tAnType = aGen->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      tCentType = aGen->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      tDataCf = aGen->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tAMPTCf = GetAMPTCf(tAnType, tCentType, tCombineAllcLamcKch, tCombineConjugates);
      //---------------------------------------------------------------------------------------------------------
      tCanPart->AddGraph(i, j, tDataCf, "", tMarkerStyleData, tColorData, tMarkerSize);
      tCanPart->AddGraph(i, j, tAMPTCf, "", tMarkerStyleAMPT, tColorAMPT, tMarkerSize);
      if(aDrawTHERM) 
      {
        int tImpactParam = 2;
        if     (tCentType==k0010) tImpactParam=2;
        else if(tCentType==k1030) tImpactParam=8;
        else if(tCentType==k3050) tImpactParam=8;
        else assert(0);

        tTHERMCf = GetTHERMCf(tAnType, tImpactParam, tCombineConjugates);
        tCanPart->AddGraph(i, j, tTHERMCf, "", tMarkerStyleTHERM, tColorTHERM, tMarkerSize);
      }
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart->SetupTLegend(cAnalysisRootTags[tAnType], i, j, 0.25, 0.05, 0.35, 0.50);
        tCanPart->AddLegendEntry(i, j, tDataCf, "Data", "p");
        tCanPart->AddLegendEntry(i, j, tAMPTCf, "AMPT", "p");
        if(aDrawTHERM) tCanPart->AddLegendEntry(i, j, tTHERMCf, "THERM", "p");
      }
    }
  }

  if(aZoom0010)
  {
    double tZoomYLow = 0.965;
    double tZoomYHigh = 1.015;

    ((TH1*)tCanPart->GetGraphsInPad(0,0)->At(0))->GetYaxis()->SetRangeUser(tZoomYLow, tZoomYHigh);
    ((TH1*)tCanPart->GetGraphsInPad(1,0)->At(0))->GetYaxis()->SetRangeUser(tZoomYLow, tZoomYHigh);
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);


  return tCanPart->GetCanvas();

}



//________________________________________________________________________________________________________________
TCanvas* CompareLamKchAvgToLamK0(FitGenerator* aLamKchP, FitGenerator* aLamKchM, FitGenerator* aLamK0, bool aDrawIndividualKchAlso=false)
{
  TString tCanvasName = "CompareLamKchAvgToLamK0";
  if(aDrawIndividualKchAlso) tCanvasName += TString("_wIndivKch");

  vector<CentralityType> tCentralityTypes = aLamKchP->GetCentralityTypes();
  int tNAnalyses = aLamKchP->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[tCentralityTypes[i]]);


  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 1.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  double tMarkerSizeLarge = 0.5;
  double tMarkerSizeSmall = 0.25;

  int tColorLamK0 = kBlack;
  int tColorLamKchP = kRed+1;
  int tColorLamKchM = kBlue+1;
  int tColorLamKchAvg = kMagenta+1;

  int tNx_Leg=0, tNy_Leg=0;

  TH1 *tCfLamKchP, *tCfLamKchM, *tCfLamK0;
  TH1 *tCfLamKchAvg;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCfLamKchP = aLamKchP->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamKchM = aLamKchM->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamK0 = aLamK0->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCfLamKchAvg = (TH1*)tCfLamKchP->Clone();
      tCfLamKchAvg->Add(tCfLamKchM);
      tCfLamKchAvg->Scale(0.5);

      tCanPart->AddGraph(i, j, tCfLamKchAvg, "", tMarkerStyle, tColorLamKchAvg, tMarkerSizeLarge);
      tCanPart->AddGraph(i, j, tCfLamK0, "", tMarkerStyle, tColorLamK0, tMarkerSizeLarge);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart->SetupTLegend("", i, j, 0.25, 0.05, 0.35, 0.50);
        tCanPart->AddLegendEntry(i, j, tCfLamK0, cAnalysisRootTags[kLamK0], "p");
        tCanPart->AddLegendEntry(i, j, tCfLamKchAvg, TString::Format("0.5*(%s+%s)", cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kLamKchM]), "p");
      }

      if(aDrawIndividualKchAlso)
      {
        tCanPart->AddGraph(i, j, tCfLamKchP, "", tMarkerStyle, tColorLamKchP, tMarkerSizeSmall);
        tCanPart->AddGraph(i, j, tCfLamKchM, "", tMarkerStyle, tColorLamKchM, tMarkerSizeSmall);
        if(i==tNx_Leg && j==tNy_Leg)
        {
          tCanPart->AddLegendEntry(i, j, tCfLamKchP, cAnalysisRootTags[kLamKchP], "p");
          tCanPart->AddLegendEntry(i, j, tCfLamKchM, cAnalysisRootTags[kLamKchM], "p");
        }
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);


  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
TCanvas* CompareEPBinning(FitGenerator* aGenNoEPBin, FitGenerator* aGenEPBin8, FitGenerator* aGenEPBin16)
{
  AnalysisType tAnType = aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();

  TString tCanvasName = TString::Format("CompareEPBinning_%s", cAnalysisBaseTags[tAnType]);
  vector<CentralityType> tCentralityTypes = aGenNoEPBin->GetCentralityTypes();
  int tNAnalyses = aGenNoEPBin->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[tCentralityTypes[i]]);


  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 1.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyleNoBin = 20;
  int tMarkerStyleBin8 = 22;
  int tMarkerStyleBin16 = 32;

  double tMarkerSize = 0.5;

  int tColor;
  if     (tAnType==kLamK0 || tAnType==kALamK0)     tColor = kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor = kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor = kBlue+1;
  else assert(0);

  int tColorBin8 = kMagenta;
  int tColorBin16 = kGreen;

  int tNx_Leg=0, tNy_Leg=0;

  TH1 *tCfNoEPBin, *tCfEPBin8, *tCfEPBin16;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCfNoEPBin = aGenNoEPBin->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfEPBin8 = aGenEPBin8->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfEPBin16 = aGenEPBin16->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCanPart->AddGraph(i, j, tCfNoEPBin, "", tMarkerStyleNoBin, tColor, tMarkerSize);
      tCanPart->AddGraph(i, j, tCfEPBin8, "", tMarkerStyleBin8, tColorBin8, tMarkerSize);
      tCanPart->AddGraph(i, j, tCfEPBin16, "", tMarkerStyleBin16, tColorBin16, tMarkerSize);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart->SetupTLegend("", i, j, 0.25, 0.05, 0.35, 0.50);
        tCanPart->AddLegendEntry(i, j, tCfNoEPBin, "No Binning", "p");
        tCanPart->AddLegendEntry(i, j, tCfEPBin8, "8 Bins", "p");
        tCanPart->AddLegendEntry(i, j, tCfEPBin16, "16 Bins", "p");
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);


  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
TObjArray* DrawCfRatiosAndDiffs(FitGenerator* aGen1, FitGenerator* aGen2)
{
  AnalysisType tAnType1 = aGen1->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aGen2->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  TString tCanvasName1 = TString::Format("DrawCfRatios_%svs%s", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2]);
  TString tCanvasName2 = TString::Format("DrawCfDiffs_%svs%s", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2]);

  vector<CentralityType> tCentralityTypes = aGen1->GetCentralityTypes();
  int tNAnalyses = aGen1->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) 
  {
    tCanvasName1 += TString(cCentralityTags[tCentralityTypes[i]]);
    tCanvasName2 += TString(cCentralityTags[tCentralityTypes[i]]);
  }

  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;

  double tYLow1 = 0.97;
  double tYHigh1 = 1.09;

  double tYLow2 = -0.09;
  double tYHigh2 = 0.19;

  CanvasPartition* tCanPart1 = new CanvasPartition(tCanvasName1,tNx,tNy,tXLow,tXHigh,tYLow1,tYHigh1,0.12,0.05,0.13,0.05);
  CanvasPartition* tCanPart2 = new CanvasPartition(tCanvasName2,tNx,tNy,tXLow,tXHigh,tYLow2,tYHigh2,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  double tMarkerSize = 0.5;

  int tColor;

  if( ((tAnType1==kLamK0 || tAnType1==kALamK0) && (tAnType2==kLamKchP || tAnType2==kALamKchM)) ||
      ((tAnType2==kLamK0 || tAnType2==kALamK0) && (tAnType1==kLamKchP || tAnType1==kALamKchM)) )
  {
    tColor = kRed+2;
  }
  else if( ((tAnType1==kLamK0 || tAnType1==kALamK0) && (tAnType2==kLamKchM || tAnType2==kALamKchP)) ||
           ((tAnType2==kLamK0 || tAnType2==kALamK0) && (tAnType1==kLamKchM || tAnType1==kALamKchP)) )
  {
    tColor = kBlue+2;
  }
  else if( ((tAnType1==kLamKchP || tAnType1==kALamKchM) && (tAnType2==kLamKchM || tAnType2==kALamKchP)) ||
           ((tAnType2==kLamKchP || tAnType2==kALamKchM) && (tAnType1==kLamKchM || tAnType1==kALamKchP)) )
  {
    tColor = kMagenta+1;
  }
  else tColor = kYellow;


  int tNx_Leg=0, tNy_Leg=0;

  TString tTitle1 = TString::Format("#frac{C_{%s}(k*)}{C_{%s}(k*)}", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  TString tTitle2 = TString::Format("C_{%s}(k*)-C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);

  TH1 *tCf1, *tCf2;
  TH1 *tCfRatio, *tCfDiff;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCf1 = aGen1->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCf2 = aGen2->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCfRatio = (TH1*)tCf1->Clone();
      tCfRatio->Divide(tCf2);

      tCfDiff = (TH1*)tCf1->Clone();
      tCfDiff->Add(tCf2, -1.0);

      tCanPart1->AddGraph(i, j, tCfRatio, "", tMarkerStyle, tColor, tMarkerSize);
      tCanPart2->AddGraph(i, j, tCfDiff, "", tMarkerStyle, tColor, tMarkerSize);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart1->SetupTLegend("", i, j, 0.25, 0.50, 0.35, 0.25);
        tCanPart1->AddLegendEntry(i, j, tCfRatio, tTitle1.Data(), "p");

        tCanPart2->SetupTLegend("", i, j, 0.25, 0.50, 0.60, 0.25);
        tCanPart2->AddLegendEntry(i, j, tCfDiff, tTitle2.Data(), "p");

      }
    }
  }

  TString tYaxisTitle1 = TString::Format("C_{%s}(k*)/C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  tCanPart1->SetDrawUnityLine(true);
  tCanPart1->DrawAll();
  tCanPart1->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart1->DrawYaxisTitle(tYaxisTitle1,43,20,0.075,0.65);

  TString tYaxisTitle2 = TString::Format("C_{%s}(k*)-C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  tCanPart2->SetDrawUnityLine(true);
  tCanPart2->DrawAll();
  tCanPart2->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart2->DrawYaxisTitle(tYaxisTitle2,43,20,0.075,0.65);


  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)tCanPart1->GetCanvas());
  tReturnArray->Add((TCanvas*)tCanPart2->GetCanvas());
  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* DrawCfRatiosAndDiffs_LamKchAvgToLamK0(FitGenerator* aLamKchP, FitGenerator* aLamKchM, FitGenerator* aLamK0)
{
  TString tCanvasName1 = "DrawCfRatios_LamKchAvgToLamK0";
  TString tCanvasName2 = "DrawCfDiffs_LamKchAvgToLamK0";


  vector<CentralityType> tCentralityTypes = aLamKchP->GetCentralityTypes();
  int tNAnalyses = aLamKchP->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) 
  {
    tCanvasName1 += TString(cCentralityTags[tCentralityTypes[i]]);
    tCanvasName2 += TString(cCentralityTags[tCentralityTypes[i]]);
  }


  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;

  double tYLow1 = 0.97;
  double tYHigh1 = 1.09;

  double tYLow2 = -0.09;
  double tYHigh2 = 0.19;


  CanvasPartition* tCanPart1 = new CanvasPartition(tCanvasName1,tNx,tNy,tXLow,tXHigh,tYLow1,tYHigh1,0.12,0.05,0.13,0.05);
  CanvasPartition* tCanPart2 = new CanvasPartition(tCanvasName2,tNx,tNy,tXLow,tXHigh,tYLow2,tYHigh2,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;


  int tColor = kGray;
  int tMarkerStyle = 20;
  double tMarkerSize = 0.5;

  int tNx_Leg=0, tNy_Leg=0;

  TString tTitle1 = "#frac{C_{#LambdaK+ + #LambdaK-}(k*)}{C_{#LambdaK^{0}_{S}}(k*)}";
  TString tTitle2 = "C_{#LambdaK+ + #LambdaK-}(k*)-C_{#LambdaK^{0}_{S}}(k*)";


  TH1 *tCfLamKchP, *tCfLamKchM, *tCfLamK0;
  TH1 *tCfLamKchAvg;
  TH1 *tCfRatio, *tCfDiff;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCfLamKchP = aLamKchP->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamKchM = aLamKchM->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamK0 = aLamK0->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCfLamKchAvg = (TH1*)tCfLamKchP->Clone();
      tCfLamKchAvg->Add(tCfLamKchM);
      tCfLamKchAvg->Scale(0.5);
      //---------------------------------------------------------------------------------------------------------
      tCfRatio = (TH1*)tCfLamKchAvg->Clone();
      tCfRatio->Divide(tCfLamK0);

      tCfDiff = (TH1*)tCfLamKchAvg->Clone();
      tCfDiff->Add(tCfLamK0, -1.0);

      tCanPart1->AddGraph(i, j, tCfRatio, "", tMarkerStyle, tColor, tMarkerSize);
      tCanPart2->AddGraph(i, j, tCfDiff, "", tMarkerStyle, tColor, tMarkerSize);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart1->SetupTLegend("", i, j, 0.25, 0.50, 0.35, 0.25);
        tCanPart1->AddLegendEntry(i, j, tCfRatio, tTitle1.Data(), "p");

        tCanPart2->SetupTLegend("", i, j, 0.25, 0.50, 0.60, 0.25);
        tCanPart2->AddLegendEntry(i, j, tCfDiff, tTitle2.Data(), "p");

      }
    }
  }

  TString tYaxisTitle1 = "C_{#LambdaK+ + #LambdaK-}(k*)/C_{#LambdaK^{0}_{S}}(k*)";
  tCanPart1->SetDrawUnityLine(true);
  tCanPart1->DrawAll();
  tCanPart1->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart1->DrawYaxisTitle(tYaxisTitle1,43,20,0.075,0.65);

  TString tYaxisTitle2 = "C_{#LambdaK+ + #LambdaK-}(k*)-C_{#LambdaK^{0}_{S}}(k*)";
  tCanPart2->SetDrawUnityLine(true);
  tCanPart2->DrawAll();
  tCanPart2->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart2->DrawYaxisTitle(tYaxisTitle2,43,20,0.075,0.65);


  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)tCanPart1->GetCanvas());
  tReturnArray->Add((TCanvas*)tCanPart2->GetCanvas());
  return tReturnArray;
}


//________________________________________________________________________________________________________________
TObjArray* DrawCfRatiosAndDiffs_Centralities(FitGenerator* aGen1)
{
  AnalysisType tAnType1 = aGen1->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tConjAnType1 = aGen1->GetSharedAn()->GetFitPairAnalysis(1)->GetAnalysisType();
  TString tCanvasName1 = TString::Format("DrawCfRatios_Centralities_%s", cAnalysisBaseTags[tAnType1]);
  TString tCanvasName2 = TString::Format("DrawCfDiffs_Centralities_%s", cAnalysisBaseTags[tAnType1]);

  vector<CentralityType> tCentralityTypes = aGen1->GetCentralityTypes();
  int tNAnalyses = aGen1->GetNAnalyses();

  assert(tCentralityTypes.size()==3);  //expecting k0010, k1030, and k3050
  assert(tNAnalyses==6);  //expecting pair with conj

  int tNx=2, tNy=1;

  double tXLow = -0.02;
  double tXHigh = 1.99;

  double tYLow1 = 0.97;
  double tYHigh1 = 1.09;

  double tYLow2 = -0.09;
  double tYHigh2 = 0.19;

  CanvasPartition* tCanPart1 = new CanvasPartition(tCanvasName1,tNx,tNy,tXLow,tXHigh,tYLow1,tYHigh1,0.12,0.05,0.13,0.05);
  CanvasPartition* tCanPart2 = new CanvasPartition(tCanvasName2,tNx,tNy,tXLow,tXHigh,tYLow2,tYHigh2,0.12,0.05,0.13,0.05);

  int tMarkerStyle = 20;
  double tMarkerSize = 0.5;

  int tColor1, tColor2, tColor3;
  tColor1 = kBlack;
  tColor2 = kRed;
  tColor3 = kBlue;

  TH1 *tCf_0010, *tCf_1030, *tCf_3050;
  TH1 *tRatio1, *tRatio2, *tRatio3;
  TH1 *tDiff1, *tDiff2, *tDiff3;

  TH1 *tConjCf_0010, *tConjCf_1030, *tConjCf_3050;
  TH1 *tConjRatio1, *tConjRatio2, *tConjRatio3;
  TH1 *tConjDiff1, *tConjDiff2, *tConjDiff3;

  //----------------------------------------------------------------------------

  tCf_0010     = aGen1->GetSharedAn()->GetKStarCfHeavy(0)->GetHeavyCfClone();
  tConjCf_0010 = aGen1->GetSharedAn()->GetKStarCfHeavy(1)->GetHeavyCfClone();

  tCf_1030     = aGen1->GetSharedAn()->GetKStarCfHeavy(2)->GetHeavyCfClone();
  tConjCf_1030 = aGen1->GetSharedAn()->GetKStarCfHeavy(3)->GetHeavyCfClone();

  tCf_3050     = aGen1->GetSharedAn()->GetKStarCfHeavy(4)->GetHeavyCfClone();
  tConjCf_3050 = aGen1->GetSharedAn()->GetKStarCfHeavy(5)->GetHeavyCfClone();

  //----------------------------------------------------------------------------

  tRatio1 = (TH1*)tCf_0010->Clone();
  tRatio1->Divide(tCf_1030);

  tRatio2 = (TH1*)tCf_1030->Clone();
  tRatio2->Divide(tCf_3050);

  tRatio3 = (TH1*)tCf_0010->Clone();
  tRatio3->Divide(tCf_3050);

  //------------

  tDiff1 = (TH1*)tCf_0010->Clone();
  tDiff1->Add(tCf_1030, -1.);

  tDiff2 = (TH1*)tCf_1030->Clone();
  tDiff2->Add(tCf_3050, -1.);

  tDiff3 = (TH1*)tCf_0010->Clone();
  tDiff3->Add(tCf_3050, -1.);

  //-------------------------------------

  tConjRatio1 = (TH1*)tConjCf_0010->Clone();
  tConjRatio1->Divide(tConjCf_1030);

  tConjRatio2 = (TH1*)tConjCf_1030->Clone();
  tConjRatio2->Divide(tConjCf_3050);

  tConjRatio3 = (TH1*)tConjCf_0010->Clone();
  tConjRatio3->Divide(tConjCf_3050);

  //------------

  tConjDiff1 = (TH1*)tConjCf_0010->Clone();
  tConjDiff1->Add(tConjCf_1030, -1.);

  tConjDiff2 = (TH1*)tConjCf_1030->Clone();
  tConjDiff2->Add(tConjCf_3050, -1.);

  tConjDiff3 = (TH1*)tConjCf_0010->Clone();
  tConjDiff3->Add(tConjCf_3050, -1.);

  //---------------------------------------------------------------------------------------------------------

  tCanPart1->AddGraph(0, 0, tRatio1, "", tMarkerStyle, tColor1, tMarkerSize);
  tCanPart1->AddGraph(0, 0, tRatio2, "", tMarkerStyle, tColor2, tMarkerSize);
  tCanPart1->AddGraph(0, 0, tRatio3, "", tMarkerStyle, tColor3, tMarkerSize);
  
  tCanPart1->SetupTLegend(TString::Format("%s (Ratios)", cAnalysisRootTags[tAnType1]), 0, 0, 0.25, 0.70, 0.50, 0.25);
  tCanPart1->AddLegendEntry(0, 0, tRatio1, TString::Format("%s / %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k1030]), "p");
  tCanPart1->AddLegendEntry(0, 0, tRatio2, TString::Format("%s / %s",cPrettyCentralityTags[k1030], cPrettyCentralityTags[k3050]), "p");
  tCanPart1->AddLegendEntry(0, 0, tRatio3, TString::Format("%s / %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k3050]), "p");

  //------------

  tCanPart1->AddGraph(1, 0, tConjRatio1, "", tMarkerStyle, tColor1, tMarkerSize);
  tCanPart1->AddGraph(1, 0, tConjRatio2, "", tMarkerStyle, tColor2, tMarkerSize);
  tCanPart1->AddGraph(1, 0, tConjRatio3, "", tMarkerStyle, tColor3, tMarkerSize);

  tCanPart1->SetupTLegend(TString::Format("%s (Ratios)", cAnalysisRootTags[tConjAnType1]), 1, 0, 0.25, 0.70, 0.50, 0.25);
  tCanPart1->AddLegendEntry(1, 0, tRatio1, TString::Format("%s / %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k1030]), "p");
  tCanPart1->AddLegendEntry(1, 0, tRatio2, TString::Format("%s / %s",cPrettyCentralityTags[k1030], cPrettyCentralityTags[k3050]), "p");
  tCanPart1->AddLegendEntry(1, 0, tRatio3, TString::Format("%s / %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k3050]), "p");

  //---------------------------------------------------------------------------------------------------------

  tCanPart2->AddGraph(0, 0, tDiff1, "", tMarkerStyle, tColor1, tMarkerSize);
  tCanPart2->AddGraph(0, 0, tDiff2, "", tMarkerStyle, tColor2, tMarkerSize);
  tCanPart2->AddGraph(0, 0, tDiff3, "", tMarkerStyle, tColor3, tMarkerSize);

  tCanPart2->SetupTLegend(TString::Format("%s (Diffs)", cAnalysisRootTags[tAnType1]), 0, 0, 0.25, 0.70, 0.50, 0.25);
  tCanPart2->AddLegendEntry(0, 0, tRatio1, TString::Format("%s - %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k1030]), "p");
  tCanPart2->AddLegendEntry(0, 0, tRatio2, TString::Format("%s - %s",cPrettyCentralityTags[k1030], cPrettyCentralityTags[k3050]), "p");
  tCanPart2->AddLegendEntry(0, 0, tRatio3, TString::Format("%s - %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k3050]), "p");

  //------------

  tCanPart2->AddGraph(1, 0, tConjDiff1, "", tMarkerStyle, tColor1, tMarkerSize);
  tCanPart2->AddGraph(1, 0, tConjDiff2, "", tMarkerStyle, tColor2, tMarkerSize);
  tCanPart2->AddGraph(1, 0, tConjDiff3, "", tMarkerStyle, tColor3, tMarkerSize);

  tCanPart2->SetupTLegend(TString::Format("%s (Diffs)", cAnalysisRootTags[tConjAnType1]), 1, 0, 0.25, 0.70, 0.50, 0.25);
  tCanPart2->AddLegendEntry(1, 0, tRatio1, TString::Format("%s - %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k1030]), "p");
  tCanPart2->AddLegendEntry(1, 0, tRatio2, TString::Format("%s - %s",cPrettyCentralityTags[k1030], cPrettyCentralityTags[k3050]), "p");
  tCanPart2->AddLegendEntry(1, 0, tRatio3, TString::Format("%s - %s",cPrettyCentralityTags[k0010], cPrettyCentralityTags[k3050]), "p");

  //---------------------------------------------------------------------------------------------------------



  TString tYaxisTitle1 = "C_{1}(k*)/C_{2}(k*)";
  tCanPart1->SetDrawUnityLine(true);
  tCanPart1->DrawAll();
  tCanPart1->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart1->DrawYaxisTitle(tYaxisTitle1,43,20,0.075,0.65);

  TString tYaxisTitle2 = "C_{1}(k*)-C_{2}(k*)";
  tCanPart2->SetDrawUnityLine(true);
  tCanPart2->DrawAll();
  tCanPart2->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart2->DrawYaxisTitle(tYaxisTitle2,43,20,0.075,0.65);


  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)tCanPart1->GetCanvas());
  tReturnArray->Add((TCanvas*)tCanPart2->GetCanvas());
  return tReturnArray;
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
//  TString tResultsDate = "20161027";
//  TString tResultsDate = "20171220_onFlyStatusFalse";
  TString tResultsDate = "20180307";
//  TString tResultsDate = "20171227_LHC10h";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodTrue";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodFalse";

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;

  bool bDrawEPBinningComparison = false;

  bool SaveImages = false;
  TString tSaveFileType = "eps";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";


//-----------------------------------------------------------------------------
  TString tDirectoryBase_cLamcKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamcKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamcKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());

  TString tDirectoryBase_cLamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());
//-----------------------------------------------------------------------------

  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase_cLamcKch,tFileLocationBaseMC_cLamcKch,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
  FitGenerator* tLamKchM = new FitGenerator(tFileLocationBase_cLamcKch,tFileLocationBaseMC_cLamcKch,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
  FitGenerator* tLamK0 = new FitGenerator(tFileLocationBase_cLamK0,tFileLocationBaseMC_cLamK0,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

//-------------------------------------------------------------------------------


  bool tDrawIndividualKchAlso = true;
  TCanvas* tCanCompareLamKchAvgToLamK0 = CompareLamKchAvgToLamK0(tLamKchP, tLamKchM, tLamK0, tDrawIndividualKchAlso);
  if(SaveImages) tCanCompareLamKchAvgToLamK0->SaveAs(tSaveDir + tCanCompareLamKchAvgToLamK0->GetName() + TString::Format(".%s", tSaveFileType.Data()));

  //-----------------
  bool aDrawTHERM = true;
  bool aZoom0010=false;
  TCanvas* tCompareDataToAMPT_LamK0 = CompareDataToAMPT(tLamK0, aDrawTHERM, aZoom0010);
  TCanvas* tCompareDataToAMPT_LamKchP = CompareDataToAMPT(tLamKchP, aDrawTHERM, aZoom0010);
  TCanvas* tCompareDataToAMPT_LamKchM = CompareDataToAMPT(tLamKchM, aDrawTHERM, aZoom0010);
  if(SaveImages)
  {
    tCompareDataToAMPT_LamK0->SaveAs(tSaveDir + tCompareDataToAMPT_LamK0->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    tCompareDataToAMPT_LamKchP->SaveAs(tSaveDir + tCompareDataToAMPT_LamKchP->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    tCompareDataToAMPT_LamKchM->SaveAs(tSaveDir + tCompareDataToAMPT_LamKchM->GetName() + TString::Format(".%s", tSaveFileType.Data()));
  }

  //-----------------
/*
  TObjArray* DrawCfRatiosAndDiffs_Centralities_LamK0 = DrawCfRatiosAndDiffs_Centralities(tLamK0);
  TObjArray* DrawCfRatiosAndDiffs_Centralities_LamKchP = DrawCfRatiosAndDiffs_Centralities(tLamKchP);
  TObjArray* DrawCfRatiosAndDiffs_Centralities_LamKchM = DrawCfRatiosAndDiffs_Centralities(tLamKchM);
  if(SaveImages)
  {
    for(int i=0; i<DrawCfRatiosAndDiffs_Centralities_LamK0->GetEntries(); i++)
    {
      ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<DrawCfRatiosAndDiffs_Centralities_LamKchP->GetEntries(); i++)
    {
      ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamKchP->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamKchP->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<DrawCfRatiosAndDiffs_Centralities_LamKchM->GetEntries(); i++)
    {
      ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamKchM->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)DrawCfRatiosAndDiffs_Centralities_LamKchM->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }
  }

  //-----------------


  TObjArray* tDrawCfRatiosAndDiffs_LamKchM_LamKchP = DrawCfRatiosAndDiffs(tLamKchM, tLamKchP);
  TObjArray* tDrawCfRatiosAndDiffs_LamKchM_LamK0 = DrawCfRatiosAndDiffs(tLamKchM, tLamK0);
  TObjArray* tDrawCfRatiosAndDiffs_LamKchP_LamK0 = DrawCfRatiosAndDiffs(tLamKchP, tLamK0);
  if(SaveImages)
  {
    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchM_LamKchP->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamKchP->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamKchP->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchM_LamK0->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchP_LamK0->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchP_LamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchP_LamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }
  }

  //-----------------

  TObjArray* tDrawCfRatiosAndDiffs_LamKchAvgToLamK0 = DrawCfRatiosAndDiffs_LamKchAvgToLamK0(tLamKchP, tLamKchM, tLamK0);
  if(SaveImages)
  {
    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchAvgToLamK0->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchAvgToLamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchAvgToLamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }
  }
*/
//-------------------------------------------------------------------------------

  if(bDrawEPBinningComparison)
  {

    TString tResultsDate_EPBin8 = "20180313_EPBinning8";

    TString tDirectoryBase_cLamcKch_EPBin8 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_EPBin8.Data());
    TString tFileLocationBase_cLamcKch_EPBin8 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch_EPBin8.Data(),tResultsDate_EPBin8.Data());
    TString tFileLocationBaseMC_cLamcKch_EPBin8 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch_EPBin8.Data(),tResultsDate_EPBin8.Data());

    TString tDirectoryBase_cLamK0_EPBin8 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_EPBin8.Data());
    TString tFileLocationBase_cLamK0_EPBin8 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0_EPBin8.Data(),tResultsDate_EPBin8.Data());
    TString tFileLocationBaseMC_cLamK0_EPBin8 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0_EPBin8.Data(),tResultsDate_EPBin8.Data());
    //-----------------------------------------------------------------------------

    FitGenerator* tLamKchP_EPBin8 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin8,tFileLocationBaseMC_cLamcKch_EPBin8,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamKchM_EPBin8 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin8,tFileLocationBaseMC_cLamcKch_EPBin8,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamK0_EPBin8 = new FitGenerator(tFileLocationBase_cLamK0_EPBin8,tFileLocationBaseMC_cLamK0_EPBin8,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

    //-------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------
    TString tResultsDate_EPBin16 = "20180314_EPBinning16";

    TString tDirectoryBase_cLamcKch_EPBin16 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_EPBin16.Data());
    TString tFileLocationBase_cLamcKch_EPBin16 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch_EPBin16.Data(),tResultsDate_EPBin16.Data());
    TString tFileLocationBaseMC_cLamcKch_EPBin16 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch_EPBin16.Data(),tResultsDate_EPBin16.Data());

    TString tDirectoryBase_cLamK0_EPBin16 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_EPBin16.Data());
    TString tFileLocationBase_cLamK0_EPBin16 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0_EPBin16.Data(),tResultsDate_EPBin16.Data());
    TString tFileLocationBaseMC_cLamK0_EPBin16 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0_EPBin16.Data(),tResultsDate_EPBin16.Data());
    //-----------------------------------------------------------------------------

    FitGenerator* tLamKchP_EPBin16 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin16,tFileLocationBaseMC_cLamcKch_EPBin16,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamKchM_EPBin16 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin16,tFileLocationBaseMC_cLamcKch_EPBin16,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamK0_EPBin16 = new FitGenerator(tFileLocationBase_cLamK0_EPBin16,tFileLocationBaseMC_cLamK0_EPBin16,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

    //-------------------------------------------------------------------------------


    TCanvas* tCompareEPBinning_LamK0 = CompareEPBinning(tLamK0, tLamK0_EPBin8, tLamK0_EPBin16);
    TCanvas* tCompareEPBinning_LamKchP = CompareEPBinning(tLamKchP, tLamKchP_EPBin8, tLamKchP_EPBin16);
    TCanvas* tCompareEPBinning_LamKchM = CompareEPBinning(tLamKchM, tLamKchM_EPBin8, tLamKchM_EPBin16);
    if(SaveImages)
    {
      tCompareEPBinning_LamK0->SaveAs(tSaveDir + tCompareEPBinning_LamK0->GetName() + TString::Format(".%s", tSaveFileType.Data()));
      tCompareEPBinning_LamKchP->SaveAs(tSaveDir + tCompareEPBinning_LamKchP->GetName() + TString::Format(".%s", tSaveFileType.Data()));
      tCompareEPBinning_LamKchM->SaveAs(tSaveDir + tCompareEPBinning_LamKchM->GetName() + TString::Format(".%s", tSaveFileType.Data()));
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
