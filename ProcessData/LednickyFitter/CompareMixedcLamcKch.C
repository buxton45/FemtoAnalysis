/* CompareTwoAnalyses.C */
/* Originally CompareIgnoreOnFlyStatus.C, used to compare different settings of
   ignoreOnFlyStatus of V0s in analyses */

#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;

//________________________________________________________________________________________________________________
int GetMixedAnalysisColor(AnalysisType aAnType1, AnalysisType aAnType2)
{
  int tColor = kYellow;

  if((aAnType1==kLamK0 || aAnType1==kALamK0) && (aAnType2==kLamK0 || aAnType2==kALamK0))
  {
    tColor = kBlack;
  }
  else if((aAnType1==kLamKchP || aAnType1==kALamKchM) && (aAnType2==kLamKchP || aAnType2==kALamKchM))
  {
    tColor = kRed+1;
  }
  else if((aAnType1==kLamKchM || aAnType1==kALamKchP) && (aAnType2==kLamKchM || aAnType2==kALamKchP))
  {
    tColor = kBlue+1;
  }

  //-----------------------------------------

  else if( ((aAnType1==kLamK0 || aAnType1==kALamK0) && (aAnType2==kLamKchP || aAnType2==kALamKchM)) ||
           ((aAnType2==kLamK0 || aAnType2==kALamK0) && (aAnType1==kLamKchP || aAnType1==kALamKchM)) )
  {
    tColor = kRed+2;
  }
  else if( ((aAnType1==kLamK0 || aAnType1==kALamK0) && (aAnType2==kLamKchM || aAnType2==kALamKchP)) ||
           ((aAnType2==kLamK0 || aAnType2==kALamK0) && (aAnType1==kLamKchM || aAnType1==kALamKchP)) )
  {
    tColor = kBlue+2;
  }
  else if( ((aAnType1==kLamKchP || aAnType1==kALamKchM) && (aAnType2==kLamKchM || aAnType2==kALamKchP)) ||
           ((aAnType2==kLamKchP || aAnType2==kALamKchM) && (aAnType1==kLamKchM || aAnType1==kALamKchP)) )
  {
    tColor = kMagenta+1;
  }
  else tColor = kYellow;

  return tColor;
}


//________________________________________________________________________________________________________________
TObjArray* GetNumOrDenCollection(CfHeavy* aCfHeavy, bool aUseNum)
{
  TObjArray* tReturnArray;
  if(aUseNum) tReturnArray = aCfHeavy->GetNumCollection();
  else        tReturnArray = aCfHeavy->GetDenCollection();

  return tReturnArray;
}

//________________________________________________________________________________________________________________
TObjArray* GetNumOrDenCollection(int aAnNum, FitGenerator* aFG, bool aUseNum)
{
  CfHeavy* tCfHeavy = aFG->GetKStarCfHeavy(aAnNum);
  TObjArray* tReturnArray = GetNumOrDenCollection(tCfHeavy, aUseNum);
  return tReturnArray;
}


//________________________________________________________________________________________________________________
CfHeavy* BuildMixedCfHeavy(int aAnNum, FitGenerator* aFG1, FitGenerator* aFG2, bool aUseNum1=true, bool aUseNum2=false)
{
  FitPairAnalysis *tAn1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(aAnNum); 
  FitPairAnalysis *tAn2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(aAnNum);
  //---------------------------------------------------------------------------------------------------------
  AnalysisType tAnType1 = tAn1->GetAnalysisType();
  AnalysisType tAnType2 = tAn2->GetAnalysisType();

  CentralityType tCentType1 = tAn1->GetCentralityType();
  CentralityType tCentType2 = tAn2->GetCentralityType();
  assert(tCentType1==tCentType2);

  const char* const tUseNumTags[2] = {"Den", "Num"};
  TString tCfHeavyName = "CfHeavy";
  TString tAnInfo1 = TString(cAnalysisBaseTags[tAnType1]);
  TString tAnInfo2 = TString(cAnalysisBaseTags[tAnType2]);
  if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(aAnNum)->GetUseNumRotPar2InsteadOfDen() && !aUseNum1) 
  {
    tAnInfo1 += TString("NumRotPar2");
  }
  else tAnInfo1 += TString(tUseNumTags[aUseNum1]);
  if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(aAnNum)->GetUseNumRotPar2InsteadOfDen() && !aUseNum2) 
  {
    tAnInfo2 += TString("NumRotPar2");
  }
  else tAnInfo2 += TString(tUseNumTags[aUseNum2]);

  tCfHeavyName += TString::Format("_%s_%s%s", tAnInfo1.Data(), tAnInfo2.Data(), cCentralityTags[tCentType1]);
  //---------------------------------------------------------------------------------------------------------
  CfHeavy *tCfHeavy1 = tAn1->GetKStarCfHeavy();
  CfHeavy *tCfHeavy2 = tAn2->GetKStarCfHeavy();

  TObjArray *tCollection1 = GetNumOrDenCollection(tCfHeavy1, aUseNum1);
  TObjArray *tCollection2 = GetNumOrDenCollection(tCfHeavy2, aUseNum2);

  assert(tCollection1->GetEntries()==tCollection2->GetEntries());
  assert(tCfHeavy1->GetMinNorm()==tCfHeavy2->GetMinNorm());
  assert(tCfHeavy1->GetMaxNorm()==tCfHeavy2->GetMaxNorm());

  int tNEntries = tCollection1->GetEntries();
  TString tCfLiteName = "";
  CfLite* tTempCfLite;
  vector<CfLite*> tCfLiteVec(0);
  for(int i=0; i<tNEntries; i++)
  {
    tCfLiteName = TString::Format("%s_%d", tCfHeavyName.Data(), i);
    
    tTempCfLite = new CfLite(tCfLiteName, tCfLiteName, 
                             (TH1*)tCollection1->At(i), (TH1*)tCollection2->At(i), 
                             tCfHeavy1->GetMinNorm(), tCfHeavy1->GetMaxNorm());
    tCfLiteVec.push_back(tTempCfLite);
  }

  CfHeavy* tCfHeavy = new CfHeavy(tCfHeavyName, tCfHeavyName, tCfLiteVec, tCfHeavy1->GetMinNorm(), tCfHeavy1->GetMaxNorm());
  return tCfHeavy;
}

//________________________________________________________________________________________________________________
TH1* GetMixedCf(int aAnNum, FitGenerator* aFG1, FitGenerator* aFG2, bool aUseNum1=true, bool aUseNum2=false)
{
  CfHeavy* tCfHeavy = BuildMixedCfHeavy(aAnNum, aFG1, aFG2, aUseNum1, aUseNum2);
  return (TH1*)tCfHeavy->GetHeavyCfClone();
}


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, bool aUseNum1=true, bool aUseNum2=false, bool aZoom=false)
{
  AnalysisType tAnType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();

  CentralityType tCentType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  CentralityType tCentType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  assert(tCentType1==tCentType2);


  const char* const tUseNumTags[2] = {"Den", "Num"};
  TString tCanvasName = "canKStarCfs";
  TString tAnInfo1 = TString(cAnalysisBaseTags[tAnType1]);
  TString tAnInfo2 = TString(cAnalysisBaseTags[tAnType2]);
  if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aUseNum1) 
  {
    tAnInfo1 += TString("NumRotPar2");
  }
  else tAnInfo1 += TString(tUseNumTags[aUseNum1]);
  if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aUseNum2) 
  {
    tAnInfo2 += TString("NumRotPar2");
  }
  else tAnInfo2 += TString(tUseNumTags[aUseNum2]);

  tCanvasName += TString::Format("_%s_%s%s", tAnInfo1.Data(), tAnInfo2.Data(), cCentralityTags[tCentType1]);

  if(aZoom) tCanvasName += TString("Zoom");
  if(aFG1->GetGeneratorType()==kPairwConj) tCanvasName += TString("wConj");

  int tNx=2, tNy=3;
  assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==aFG2->GetFitSharedAnalyses()->GetNFitPairAnalysis());
  if(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==3) tNx=1;
  else assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==6);

  double tXLow = -0.02;
//  double tXHigh = 0.99;
  double tXHigh = aFG1->GetKStarCf(0)->GetXaxis()->GetBinUpEdge(aFG1->GetKStarCf(0)->GetNbinsX());
  if(aZoom) tXHigh = 0.32;

  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = GetMixedAnalysisColor(tAnType1, tAnType2);
  double tMarkerSize = 0.5;

  int tAnalysisNumber=0;
  TH1* tCf;
  TString tTextAnType;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCf = (TH1*)GetMixedCf(tAnalysisNumber, aFG1, aFG2, aUseNum1, aUseNum2);
      tCanPart->AddGraph(i, j, tCf, "", tMarkerStyle1, tMarkerColor1, tMarkerSize);
      //---------------------------------------------------------------------------------------------------------
      tAnType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      tAnType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();

      tCentType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();
      tCentType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();
      assert(tCentType1==tCentType2);
      //---------------------------------------------------------------------------------------------------------
      tAnInfo1 = TString(cAnalysisRootTags[tAnType1]);
      tAnInfo2 = TString(cAnalysisRootTags[tAnType2]);
      if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aUseNum1) 
      {
        tAnInfo1 += TString("NumRotPar2");
      }
      else tAnInfo1 += TString(tUseNumTags[aUseNum1]);
      if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aUseNum2) 
      {
        tAnInfo2 += TString("NumRotPar2");
      }
      else tAnInfo2 += TString(tUseNumTags[aUseNum2]);


      tTextAnType = TString::Format("%s / %s", tAnInfo1.Data(), tAnInfo2.Data());

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType1]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}





//________________________________________________________________________________________________________________
TCanvas* DrawNumDenRatiosPartAn(bool aDrawNum, FitGenerator* aFG1, FitGenerator* aFG2)
{
  AnalysisType tAnType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();

  CentralityType tCentType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  CentralityType tCentType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  assert(tCentType1==tCentType2);

  const char* const tUseNumTags[2] = {"Den", "Num"};
  TString tAnInfo1 = TString(cAnalysisBaseTags[tAnType1]);
  TString tAnInfo2 = TString(cAnalysisBaseTags[tAnType2]);
  if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
  {
    tAnInfo1 += TString("NumRotPar2");
  }
  else tAnInfo1 += TString(tUseNumTags[aDrawNum]);
  if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
  {
    tAnInfo2 += TString("NumRotPar2");
  }
  else tAnInfo2 += TString(tUseNumTags[aDrawNum]);

  TString tCanvasName = TString("canNumDenRatiosPartAn");
  tCanvasName += TString::Format("_%s_%s", tAnInfo1.Data(), tAnInfo2.Data());
  if(aFG1->GetGeneratorType()==kPairwConj) tCanvasName += TString("wConj");


  int tNx=2, tNy=6;
  assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==aFG2->GetFitSharedAnalyses()->GetNFitPairAnalysis());
  if(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==3) tNy=3;
  else assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==6);

  double tXLow = -0.02;
  double tXHigh = 0.98;
  double tYLow = 0.52;
  double tYHigh = 1.02;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
  tCanPart->GetCanvas()->SetCanvasSize(700,1500);

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = GetMixedAnalysisColor(tAnType1, tAnType2);
  double tMarkerSize = 0.5;

  TH1* tRatio;
  TString tRatioName;
  int tAnalysisNumber=0;
  int tPartialAnNumber = 0;
  FitPartialAnalysis *tPartAn1, *tPartAn2;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tPartialAnNumber = i;
      //---------------------------------------------------------------------------------------------------------
      tPartAn1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetFitPartialAnalysis(tPartialAnNumber);
      tPartAn2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetFitPartialAnalysis(tPartialAnNumber);
      //---------------------------------------------------------------------------------------------------------
      if(aDrawNum) 
      {
        tRatioName = TString::Format("NumRatios%s%s_%d_%d", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2], tAnalysisNumber, tPartialAnNumber);
        tRatio = (TH1*)tPartAn1->GetKStarCfLite()->Num()->Clone(tRatioName);
        tRatio->Divide((TH1*)tPartAn2->GetKStarCfLite()->Num());
      }
      else 
      {
        tRatioName = TString::Format("DenRatios%s%s_%d_%d", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2], tAnalysisNumber, tPartialAnNumber);
        tRatio = (TH1*)tPartAn1->GetKStarCfLite()->Den()->Clone(tRatioName);
        tRatio->Divide((TH1*)tPartAn2->GetKStarCfLite()->Den());
      }
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);

      assert(tPartAn1->GetBFieldType() == tPartAn2->GetBFieldType());
      assert(tPartAn1->GetCentralityType() == tPartAn2->GetCentralityType());

      tAnInfo1 = TString(cAnalysisRootTags[tAnType1]);
      tAnInfo2 = TString(cAnalysisRootTags[tAnType2]);
      if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
      {
        tAnInfo1 += TString("NumRotPar2");
      }
      else tAnInfo1 += TString(tUseNumTags[aDrawNum]);
      if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
      {
        tAnInfo2 += TString("NumRotPar2");
      }
      else tAnInfo2 += TString(tUseNumTags[aDrawNum]);
      TString tTextAnType = TString::Format("%s / %s (%s)", tAnInfo1.Data(), tAnInfo2.Data(), cBFieldTags[tPartAn1->GetBFieldType()]);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.6,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[tPartAn1->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      tAnalysisNumber += i;
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  if(aDrawNum) tCanPart->DrawYaxisTitle("Num1/Num2",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("Den1/Den2",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawNumDenRatiosAn(bool aDrawNum, FitGenerator* aFG1, FitGenerator* aFG2, bool aNormalize=false)
{
  AnalysisType tAnType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();

  CentralityType tCentType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  CentralityType tCentType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetCentralityType();
  assert(tCentType1==tCentType2);

  const char* const tUseNumTags[2] = {"Den", "Num"};
  TString tAnInfo1 = TString(cAnalysisBaseTags[tAnType1]);
  TString tAnInfo2 = TString(cAnalysisBaseTags[tAnType2]);
  if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
  {
    tAnInfo1 += TString("NumRotPar2");
  }
  else tAnInfo1 += TString(tUseNumTags[aDrawNum]);
  if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
  {
    tAnInfo2 += TString("NumRotPar2");
  }
  else tAnInfo2 += TString(tUseNumTags[aDrawNum]);

  TString tCanvasName = TString("canNumDenRatios");
  tCanvasName += TString::Format("_%s_%s", tAnInfo1.Data(), tAnInfo2.Data());
  if(aFG1->GetGeneratorType()==kPairwConj) tCanvasName += TString("wConj");

  int tNx=2, tNy=3;
  assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==aFG2->GetFitSharedAnalyses()->GetNFitPairAnalysis());
  if(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==3) tNx=1;
  else assert(aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis()==6);

  double tXLow = -0.02;
  double tXHigh = 0.98;
  double tYLow = 0.52;
  double tYHigh = 1.02;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
//  tCanPart->GetCanvas()->SetCanvasSize(700,1500);

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = GetMixedAnalysisColor(tAnType1, tAnType2);
  double tMarkerSize = 0.5;

  TH1* tRatio;
  TString tRatioName;
  int tAnalysisNumber=0;
  FitPairAnalysis *tAn1, *tAn2;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------
      tAn1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber);
      tAn2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber);
      //---------------------------------------------------------------------------------------------------------
      if(aDrawNum) tRatioName = TString::Format("NumRatios%s%s_%d", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2], tAnalysisNumber);
      else tRatioName = TString::Format("DenRatios%s%s_%d", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2], tAnalysisNumber);

      tRatio = (TH1*)tAn1->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum);
      tRatio->Divide((TH1*)tAn2->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum));
      if(aNormalize) tRatio->Scale(((TH1*)tAn2->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum))->Integral()/((TH1*)tAn1->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum))->Integral());
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);

      assert(tAn1->GetCentralityType() == tAn2->GetCentralityType());

      tAnInfo1 = TString(cAnalysisRootTags[tAnType1]);
      tAnInfo2 = TString(cAnalysisRootTags[tAnType2]);
      if(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
      {
        tAnInfo1 += TString("NumRotPar2");
      }
      else tAnInfo1 += TString(tUseNumTags[aDrawNum]);
      if(aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetUseNumRotPar2InsteadOfDen() && !aDrawNum) 
      {
        tAnInfo2 += TString("NumRotPar2");
      }
      else tAnInfo2 += TString(tUseNumTags[aDrawNum]);
      TString tTextAnType = TString::Format("%s / %s", tAnInfo1.Data(), tAnInfo2.Data());


      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.6,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[tAn1->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  if(aDrawNum) tCanPart->DrawYaxisTitle("Num1/Num2",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("Den1/Den2",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
void Run1vs2(FitGenerator* aFG1, FitGenerator* aFG2, bool aUseNum1, bool aUseNum2, 
             bool aZoom, bool aDrawKStarCfs, bool aDrawNumDenRatiosPartAn, bool aDrawNumDenRatiosAn,
             bool aSave, TString aSaveDir="")
{
  AnalysisType tAnType1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();

  TString tSaveDir = TString::Format("%s%s_%s/", aSaveDir.Data(), cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2]);
  gSystem->mkdir(tSaveDir.Data());

  //-----------------------------------------------------------------------------------------------
  bool tSwitchGenOrder = false;
  if(!aUseNum1 && !aUseNum2) //both false => both denominators, but one or both could be NumRotPar2
  {
    bool tUseNumRotPar2_1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen();
    bool tUseNumRotPar2_2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetUseNumRotPar2InsteadOfDen();
    if(tUseNumRotPar2_1 == tUseNumRotPar2_2)  //both Dens, or both NumRotPar2, either way, order doesn't matter
    {
      tSwitchGenOrder = false;
    }
    else
    {
      if(tUseNumRotPar2_1) //aFG1 is NumRotPar2, and aFG2 is Den
      {
        tSwitchGenOrder = false;
      }
      else if(tUseNumRotPar2_2) //aFG2 is NumRotPar2, and aFG1 is Den
      {
        tSwitchGenOrder = true;
      }
      else assert(0);
    }
  }
  else if(aUseNum1 && aUseNum2)  //if both true  => both numerators, no ordering problem
  {
    tSwitchGenOrder = false;
  }
  else
  {
    if(aUseNum1)  //aFG1 is Num, and aFG2 is Den or NumRotPar2
    {
      tSwitchGenOrder = false;
    }
    else if(aUseNum2)  //aFG2 is Num, and aFG1 is Den or NumRotPar2
    {
      tSwitchGenOrder = true;
    }
    else assert(0);
  }
  //-----------------------------------------------------------------------------------------------

  if(aDrawKStarCfs)
  {
    TCanvas* tCan;
    if(!tSwitchGenOrder) tCan = DrawKStarCfs(aFG1, aFG2, aUseNum1, aUseNum2, aZoom);
    else                 tCan = DrawKStarCfs(aFG2, aFG1, aUseNum2, aUseNum1, aZoom);

    if(aSave) tCan->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tCan->GetName()));
  }

  if(aDrawNumDenRatiosPartAn)
  {
    TCanvas *tRatioNumPartAn, *tRatioDenPartAn;
    if(!tSwitchGenOrder)
    {
      tRatioNumPartAn = DrawNumDenRatiosPartAn(true, aFG1, aFG2);
      tRatioDenPartAn = DrawNumDenRatiosPartAn(false, aFG1, aFG2);
    }
    else
    {
      tRatioNumPartAn = DrawNumDenRatiosPartAn(true, aFG2, aFG1);
      tRatioDenPartAn = DrawNumDenRatiosPartAn(false, aFG2, aFG1);
    }

    if(aSave)
    {
      tRatioNumPartAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioNumPartAn->GetName()));
      tRatioDenPartAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioDenPartAn->GetName()));
    }
  }

  if(aDrawNumDenRatiosAn)
  {
    TCanvas *tRatioNumAn, *tRatioDenAn;
    if(!tSwitchGenOrder)
    {
      tRatioNumAn = DrawNumDenRatiosAn(true, aFG1, aFG2, false);
      tRatioDenAn = DrawNumDenRatiosAn(false, aFG1, aFG2, false);
    }
    else
    {
      tRatioNumAn = DrawNumDenRatiosAn(true, aFG2, aFG1, false);
      tRatioDenAn = DrawNumDenRatiosAn(false, aFG2, aFG1, false);
    }

    if(aSave)
    {
      tRatioNumAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioNumAn->GetName()));
      tRatioDenAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioDenAn->GetName()));
    }
  }


}

//________________________________________________________________________________________________________________
void RunAll1vs2(FitGenerator* aFG1, FitGenerator* aFG2, 
                bool aZoom, bool aDrawKStarCfs, bool aDrawNumDenRatiosPartAn, bool aDrawNumDenRatiosAn,
                bool aSave, TString aSaveDir="")
{
  Run1vs2(aFG1, aFG2, true, true,
          aZoom, aDrawKStarCfs, aDrawNumDenRatiosPartAn, aDrawNumDenRatiosAn,
          aSave, aSaveDir);

  //Only have to run DrawKStarCfs for other permutations
  Run1vs2(aFG1, aFG2, true, false,
          aZoom, true, false, false,
          aSave, aSaveDir);

  Run1vs2(aFG1, aFG2, false, true,
          aZoom, true, false, false,
          aSave, aSaveDir);

  Run1vs2(aFG1, aFG2, false, false,
          aZoom, true, false, false,
          aSave, aSaveDir);

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
  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPair;
  //------------------------------------

  TString tResultsDate;
  tResultsDate = "20180416";

  bool bUseNumRotPar2InsteadOfDen_LamKchP = false;
  bool bUseNumRotPar2InsteadOfDen_ALamKchM = false;

  bool bUseNumRotPar2InsteadOfDen_LamKchM = false;
  bool bUseNumRotPar2InsteadOfDen_ALamKchP = false;

  bool bUseNumRotPar2InsteadOfDen_LamK0 = false;
  bool bUseNumRotPar2InsteadOfDen_ALamK0 = false;


  bool SaveImages = false;
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20180426/Figures/";

  //-----------------------------------------------------------------------------
  TString tDirBase_cLamcKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/", "cLamcKch" , tResultsDate.Data());
  TString tFileLocationBase_cLamcKch = TString::Format("%sResults_%s_%s", tDirBase_cLamcKch.Data(), "cLamcKch", tResultsDate.Data());
  TString tFileLocationBaseMC_cLamcKch = TString::Format("%sResults_%sMC_%s", tDirBase_cLamcKch.Data(), "cLamcKch", tResultsDate.Data());

  TString tDirBase_cLamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/", "cLamK0" ,tResultsDate.Data());
  TString tFileLocationBase_cLamK0 = TString::Format("%sResults_%s_%s", tDirBase_cLamK0.Data(), "cLamK0", tResultsDate.Data());
  TString tFileLocationBaseMC_cLamK0 = TString::Format("%sResults_%sMC_%s", tDirBase_cLamK0.Data(), "cLamK0", tResultsDate.Data());

  //-----------------------------------------------------------------------------

  bool bIncludeLamK0=false;
  bool bZoom = false;
  bool bDrawKStarCfs = true;

  bool bDrawNumDenRatiosPartAn = true;
  bool bDrawNumDenRatiosAn = true;

  bool bDrawNumRotPar2OverBgd = false;
  if(bDrawNumRotPar2OverBgd) assert(!bUseNumRotPar2InsteadOfDen_LamKchP && !bUseNumRotPar2InsteadOfDen_ALamKchM && 
                                    !bUseNumRotPar2InsteadOfDen_LamKchM && !bUseNumRotPar2InsteadOfDen_ALamKchP && 
                                    !bUseNumRotPar2InsteadOfDen_LamK0 && !bUseNumRotPar2InsteadOfDen_ALamK0);

  //-----------------------------------------------------------------------------
  FitGenerator *tLamKchP, *tLamKchM, *tLamK0;
  FitGenerator *tALamKchM, *tALamKchP, *tALamK0;

  tLamKchP = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, 
                                            tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseNumRotPar2InsteadOfDen_LamKchP);
  tLamKchM = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchM, tCentType, 
                                            tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseNumRotPar2InsteadOfDen_LamKchM);
  tLamK0 =   new FitGenerator(tFileLocationBase_cLamK0, tFileLocationBaseMC_cLamK0, kLamK0, tCentType, 
                                            tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseNumRotPar2InsteadOfDen_LamK0);

  vector<FitGenerator*> tFGVec(0);
  if(tGenType == kPairwConj)
  {
    tFGVec.push_back(tLamKchP);
    tFGVec.push_back(tLamKchM);
    if(bIncludeLamK0) tFGVec.push_back(tLamK0);
  }
  else if(tGenType == kPair)
  {
    tALamKchM = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", bUseNumRotPar2InsteadOfDen_ALamKchM);
    tALamKchP = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchM, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", bUseNumRotPar2InsteadOfDen_ALamKchP);
    tALamK0 =   new FitGenerator(tFileLocationBase_cLamK0, tFileLocationBaseMC_cLamK0, kLamK0, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", bUseNumRotPar2InsteadOfDen_ALamK0);
    //--------------------------------
    tFGVec.push_back(tLamKchP);
    tFGVec.push_back(tALamKchM);

    tFGVec.push_back(tLamKchM);
    tFGVec.push_back(tALamKchP);

    if(bIncludeLamK0)
    {
      tFGVec.push_back(tLamK0);
      tFGVec.push_back(tALamK0);
    }
  }
  else assert(0);

  for(int i=0; i<tFGVec.size(); i++)
  {
    for(int j=i+1; j<tFGVec.size(); j++)
    {
      RunAll1vs2(tFGVec[i], tFGVec[j],
                 bZoom, bDrawKStarCfs, bDrawNumDenRatiosPartAn, bDrawNumDenRatiosAn, 
                 SaveImages, tSaveDir);
    }
  }

//-------------------------------------------------------------------------------

  if(bDrawNumRotPar2OverBgd)
  {
    FitGenerator* tLamKchP_RotatePar2 = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, 
                                              tAnRunType, tNPartialAnalysis, tGenType, false, false, "", true);
    FitGenerator* tLamKchM_RotatePar2 = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchM, tCentType, 
                                              tAnRunType, tNPartialAnalysis, tGenType, false, false, "", true);
    FitGenerator* tLamK0_RotatePar2 =   new FitGenerator(tFileLocationBase_cLamK0, tFileLocationBaseMC_cLamK0, kLamK0, tCentType, 
                                              tAnRunType, tNPartialAnalysis, tGenType, false, false, "", true);

    Run1vs2(tLamKchP_RotatePar2, tLamKchP, false, false, 
            bZoom, bDrawKStarCfs, false, false, 
            SaveImages, tSaveDir);

    Run1vs2(tLamKchM_RotatePar2, tLamKchM, false, false, 
            bZoom, bDrawKStarCfs, false, false, 
            SaveImages, tSaveDir);

    Run1vs2(tLamK0_RotatePar2, tLamK0, false, false, 
            bZoom, bDrawKStarCfs, false, false, 
            SaveImages, tSaveDir);

    if(tGenType == kPair)
    {
      FitGenerator* tALamKchM_RotatePar2 = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", true);
      FitGenerator* tALamKchP_RotatePar2 = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchM, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", true);
      FitGenerator* tALamK0_RotatePar2 =   new FitGenerator(tFileLocationBase_cLamK0, tFileLocationBaseMC_cLamK0, kLamK0, tCentType, 
                                               tAnRunType, tNPartialAnalysis, kConjPair, false, false, "", true);

      Run1vs2(tALamKchM_RotatePar2, tALamKchM, false, false, 
              bZoom, bDrawKStarCfs, false, false, 
              SaveImages, tSaveDir);

      Run1vs2(tALamKchP_RotatePar2, tALamKchP, false, false, 
              bZoom, bDrawKStarCfs, false, false, 
              SaveImages, tSaveDir);

      Run1vs2(tALamK0_RotatePar2, tALamK0, false, false, 
              bZoom, bDrawKStarCfs, false, false, 
              SaveImages, tSaveDir);
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
