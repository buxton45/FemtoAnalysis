#include "FitGenerator.h"
class FitGenerator;

//________________________________________________________________________________________________________________
void SetStyleAndColor(TH1* aHist, int aMarkerStyle, int aColor)
{
  aHist->SetLineColor(aColor);
  aHist->SetMarkerColor(aColor);
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerSize(0.5);
}

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
TH1* GetTHERMCf(TString aFileName, AnalysisType aAnType, int aImpactParam=8, bool aCombineConj = true, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
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
  TString tFileLocation = TString::Format("%s%s", tDirectory.Data(), aFileName.Data());
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

  TH1* tReturnHist;
  if(!aCombineConj) tReturnHist = tCfLite1->Cf();
  else
  {
    vector<CfLite*> tCfLiteVec {tCfLite1, tCfLite2};
    CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                                    TString::Format("CfHeavy_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                                    tCfLiteVec, aMinNorm, aMaxNorm);
    tReturnHist = tCfHeavy->GetHeavyCf();
  }

  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* GetTHERMCf(AnalysisType aAnType, int aImpactParam=8, bool aCombineConj = true, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  return GetTHERMCf(tFileName, aAnType, aImpactParam, aCombineConj, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor);
}

//________________________________________________________________________________________________________________
TH1* GetCombinedTHERMCfs(TString aFileName, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj = true, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
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
  vector<int> tImpactParams;
  if     (aCentType == k0010) tImpactParams = vector<int>{3};
  else if(aCentType == k1030) tImpactParams = vector<int>{5, 7};
  else if(aCentType == k3050) tImpactParams = vector<int>{8, 9};
  else assert(0);
  //--------------------------------
  TString tDirectory, tFileLocation;
  vector<CfLite*> tCfLiteVec;
  for(unsigned int iIP=0; iIP < tImpactParams.size(); iIP++)
  {
    tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParams[iIP]);
    tFileLocation = TString::Format("%s%s", tDirectory.Data(), aFileName.Data());
    //--------------------------------
    TH1D* tNum1 = Get1dTHERMHist(tFileLocation, TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]));
    TH1D* tDen1 = Get1dTHERMHist(tFileLocation, TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]));
    CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  TString::Format("CfLite_%s_%d", cAnalysisBaseTags[aAnType], tImpactParams[iIP]), 
                                  tNum1, tDen1, aMinNorm, aMaxNorm);
    tCfLite1->Rebin(aRebin);
    tCfLiteVec.push_back(tCfLite1);

    if(aCombineConj)
    {
      TH1D* tNum2 = Get1dTHERMHist(tFileLocation, TString::Format("NumFull%s", cAnalysisBaseTags[tConjAnType]));
      TH1D* tDen2 = Get1dTHERMHist(tFileLocation, TString::Format("DenFull%s", cAnalysisBaseTags[tConjAnType]));
      CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s_%d", cAnalysisBaseTags[tConjAnType], tImpactParams[iIP]), 
                                    TString::Format("CfLite_%s_%d", cAnalysisBaseTags[tConjAnType], tImpactParams[iIP]), 
                                    tNum2, tDen2, aMinNorm, aMaxNorm);
      tCfLite2->Rebin(aRebin);
      tCfLiteVec.push_back(tCfLite2);
    }
  }

  CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfHeavy_%s_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType], cCentralityTags[aCentType]), 
                                  TString::Format("CfHeavy_%s_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType], cCentralityTags[aCentType]), 
                                  tCfLiteVec, aMinNorm, aMaxNorm);

  TH1* tReturnHist = tCfHeavy->GetHeavyCf();
  SetStyleAndColor(tReturnHist, aMarkerStyle, aColor);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* GetCombinedTHERMCfs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConj = true, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1)
{
  TString tFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  return GetCombinedTHERMCfs(tFileName, aAnType, aCentType, aCombineConj, aRebin, aMinNorm, aMaxNorm, aMarkerStyle, aColor);
}
