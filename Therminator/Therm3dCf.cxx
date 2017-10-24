/* Therm3dCf.cxx */

#include "Therm3dCf.h"

#ifdef __ROOT__
ClassImp(Therm3dCf)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
Therm3dCf::Therm3dCf(AnalysisType aAnType, TString aFileLocation, int aRebin) :
  fAnalysisType(aAnType),
  fPartType1(kPDGNull),
  fPartType2(kPDGNull),
  fPartIndex1(-1),
  fPartIndex2(-1),
  fMinNorm(0.32),
  fMaxNorm(0.40),
  fRebin(aRebin),

  fNum3d(nullptr),
  fDen3d(nullptr)

{
  SetPartTypes();
  fPartIndex1 = GetParticleIndexInPidInfo(fPartType1) + 1;
  fPartIndex2 = GetParticleIndexInPidInfo(fPartType2) + 1;


  fNum3d = Get3dHisto(aFileLocation, TString::Format("Num3d%s", cAnalysisBaseTags[fAnalysisType]));
  fDen3d = Get3dHisto(aFileLocation, TString::Format("Den3d%s", cAnalysisBaseTags[fAnalysisType]));
}



//________________________________________________________________________________________________________________
Therm3dCf::~Therm3dCf()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void Therm3dCf::SetPartTypes()
{
  vector<ParticlePDGType> tTempVec = ThermPairAnalysis::GetPartTypes(fAnalysisType);
  fPartType1 = tTempVec[0];
  fPartType2 = tTempVec[1];
}

//________________________________________________________________________________________________________________
void Therm3dCf::SetNormalizationRegion(double aMinNorm, double aMaxNorm) 
{
  //double check the normalization is within range of histogram
  assert(aMinNorm >= fNum3d->GetZaxis()->GetBinLowEdge(1));
  assert(aMaxNorm <= fNum3d->GetZaxis()->GetBinUpEdge(fNum3d->GetNbinsZ()));

  fMinNorm = aMinNorm;
  fMaxNorm = aMaxNorm;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetFull(TH3D* aHist3d, TString aPreName)
{
  TH1D* tReturnHist = aHist3d->ProjectionZ(TString::Format("%sFull%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                           1, aHist3d->GetNbinsX(), 
                                           1, aHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryOnly(TH3D* aHist3d, TString aPreName)
{
  TH1D* tReturnHist = aHist3d->ProjectionZ(TString::Format("%sPrimaryOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                           fPartIndex1, fPartIndex1, 
                                           fPartIndex2, fPartIndex2, "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSecondaryOnly(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetSecondaryOnlyTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
    {
      if(i==fPartIndex1 || j==fPartIndex2)
      {
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSecondaryOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetAtLeastOneSecondaryInPair(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetAtLeastOneSecondaryInPairTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
    {
      if(i==fPartIndex1 && j==fPartIndex2)
      {
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sAtLeastOneSecondaryInPair%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetWithoutSigmaSt(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetWithoutSigmaStTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1==kPDGSigStP || tType1==kPDGASigStM ||
       tType1==kPDGSigStM || tType1==kPDGASigStP ||
       tType1==kPDGSigSt0 || tType1==kPDGASigSt0)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sWithoutSigmaSt%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStOnly(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetSigmaStOnlyTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1 != kPDGSigStP && tType1 != kPDGASigStM &&
       tType1 != kPDGSigStM && tType1 != kPDGASigStP &&
       tType1 != kPDGSigSt0 && tType1 != kPDGASigSt0)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSigmaStOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStPOnly(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetSigmaStPOnlyTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1 != kPDGSigStP && tType1 != kPDGASigStM)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSigmaStPOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStMOnly(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetSigmaStMOnlyTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1 != kPDGSigStM && tType1 != kPDGASigStP)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSigmaStMOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaSt0Only(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetSigmaSt0OnlyTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1 != kPDGSigSt0 && tType1 != kPDGASigSt0)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSigmaSt0Only%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryAndShortDecays(TH3D* aHist3d, TString aPreName)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetPrimaryAndShortDecaysTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1, tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
    {
      tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
      if((i != fPartIndex1 || j != fPartIndex2) && !IncludeAsPrimary(tType1, tType2, 5.0))
      {
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sPrimaryAndShortDecays%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;

}


//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetLongDecays(TH3D* aHist3d, TString aPreName, double aMinDecayLength)
{
  TH3D* tTempHist3d = (TH3D*)aHist3d->Clone(TString::Format("%sGetLongDecaysTempHist3d%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]));

  ParticlePDGType tType1, tType2;
  for(int i=1; i<=tTempHist3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
    {
      tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
      if((i==fPartIndex1 && j==fPartIndex2) || GetParticleDecayLength(tType1) < aMinDecayLength || GetParticleDecayLength(tType2) < aMinDecayLength)
      {
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
          tTempHist3d->SetBinError(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sLongDecays%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY(), "e");
  return tReturnHist;

}









//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetFullCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetFull(fNum3d, "Num");
  TH1D* tDen = GetFull(fDen3d, "Den");

  TString tReturnName = TString::Format("CfFull%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryOnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetPrimaryOnly(fNum3d, "Num");
  TH1D* tDen = GetPrimaryOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSecondaryOnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetSecondaryOnly(fNum3d, "Num");
  TH1D* tDen = GetSecondaryOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetAtLeastOneSecondaryInPairCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetAtLeastOneSecondaryInPair(fNum3d, "Num");
  TH1D* tDen = GetAtLeastOneSecondaryInPair(fDen3d, "Den");

  TString tReturnName = TString::Format("CfAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetWithoutSigmaStCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetWithoutSigmaSt(fNum3d, "Num");
  TH1D* tDen = GetWithoutSigmaSt(fDen3d, "Den");

  TString tReturnName = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStOnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetSigmaStOnly(fNum3d, "Num");
  TH1D* tDen = GetSigmaStOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStPOnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetSigmaStPOnly(fNum3d, "Num");
  TH1D* tDen = GetSigmaStPOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSigmaStPOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}


//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStMOnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetSigmaStMOnly(fNum3d, "Num");
  TH1D* tDen = GetSigmaStMOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSigmaStMOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}


//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaSt0OnlyCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetSigmaSt0Only(fNum3d, "Num");
  TH1D* tDen = GetSigmaSt0Only(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSigmaSt0Only%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}


//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryAndShortDecaysCf(int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetPrimaryAndShortDecays(fNum3d, "Num");
  TH1D* tDen = GetPrimaryAndShortDecays(fDen3d, "Den");

  TString tReturnName = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}


//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetLongDecaysCf(double aMinDecayLength, int aMarkerStyle, int aColor)
{
  TH1D* tNum = GetLongDecays(fNum3d, "Num", aMinDecayLength);
  TH1D* tDen = GetLongDecays(fDen3d, "Den", aMinDecayLength);

  TString tReturnName = TString::Format("CfLongDecays%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  tCf->SetLineColor(aColor);
  tCf->SetMarkerColor(aColor);
  tCf->SetMarkerStyle(aMarkerStyle);
  tCf->SetMarkerSize(0.5);

  return tCf;
}

//________________________________________________________________________________________________________________
void Therm3dCf::DrawAllCfs(TPad* aPad, int aCommonMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------

  int tColorFull = 1;
  int tColorPrimaryOnly = 2;
  int tColorPrimaryAndShortDecays = 3;
  int tColorWithoutSigmaSt = 4;
  int tColorSigmaStOnly = 20;
  int tColorSecondaryOnly = 6;
  int tColorAtLeastOneSecondaryInPair = 28;
//  int tColorLongDecays = 16;

  //---------------------------------------------------------------

  TH1D* tCfFull = GetFullCf(aCommonMarkerStyle, tColorFull);
  TH1D* tCfPrimaryOnly = GetPrimaryOnlyCf(aCommonMarkerStyle, tColorPrimaryOnly);
  TH1D* tCfPrimaryAndShortDecays = GetPrimaryAndShortDecaysCf(aCommonMarkerStyle, tColorPrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaSt = GetWithoutSigmaStCf(aCommonMarkerStyle, tColorWithoutSigmaSt);
  TH1D* tCfSigmaStOnly = GetSigmaStOnlyCf(aCommonMarkerStyle, tColorSigmaStOnly);
  TH1D* tCfSecondaryOnly = GetSecondaryOnlyCf(aCommonMarkerStyle, tColorSecondaryOnly);
  TH1D* tCfAtLeastOneSecondaryInPair = GetAtLeastOneSecondaryInPairCf(aCommonMarkerStyle, tColorAtLeastOneSecondaryInPair);
//  TH1D* tCfLongDecays = GetLongDecaysCf(1000, aCommonMarkerStyle, tColorLongDecays);

  tCfFull->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfFull->GetYaxis()->SetTitle("C(k*)");

//  tCfFull->GetXaxis()->SetRangeUser(0.,0.329);
  tCfFull->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfFull->Draw();
  tCfPrimaryOnly->Draw("same");
  tCfPrimaryAndShortDecays->Draw("same");
  tCfWithoutSigmaSt->Draw("same");
  tCfSigmaStOnly->Draw("same");
  tCfSecondaryOnly->Draw("same");
  tCfAtLeastOneSecondaryInPair->Draw("same");
//  tCfLongDecays->Draw("same");
  tCfFull->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[GetAnalysisType()]));

  tLeg->AddEntry(tCfFull, "Full");
  tLeg->AddEntry(tCfPrimaryOnly, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecays, "Primary and short decays");
  tLeg->AddEntry(tCfWithoutSigmaSt, "w/o #Sigma*");
  tLeg->AddEntry(tCfAtLeastOneSecondaryInPair, "At Least One Secondary");
  tLeg->AddEntry(tCfSecondaryOnly, "Secondary Only");
  tLeg->AddEntry(tCfSigmaStOnly, "#Sigma* Only");
//  tLeg->AddEntry(tCfLongDecays, "Long Decays");

  tLeg->Draw();

  TLine* tLine = new TLine(tCfFull->GetXaxis()->GetBinLowEdge(1), 1, tCfFull->GetXaxis()->GetBinUpEdge(tCfFull->GetNbinsX()), 1);
  tLine->SetLineColor(14);
  tLine->Draw();
}


//________________________________________________________________________________________________________________
void Therm3dCf::DrawAllSigmaStFlavors(TPad* aPad)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------

  int tColorSigmaStOnly = 20;
  int tColorSigmaStPOnly = 20;
  int tColorSigmaStMOnly = 20;
  int tColorSigmaSt0Only = 20;

  int tMarkerStyleSigmaStOnly = 20;
  int tMarkerStyleSigmaStPOnly = 24;
  int tMarkerStyleSigmaStMOnly = 25;
  int tMarkerStyleSigmaSt0Only = 26;

  //---------------------------------------------------------------

  TH1D* tCfSigmaStOnly = GetSigmaStOnlyCf(tMarkerStyleSigmaStOnly, tColorSigmaStOnly);
  TH1D* tCfSigmaStPOnly = GetSigmaStPOnlyCf(tMarkerStyleSigmaStPOnly, tColorSigmaStPOnly);
  TH1D* tCfSigmaStMOnly = GetSigmaStMOnlyCf(tMarkerStyleSigmaStMOnly, tColorSigmaStMOnly);
  TH1D* tCfSigmaSt0Only = GetSigmaSt0OnlyCf(tMarkerStyleSigmaSt0Only, tColorSigmaSt0Only);


  tCfSigmaStOnly->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfSigmaStOnly->GetYaxis()->SetTitle("C(k*)");

//  tCfSigmaStOnly->GetXaxis()->SetRangeUser(0.,0.329);
  tCfSigmaStOnly->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfSigmaStOnly->Draw();
  tCfSigmaStPOnly->Draw("same");
  tCfSigmaStMOnly->Draw("same");
  tCfSigmaSt0Only->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[GetAnalysisType()]));

  tLeg->AddEntry(tCfSigmaStOnly, "#Sigma* Only (Total)");
  tLeg->AddEntry(tCfSigmaStPOnly, "#Sigma*^{+} (#bar{#Sigma*}^{-}) Only");
  tLeg->AddEntry(tCfSigmaStMOnly, "#Sigma*^{-} (#bar{#Sigma*}^{+}) Only");
  tLeg->AddEntry(tCfSigmaSt0Only, "#Sigma*^{0} (#bar{#Sigma*}^{0}) Only");

  tLeg->Draw();

  TLine* tLine = new TLine(tCfSigmaStOnly->GetXaxis()->GetBinLowEdge(1), 1, tCfSigmaStOnly->GetXaxis()->GetBinUpEdge(tCfSigmaStOnly->GetNbinsX()), 1);
  tLine->SetLineColor(14);
  tLine->Draw();
}







