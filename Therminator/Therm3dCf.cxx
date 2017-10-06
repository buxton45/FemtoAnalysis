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
TH1D* Therm3dCf::GetFull(TH3D* aHist3d, TString aPreName)
{
  TH1D* tReturnHist = aHist3d->ProjectionZ(TString::Format("%sFull%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                           1, aHist3d->GetNbinsX(), 
                                           1, aHist3d->GetNbinsY());
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryOnly(TH3D* aHist3d, TString aPreName)
{
  TH1D* tReturnHist = aHist3d->ProjectionZ(TString::Format("%sPrimaryOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                           fPartIndex1, fPartIndex1, 
                                           fPartIndex2, fPartIndex2);
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
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSecondaryOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY());
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
      if(i!=fPartIndex1 || j!=fPartIndex2)
      {
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sAtLeastOneSecondaryInPair%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY());
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
       tType1==kPDGSigSt0 || tType1==kPDGSigSt0)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sWithoutSigmaSt%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY());
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
       tType1 != kPDGSigSt0 && tType1 != kPDGSigSt0)
    {
      for(int j=1; j<=tTempHist3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tTempHist3d->GetNbinsZ(); k++)
        {
          tTempHist3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sSigmaStOnly%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY());
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
        }
      }
    }
  }

  TH1D* tReturnHist = tTempHist3d->ProjectionZ(TString::Format("%sPrimaryAndShortDecays%s", aPreName.Data(), cAnalysisBaseTags[fAnalysisType]), 
                                               1, tTempHist3d->GetNbinsX(), 
                                               1, tTempHist3d->GetNbinsY());
  return tReturnHist;

}









//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetFullCf()
{
  TH1D* tNum = GetFull(fNum3d, "Num");
  TH1D* tDen = GetFull(fDen3d, "Den");

  TString tReturnName = TString::Format("CfFull%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryOnlyCf()
{
  TH1D* tNum = GetPrimaryOnly(fNum3d, "Num");
  TH1D* tDen = GetPrimaryOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSecondaryOnlyCf()
{
  TH1D* tNum = GetSecondaryOnly(fNum3d, "Num");
  TH1D* tDen = GetSecondaryOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetAtLeastOneSecondaryInPairCf()
{
  TH1D* tNum = GetAtLeastOneSecondaryInPair(fNum3d, "Num");
  TH1D* tDen = GetAtLeastOneSecondaryInPair(fDen3d, "Den");

  TString tReturnName = TString::Format("CfAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetWithoutSigmaStCf()
{
  TH1D* tNum = GetWithoutSigmaSt(fNum3d, "Num");
  TH1D* tDen = GetWithoutSigmaSt(fDen3d, "Den");

  TString tReturnName = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetSigmaStOnlyCf()
{
  TH1D* tNum = GetSigmaStOnly(fNum3d, "Num");
  TH1D* tDen = GetSigmaStOnly(fDen3d, "Den");

  TString tReturnName = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* Therm3dCf::GetPrimaryAndShortDecaysCf()
{
  TH1D* tNum = GetPrimaryAndShortDecays(fNum3d, "Num");
  TH1D* tDen = GetPrimaryAndShortDecays(fDen3d, "Den");

  TString tReturnName = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName, fMinNorm, fMaxNorm, fRebin);

  return tCf;
}





