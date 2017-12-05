/* ResidualCollection.cxx */

#include "ResidualCollection.h"

#ifdef __ROOT__
ClassImp(ResidualCollection)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
ResidualCollection::ResidualCollection(AnalysisType aAnalysisType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType) :
  fAnalysisType(aAnalysisType),
  fIncludeResidualsType(aIncludeResidualsType),
  fResPrimMaxDecayType(aResPrimMaxDecayType)
{
  BuildStandardCollection(aKStarBinCenters,aTransformMatrices,aTransformStorageMapping,aCentType);
}



//________________________________________________________________________________________________________________
ResidualCollection::~ResidualCollection()
{}


//________________________________________________________________________________________________________________
void ResidualCollection::BuildStandardCollection(td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping, CentralityType aCentType)
{
  if(aTransformStorageMapping.size() != 11)
  {
    int tResponse;
    cout << "aTransformStorageMapping.size() != 11, is this alright?" << endl;
    cout << "(0) = No \t (1) = Yes" << endl;
    cin >> tResponse;
    assert(tResponse);
  }

  assert(aTransformMatrices.size() == aTransformStorageMapping.size());
  bool tNeutral = true;
  for(unsigned int i=0; i<aTransformMatrices.size(); i++)
  {
    tNeutral = true;
    if(aTransformStorageMapping[i] == kResXiCKchP || aTransformStorageMapping[i] == kResAXiCKchM || 
       aTransformStorageMapping[i] == kResXiCKchM || aTransformStorageMapping[i] == kResAXiCKchP || 

       aTransformStorageMapping[i] == kResOmegaKchP || aTransformStorageMapping[i] == kResAOmegaKchM || 
       aTransformStorageMapping[i] == kResOmegaKchM || aTransformStorageMapping[i] == kResAOmegaKchP || 

       aTransformStorageMapping[i] == kResSigStPKchP || aTransformStorageMapping[i] == kResASigStMKchM || 
       aTransformStorageMapping[i] == kResSigStPKchM || aTransformStorageMapping[i] == kResASigStMKchP || 

       aTransformStorageMapping[i] == kResSigStMKchP || aTransformStorageMapping[i] == kResASigStPKchM || 
       aTransformStorageMapping[i] == kResSigStMKchM || aTransformStorageMapping[i] == kResASigStPKchP) tNeutral = false;

    if(tNeutral==true) fNeutralCfCollection.emplace_back(aTransformStorageMapping[i], fIncludeResidualsType, fResPrimMaxDecayType, aTransformMatrices[i], aKStarBinCenters);
    else fChargedCfCollection.emplace_back(aTransformStorageMapping[i], fIncludeResidualsType, fResPrimMaxDecayType, aTransformMatrices[i], aKStarBinCenters, aCentType);
  }

}

//________________________________________________________________________________________________________________
void ResidualCollection::SetChargedResidualsType(TString aFileDirectory, ChargedResidualsType aChargedResidualsType)
{
  AnalysisType tResType;
  double tRadiusFactor = 1.;
  bool tUseCoulombOnlyInterpCfsForChargedResiduals = true;
  bool tUseCoulombOnlyInterpCfsForXiKResiduals = false;

  if(aChargedResidualsType==kUseXiDataForAll)
  {
    tUseCoulombOnlyInterpCfsForChargedResiduals = false;
    tUseCoulombOnlyInterpCfsForXiKResiduals = false;
  }
  else if(aChargedResidualsType==kUseXiDataAndCoulombOnlyInterp)
  {
    tUseCoulombOnlyInterpCfsForChargedResiduals = true;
    tUseCoulombOnlyInterpCfsForXiKResiduals = false;
  }
  else if(aChargedResidualsType==kUseCoulombOnlyInterpForAll)
  {
    tUseCoulombOnlyInterpCfsForChargedResiduals = true;
    tUseCoulombOnlyInterpCfsForXiKResiduals = true;
  }
  else assert(0);

  for(unsigned int i=0; i<fChargedCfCollection.size(); i++) 
  {
    tResType = fChargedCfCollection[i].GetResidualType();
    if(tUseCoulombOnlyInterpCfsForChargedResiduals)
    {
      if(tResType==kResSigStPKchP || tResType==kResASigStMKchM ||
         tResType==kResSigStPKchM || tResType==kResASigStMKchP ||
         tResType==kResSigStMKchP || tResType==kResASigStPKchM ||
         tResType==kResSigStMKchM || tResType==kResASigStPKchP) fChargedCfCollection[i].LoadCoulombOnlyInterpCfs(aFileDirectory, tUseCoulombOnlyInterpCfsForChargedResiduals, tRadiusFactor);
    }
    if(tUseCoulombOnlyInterpCfsForXiKResiduals)
    {
      if(tResType==kXiKchP || tResType==kAXiKchM ||
         tResType==kXiKchM || tResType==kAXiKchP ||
         tResType==kResXiCKchP || tResType==kResAXiCKchM ||
         tResType==kResXiCKchM || tResType==kResAXiCKchP) fChargedCfCollection[i].LoadCoulombOnlyInterpCfs(aFileDirectory, tUseCoulombOnlyInterpCfsForXiKResiduals, tRadiusFactor);
    }
  }
}


//________________________________________________________________________________________________________________
void ResidualCollection::SetRadiusFactorForSigStResiduals(double aFactor)
{
  AnalysisType tResType;

  for(unsigned int i=0; i<fChargedCfCollection.size(); i++) 
  {
    tResType = fChargedCfCollection[i].GetResidualType();
    if(tResType==kResSigStPKchP || tResType==kResASigStMKchM || tResType==kResSigStPKchM || tResType==kResASigStMKchP || 
       tResType==kResSigStMKchP || tResType==kResASigStPKchM || tResType==kResSigStMKchM || tResType==kResASigStPKchP) fChargedCfCollection[i].SetRadiusFactor(aFactor);
  }

  for(unsigned int i=0; i<fNeutralCfCollection.size(); i++) 
  {
    tResType = fNeutralCfCollection[i].GetResidualType();
    if(tResType==kResSigSt0KchP || tResType==kResASigSt0KchM || tResType==kResSigSt0KchM || tResType==kResASigSt0KchP || 
       tResType==kResSigStPK0 || tResType==kResASigStMK0 || 
       tResType==kResSigStMK0 || tResType==kResASigStPK0 || 
       tResType==kResSigSt0K0 || tResType==kResASigSt0K0) fNeutralCfCollection[i].SetRadiusFactor(aFactor);
  }
}


//________________________________________________________________________________________________________________
int ResidualCollection::GetNeutralIndex(AnalysisType aResidualType)
{
  int tIndex = -1;
  for(int i=0; i<(int)fNeutralCfCollection.size(); i++)
  {
    if(aResidualType == fNeutralCfCollection[i].GetResidualType()) tIndex = i;
  }
  if(tIndex == -1)
  {
    cout << "In ResidualCollection::GetNeutralIndex, tIndex = -1 for aResidualType = " << aResidualType << endl;
    cout << "CRASH IMMINENT!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }
  assert(tIndex > -1);
  return tIndex;
}


//________________________________________________________________________________________________________________
td1dVec ResidualCollection::GetNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams)
{
  int tIndex = GetNeutralIndex(aResidualType);
  return fNeutralCfCollection[tIndex].GetNeutralResidualCorrelation(aParentCfParams);
}


//________________________________________________________________________________________________________________
td1dVec ResidualCollection::GetTransformedNeutralResidualCorrelation(AnalysisType aResidualType, double *aParentCfParams)
{
  int tIndex = GetNeutralIndex(aResidualType);
  return fNeutralCfCollection[tIndex].GetTransformedNeutralResidualCorrelation(aParentCfParams);
}

//________________________________________________________________________________________________________________
td1dVec ResidualCollection::CombinePrimaryWithResiduals(double *aCfParams, td1dVec &aPrimaryCf)
{
  td2dVec tCfs;
  tCfs.push_back(aPrimaryCf);
  for(unsigned int iResCf=0; iResCf<fNeutralCfCollection.size(); iResCf++) tCfs.push_back(fNeutralCfCollection[iResCf].GetContributionToFitCf(aCfParams));
  for(unsigned int iResCf=0; iResCf<fChargedCfCollection.size(); iResCf++) tCfs.push_back(fChargedCfCollection[iResCf].GetContributionToFitCf(aCfParams[0], aCfParams[1]));
  for(unsigned int i=1; i<tCfs.size(); i++) assert(tCfs[i-1].size()==tCfs[i].size());

  td1dVec tReturnCf(tCfs[0].size(), 0.);
  for(unsigned int iBin=0; iBin<tReturnCf.size(); iBin++)
  {
    for(unsigned int iCf=0; iCf<tCfs.size(); iCf++)
    {
      if( (iCf==0 && tCfs[iCf][iBin] > 0.) || (iCf>0 && tCfs[iCf][iBin] > -1.))  //TODO maybe change this to assert to find any cases violating this
      {
        tReturnCf[iBin] += tCfs[iCf][iBin];
      }
    }
  }
  return tReturnCf;
}


//________________________________________________________________________________________________________________
void ResidualCollection::SetUsemTScalingOfRadii(double aPower)
{
  for(unsigned int i=0; i<fChargedCfCollection.size(); i++) fChargedCfCollection[i].SetUsemTScalingOfRadii(fAnalysisType, aPower);
  for(unsigned int i=0; i<fNeutralCfCollection.size(); i++) fNeutralCfCollection[i].SetUsemTScalingOfRadii(fAnalysisType, aPower);
}



