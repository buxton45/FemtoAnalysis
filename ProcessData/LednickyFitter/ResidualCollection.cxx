/* ResidualCollection.cxx */

#include "ResidualCollection.h"

#ifdef __ROOT__
ClassImp(ResidualCollection)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
ResidualCollection::ResidualCollection(AnalysisType aAnalysisType, td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping) :
  fAnalysisType(aAnalysisType)
{
  BuildStandardCollection(aKStarBinCenters,aTransformMatrices,aTransformStorageMapping);
}



//________________________________________________________________________________________________________________
ResidualCollection::~ResidualCollection()
{}


//________________________________________________________________________________________________________________
void ResidualCollection::BuildStandardCollection(td1dVec &aKStarBinCenters, vector<TH2D*> aTransformMatrices, vector<AnalysisType> aTransformStorageMapping)
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

    if(tNeutral==true) fNeutralCfCollection.emplace_back(aTransformStorageMapping[i], aTransformMatrices[i], aKStarBinCenters);

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








