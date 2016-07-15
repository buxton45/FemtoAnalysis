///////////////////////////////////////////////////////////////////////////
// Cf2dHeavy:                                                             //
///////////////////////////////////////////////////////////////////////////


#include "Cf2dHeavy.h"

#ifdef __ROOT__
ClassImp(Cf2dHeavy)
#endif



//________________________________________________________________________________________________________________





//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________






//________________________________________________________________________________________________________________
Cf2dHeavy::Cf2dHeavy(TString aDaughterHeavyCfsBaseName, vector<Cf2dLite*> &aCf2dLiteCollection, double aMinNorm, double aMaxNorm) :
  fCf2dLiteCollection(aCf2dLiteCollection),
  fCollectionSize(aCf2dLiteCollection.size()),
  fNDaughterHeavyCfs(0),
  fDaughterHeavyCfs(0),
  fDaughterHeavyCfsBaseName(aDaughterHeavyCfsBaseName),
  fMinNorm(aMinNorm),
  fMaxNorm(aMaxNorm)

{

  for(int i=1; i<fCollectionSize; i++) {assert(fCf2dLiteCollection[i-1]->GetNDaughterCfs() == fCf2dLiteCollection[i]->GetNDaughterCfs());}

  fNDaughterHeavyCfs = fCf2dLiteCollection[0]->GetNDaughterCfs();

  CombineCfs();
}



//________________________________________________________________________________________________________________
Cf2dHeavy::~Cf2dHeavy()
{

}



//________________________________________________________________________________________________________________
void Cf2dHeavy::CombineCfs(int aRebinFactor)
{
  fDaughterHeavyCfs.clear();
  vector<CfLite*> tTempVecCfLite;

  for(int iDaughter=0; iDaughter<fNDaughterHeavyCfs; iDaughter++)
  {
    tTempVecCfLite.clear();
    for(int iColl=0; iColl<fCollectionSize; iColl++)
    {
      //build/refresh results
      fCf2dLiteCollection[iColl]->DoProjection(aRebinFactor);
      tTempVecCfLite.push_back(fCf2dLiteCollection[iColl]->GetDaughterCf(iDaughter));
    }

    TString tName = fDaughterHeavyCfsBaseName;
    tName+=iDaughter;
    CfHeavy *tTempCfHeavy = new CfHeavy(tName,tName,tTempVecCfLite,fMinNorm,fMaxNorm);

    fDaughterHeavyCfs.push_back(tTempCfHeavy);

  }

}


//________________________________________________________________________________________________________________
void Cf2dHeavy::Rebin(int aRebinFactor)
{
  CombineCfs(aRebinFactor);
}


//________________________________________________________________________________________________________________
CfHeavy* Cf2dHeavy::GetDaughterHeavyCf(int aDaughterHeavyCf)
{
  assert(aDaughterHeavyCf <= fNDaughterHeavyCfs);
  return fDaughterHeavyCfs[aDaughterHeavyCf];
}





