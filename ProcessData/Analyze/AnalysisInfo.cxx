/* AnalysisInfo.cxx */

#include "AnalysisInfo.h"

#ifdef __ROOT__
ClassImp(AnalysisInfo)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
AnalysisInfo::AnalysisInfo(AnalysisType aAnalysisType) :
  fAnalysisType(aAnalysisType)
{
//------------ cLamK0 ----------------------------------
  if(fAnalysisType == kLamK0) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kK0;
  }
  else if(fAnalysisType == kALamK0) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kK0;
  }

//------------ cLamcKch --------------------------------
  else if(fAnalysisType == kLamKchP) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kALamKchP) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kLamKchM) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kALamKchM) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kKchM;
  }

//------------ cXicKch ---------------------------------
  else if(fAnalysisType == kXiKchP) 
  {
    fParticleTypes[0] = kXi; fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kAXiKchP) 
  {
    fParticleTypes[0] = kAXi; fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kXiKchM) 
  {
    fParticleTypes[0] = kXi; fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kAXiKchM) 
  {
    fParticleTypes[0] = kAXi; fParticleTypes[1] = kKchM;
  }


//------------ cLamcLam ---------------------------------
  else if(fAnalysisType == kLamLam) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kLam;
  }
  else if(fAnalysisType == kALamALam) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kALam;
  }
  else if(fAnalysisType == kLamALam) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kALam;
  }


//------------ cLamcPi
  else if(fAnalysisType == kLamPiP) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kPiP;
  }
  else if(fAnalysisType == kALamPiP) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kPiP;
  }
  else if(fAnalysisType == kLamPiM) 
  {
    fParticleTypes[0] = kLam; fParticleTypes[1] = kPiM;
  }
  else if(fAnalysisType == kALamPiM) 
  {
    fParticleTypes[0] = kALam; fParticleTypes[1] = kPiM;
  }

}


//________________________________________________________________________________________________________________
AnalysisInfo::~AnalysisInfo()
{
  /* no-op */
}






