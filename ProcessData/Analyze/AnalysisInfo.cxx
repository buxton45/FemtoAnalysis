/* AnalysisInfo.cxx */

#include "AnalysisInfo.h"

#ifdef __ROOT__
ClassImp(AnalysisInfo)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
AnalysisInfo::AnalysisInfo() :
  fAnalysisType(kLamK0),
  fConjAnalysisType(kLamK0),
  fIsResidual(false),
  fParticleTypes(0),
  fParticlePDGTypes(0),
  fDaughterPairTypes(0),
  fCoulombType(kNeutral),
  fBohrRadius(1000000000)
{
  *this = AnalysisInfo(fAnalysisType);
}



//________________________________________________________________________________________________________________
AnalysisInfo::AnalysisInfo(AnalysisType aAnalysisType) :
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kLamK0),
  fIsResidual(false),
  fParticleTypes(0),
  fParticlePDGTypes(0),
  fDaughterPairTypes(0),
  fCoulombType(kNeutral),
  fBohrRadius(1000000000)
{
//------------ cLamK0 ----------------------------------
  if(fAnalysisType == kLamK0) 
  {
    fConjAnalysisType = kALamK0;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kK0;
  }
  else if(fAnalysisType == kALamK0) 
  {
    fConjAnalysisType = kLamK0;

    fParticleTypes[0] = kALam; 
    fParticleTypes[1] = kK0;
  }

//------------ cLamcKch --------------------------------
  else if(fAnalysisType == kLamKchP) 
  {
    fConjAnalysisType = kALamKchM;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kALamKchM) 
  {
    fConjAnalysisType = kLamKchP;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kLamKchM) 
  {
    fConjAnalysisType = kALamKchP;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kALamKchP) 
  {
    fConjAnalysisType = kLamKchM;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kKchP;
  }

//------------ cXicKch ---------------------------------
  else if(fAnalysisType == kXiKchP) 
  {
    fConjAnalysisType = kAXiKchM;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kAXiKchM) 
  {
    fConjAnalysisType = kXiKchP;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kXiKchM) 
  {
    fConjAnalysisType = kAXiKchP;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kAXiKchP) 
  {
    fConjAnalysisType = kXiKchM;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kKchP;
  }


//------------ cLamcLam ---------------------------------
  else if(fAnalysisType == kLamLam) 
  {
    fConjAnalysisType = kALamALam;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kLam;
  }
  else if(fAnalysisType == kALamALam) 
  {
    fConjAnalysisType = kLamLam;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kALam;
  }
  else if(fAnalysisType == kLamALam) 
  {
    fConjAnalysisType = kLamALam;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kALam;
  }


//------------ cLamcPi ----------------------------------
  else if(fAnalysisType == kLamPiP) 
  {
    fConjAnalysisType = kALamPiM;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kPiP;
  }
  else if(fAnalysisType == kALamPiM) 
  {
    fConjAnalysisType = kLamPiP;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kPiM;
  }
  else if(fAnalysisType == kLamPiM) 
  {
    fConjAnalysisType = kALamPiP;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kPiM;
  }
  else if(fAnalysisType == kALamPiP) 
  {
    fConjAnalysisType = kLamPiM;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kPiP;
  }

//_________________________________________________________________
//                        RESIDUALS                                
//_________________________________________________________________

//------------ RescSig0cKch ----------------------------------
  else if(fAnalysisType == kResSig0KchP) 
  {
    fConjAnalysisType = kResASig0KchM;

    fParticleTypes[0] = kSig0;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResASig0KchM) 
  {
    fConjAnalysisType = kResSig0KchP;

    fParticleTypes[0] = kASig0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResSig0KchM) 
  {
    fConjAnalysisType = kResASig0KchP;

    fParticleTypes[0] = kSig0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResASig0KchP) 
  {
    fConjAnalysisType = kResSig0KchM;

    fParticleTypes[0] = kASig0;
    fParticleTypes[1] = kKchP;
  }

//------------ RescXi0cKch ----------------------------------
  else if(fAnalysisType == kResXi0KchP) 
  {
    fConjAnalysisType = kResAXi0KchM;

    fParticleTypes[0] = kXi0;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResAXi0KchM) 
  {
    fConjAnalysisType = kResXi0KchP;

    fParticleTypes[0] = kAXi0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResXi0KchM) 
  {
    fConjAnalysisType = kResAXi0KchP;

    fParticleTypes[0] = kXi0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResAXi0KchP) 
  {
    fConjAnalysisType = kResXi0KchM;

    fParticleTypes[0] = kAXi0;
    fParticleTypes[1] = kKchP;
  }

//------------ RescXiCcKch ----------------------------------
  else if(fAnalysisType == kResXiCKchP) 
  {
    fConjAnalysisType = kResAXiCKchM;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResAXiCKchM) 
  {
    fConjAnalysisType = kResXiCKchP;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResXiCKchM) 
  {
    fConjAnalysisType = kResAXiCKchP;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResAXiCKchP) 
  {
    fConjAnalysisType = kResXiCKchM;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kKchP;
  }

//------------ RescOmegacKch ----------------------------------
  else if(fAnalysisType == kResOmegaKchP) 
  {
    fConjAnalysisType = kResAOmegaKchM;

    fParticleTypes[0] = kOmega;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResAOmegaKchM) 
  {
    fConjAnalysisType = kResOmegaKchP;

    fParticleTypes[0] = kAOmega;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResOmegaKchM) 
  {
    fConjAnalysisType = kResAOmegaKchP;

    fParticleTypes[0] = kOmega;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResAOmegaKchP) 
  {
    fConjAnalysisType = kResOmegaKchM;

    fParticleTypes[0] = kAOmega;
    fParticleTypes[1] = kKchP;
  }

//------------ RescSigStPcKch ----------------------------------
  else if(fAnalysisType == kResSigStPKchP) 
  {
    fConjAnalysisType = kResASigStMKchM;

    fParticleTypes[0] = kSigStP;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResASigStMKchM) 
  {
    fConjAnalysisType = kResSigStPKchP;

    fParticleTypes[0] = kASigStM;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResSigStPKchM) 
  {
    fConjAnalysisType = kResASigStMKchP;

    fParticleTypes[0] = kSigStP;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResASigStMKchP) 
  {
    fConjAnalysisType = kResSigStPKchM;

    fParticleTypes[0] = kASigStM;
    fParticleTypes[1] = kKchP;
  }

//------------ RescSigStMcKch ----------------------------------
  else if(fAnalysisType == kResSigStMKchP) 
  {
    fConjAnalysisType = kResASigStPKchM;

    fParticleTypes[0] = kSigStM;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResASigStPKchM) 
  {
    fConjAnalysisType = kResSigStMKchP;

    fParticleTypes[0] = kASigStP;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResSigStMKchM) 
  {
    fConjAnalysisType = kResASigStPKchP;

    fParticleTypes[0] = kSigStM;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResASigStPKchP) 
  {
    fConjAnalysisType = kResSigStMKchM;

    fParticleTypes[0] = kASigStP;
    fParticleTypes[1] = kKchP;
  }

//------------ RescSigSt0cKch ----------------------------------
  else if(fAnalysisType == kResSigSt0KchP) 
  {
    fConjAnalysisType = kResASigSt0KchM;

    fParticleTypes[0] = kSigSt0;
    fParticleTypes[1] = kKchP;
  }
  else if(fAnalysisType == kResASigSt0KchM) 
  {
    fConjAnalysisType = kResSigSt0KchP;

    fParticleTypes[0] = kASigSt0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResSigSt0KchM) 
  {
    fConjAnalysisType = kResASigSt0KchP;

    fParticleTypes[0] = kSigSt0;
    fParticleTypes[1] = kKchM;
  }
  else if(fAnalysisType == kResASigSt0KchP) 
  {
    fConjAnalysisType = kResSigSt0KchM;

    fParticleTypes[0] = kASigSt0;
    fParticleTypes[1] = kKchP;
  }

//------------ RescLamcKSt0 ----------------------------------
  else if(fAnalysisType == kResLamKSt0) 
  {
    fConjAnalysisType = kResALamAKSt0;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kKSt0;
  }
  else if(fAnalysisType == kResALamAKSt0) 
  {
    fConjAnalysisType = kResLamKSt0;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResLamAKSt0) 
  {
    fConjAnalysisType = kResALamKSt0;

    fParticleTypes[0] = kLam;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResALamKSt0) 
  {
    fConjAnalysisType = kResLamAKSt0;

    fParticleTypes[0] = kALam;
    fParticleTypes[1] = kKSt0;
  }

//------------ RescSig0cKSt0 ----------------------------------
  else if(fAnalysisType == kResSig0KSt0) 
  {
    fConjAnalysisType = kResASig0AKSt0;

    fParticleTypes[0] = kSig0;
    fParticleTypes[1] = kKSt0;
  }
  else if(fAnalysisType == kResASig0AKSt0) 
  {
    fConjAnalysisType = kResSig0KSt0;

    fParticleTypes[0] = kASig0;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResSig0AKSt0) 
  {
    fConjAnalysisType = kResASig0KSt0;

    fParticleTypes[0] = kSig0;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResASig0KSt0) 
  {
    fConjAnalysisType = kResSig0AKSt0;

    fParticleTypes[0] = kASig0;
    fParticleTypes[1] = kKSt0;
  }

//------------ RescXi0cKSt0 ----------------------------------
  else if(fAnalysisType == kResXi0KSt0) 
  {
    fConjAnalysisType = kResAXi0AKSt0;

    fParticleTypes[0] = kXi0;
    fParticleTypes[1] = kKSt0;
  }
  else if(fAnalysisType == kResAXi0AKSt0) 
  {
    fConjAnalysisType = kResXi0KSt0;

    fParticleTypes[0] = kAXi0;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResXi0AKSt0) 
  {
    fConjAnalysisType = kResAXi0KSt0;

    fParticleTypes[0] = kXi0;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResAXi0KSt0) 
  {
    fConjAnalysisType = kResXi0AKSt0;

    fParticleTypes[0] = kAXi0;
    fParticleTypes[1] = kKSt0;
  }

//------------ RescXiCcKSt0 ----------------------------------
  else if(fAnalysisType == kResXiCKSt0) 
  {
    fConjAnalysisType = kResAXiCAKSt0;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kKSt0;
  }
  else if(fAnalysisType == kResAXiCAKSt0) 
  {
    fConjAnalysisType = kResXiCKSt0;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResXiCAKSt0) 
  {
    fConjAnalysisType = kResAXiCKSt0;

    fParticleTypes[0] = kXi;
    fParticleTypes[1] = kAKSt0;
  }
  else if(fAnalysisType == kResAXiCKSt0) 
  {
    fConjAnalysisType = kResXiCAKSt0;

    fParticleTypes[0] = kAXi;
    fParticleTypes[1] = kKSt0;
  }

//__________________________________________________________________
//                           COMMON TO ALL
//__________________________________________________________________

  fParticlePDGTypes[0] = GetParticlePDGType(fParticleTypes[0]);
  fParticlePDGTypes[1] = GetParticlePDGType(fParticleTypes[1]);

  SetDaughterPairType();
  SetCoulombType();
  SetBohrRadius();
  SetIsResidual();
}


//________________________________________________________________________________________________________________
AnalysisInfo::~AnalysisInfo()
{
  /* no-op */
}


//________________________________________________________________________________________________________________
ParticlePDGType AnalysisInfo::GetParticlePDGType(ParticleType aParticleType)
{
  int tPosition = static_cast<int>(aParticleType);
  int tPDGValue = cPDGValues[tPosition];
  ParticlePDGType tReturnType = static_cast<ParticlePDGType>(tPDGValue);

  cout << "remove these lines after confirming this technique works" << endl;
  cout << "aParticleType = " << aParticleType << endl;
  cout << "tPosition = " << tPosition << endl;
  cout << "tPDGValue = " << tPDGValue << endl;
  cout << "tReturnType = " << tReturnType << endl;

  return tReturnType;
}

//________________________________________________________________________________________________________________
void AnalysisInfo::SetDaughterPairType()
{
  if( (fAnalysisType==kLamK0) || (fAnalysisType==kALamK0) || (fAnalysisType==kLamLam) || (fAnalysisType==kALamALam) || (fAnalysisType==kLamALam) ) 
  {

    fDaughterPairTypes.push_back(kPosPos);
    fDaughterPairTypes.push_back(kPosNeg);
    fDaughterPairTypes.push_back(kNegPos);
    fDaughterPairTypes.push_back(kNegNeg);

  }

  else if( (fAnalysisType==kLamKchP) || (fAnalysisType==kALamKchP) || (fAnalysisType==kLamKchM) || (fAnalysisType==kALamKchM) || (fAnalysisType==kLamPiP) || (fAnalysisType==kALamPiP) || (fAnalysisType==kLamPiM) || (fAnalysisType==kALamPiM) ) 
  {

    fDaughterPairTypes.push_back(kTrackPos);
    fDaughterPairTypes.push_back(kTrackNeg);

  }

  else if( (fAnalysisType==kXiKchP) || (fAnalysisType==kAXiKchP) || (fAnalysisType==kXiKchM) || (fAnalysisType==kAXiKchM) )
  {

    fDaughterPairTypes.push_back(kTrackPos);
    fDaughterPairTypes.push_back(kTrackNeg);
    fDaughterPairTypes.push_back(kTrackBac);

  }
}


//________________________________________________________________________________________________________________
void AnalysisInfo::SetCoulombType()
{
  if(fAnalysisType==kXiKchP || fAnalysisType==kAXiKchM || 
     fAnalysisType==kResXiCKchP || fAnalysisType==kResAXiCKchM || 
     fAnalysisType==kResOmegaKchP || fAnalysisType==kResAOmegaKchM || 
     fAnalysisType==kResSigStPKchM || fAnalysisType==kResASigStMKchP || 
     fAnalysisType==kResSigStMKchP || fAnalysisType==kResASigStPKchM)
  {
    fCoulombType = kAttractive;
  }
  else if(fAnalysisType==kXiKchM || fAnalysisType==kAXiKchP ||
          fAnalysisType==kResXiCKchM || fAnalysisType==kResAXiCKchP ||
          fAnalysisType==kResOmegaKchM || fAnalysisType==kResAOmegaKchP ||
          fAnalysisType==kResSigStPKchP || fAnalysisType==kResASigStMKchM || 
          fAnalysisType==kResSigStMKchM || kResASigStPKchP)
  {
    fCoulombType = kRepulsive;
  }
  else fCoulombType = kNeutral;
}

//________________________________________________________________________________________________________________
void AnalysisInfo::SetBohrRadius()
{
  switch(fAnalysisType) {
  case kXiKchP:
  case kAXiKchM:
  case kResXiCKchP:
  case kResAXiCKchM:
    fBohrRadius = -gBohrRadiusXiK;
    break;

  case kXiKchM:
  case kAXiKchP:
  case kResXiCKchM:
  case kResAXiCKchP:
    fBohrRadius = gBohrRadiusXiK;
    break;


  case kResOmegaKchP:
  case kResAOmegaKchM:
    fBohrRadius = -gBohrRadiusOmegaK;
    break;


  case kResOmegaKchM:
  case kResAOmegaKchP:
    fBohrRadius = gBohrRadiusOmegaK;
    break;


  case kResSigStPKchM:
  case kResASigStPKchM:
    fBohrRadius = -gBohrRadiusSigStPK;
    break;

  case kResASigStMKchP:
  case kResSigStMKchP:
    fBohrRadius = -gBohrRadiusSigStMK;
    break;


  case kResSigStPKchP:
  case kResASigStPKchP:
    fBohrRadius = gBohrRadiusSigStPK;
    break;

  case kResASigStMKchM:
  case kResSigStMKchM:
    fBohrRadius = gBohrRadiusSigStMK;
    break;

  default:
    fBohrRadius = 1000000000;
  }

  if(fCoulombType==kAttractive) assert(fBohrRadius<0);
  else if(fCoulombType==kRepulsive) assert(fBohrRadius>0 && fBohrRadius<1000000000);
  else assert(fBohrRadius==1000000000);
}


//________________________________________________________________________________________________________________
void AnalysisInfo::SetIsResidual()
{
  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0 ||
     fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP || 
     fAnalysisType==kXiKchP || fAnalysisType==kAXiKchM || fAnalysisType==kXiKchM || fAnalysisType==kAXiKchP || 
     fAnalysisType==kXiK0 || fAnalysisType==kAXiK0 || 
     fAnalysisType==kLamLam || fAnalysisType==kALamALam || fAnalysisType==kLamALam ||
     fAnalysisType==kLamPiP || fAnalysisType==kALamPiM || fAnalysisType==kLamPiM || fAnalysisType==kALamPiP)
  {
    fIsResidual = false;
  }
  else fIsResidual = true;
}


