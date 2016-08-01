///////////////////////////////////////////////////////////////////////////
// ThermParticle:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "ThermParticle.h"

#ifdef __ROOT__
ClassImp(ThermParticle)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(ParticleType aType) :
  fParticleType(aType),
  fMass(0), 
  fT(0), fX(0), fY(0), fZ(0),
  fE(0), fPx(0), fPy(0), fPz(0),
  fDecayed(0), fPID(0), fFatherPID(0), fRootPID(0),
  fEID(0), fFatherEID(0), fEventID(0)
{

}

//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(ParticleCoor* aParticle) :
//  fParticleType(aType),
  fMass(aParticle->mass), 
  fT(aParticle->t), fX(aParticle->x), fY(aParticle->y), fZ(aParticle->z),
  fE(aParticle->e), fPx(aParticle->px), fPy(aParticle->py), fPz(aParticle->pz),
  fDecayed(aParticle->decayed), fPID(aParticle->pid), fFatherPID(aParticle->fatherpid), fRootPID(aParticle->rootpid),
  fEID(aParticle->eid), fFatherEID(aParticle->fathereid), fEventID(aParticle->eventid)
{
  SetParticleType();
}


//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(const ThermParticle& aParticle) :
  fParticleType(aParticle.fParticleType), fMass(aParticle.fMass), 
  fT(aParticle.fT), fX(aParticle.fX), fY(aParticle.fY), fZ(aParticle.fZ),
  fE(aParticle.fE), fPx(aParticle.fPx), fPy(aParticle.fPy), fPz(aParticle.fPz),
  fDecayed(aParticle.fDecayed), fPID(aParticle.fPID), fFatherPID(aParticle.fFatherPID), fRootPID(aParticle.fRootPID),
  fEID(aParticle.fEID), fFatherEID(aParticle.fFatherEID), fEventID(aParticle.fEventID)
{

}

//________________________________________________________________________________________________________________
ThermParticle& ThermParticle::operator=(ThermParticle& aParticle)
{
  if(this == &aParticle) return *this;

  fParticleType = aParticle.fParticleType;
  fMass = aParticle.fMass; 
  fT = aParticle.fT;
  fX = aParticle.fX;
  fY = aParticle.fY;
  fZ = aParticle.fZ;
  fE = aParticle.fE;
  fPx = aParticle.fPx;
  fPy = aParticle.fPy;
  fPz = aParticle.fPz;
  fDecayed = aParticle.fDecayed;
  fPID = aParticle.fPID;
  fFatherPID = aParticle.fFatherPID;
  fRootPID = aParticle.fRootPID;
  fEID = aParticle.fEID;
  fFatherEID = aParticle.fFatherEID;
  fEventID = aParticle.fEventID;

  return *this;
}

//________________________________________________________________________________________________________________
ThermParticle::~ThermParticle()
{
  cout << "ThermParticle object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
void ThermParticle::SetParticleType()
{
  if(fPID == 3122) SetParticleType(kLam);
  else if(fPID == -3122) SetParticleType(kALam);

  else if(fPID == 310) SetParticleType(kK0);

  else if(fPID == 321) SetParticleType(kKchP);
  else if(fPID == -321) SetParticleType(kKchM);

  else assert(0); //TODO
}



//________________________________________________________________________________________________________________
double ThermParticle::GetTau()
{
  return TMath::Sqrt(fT*fT - fZ*fZ)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetR()
{
  return TMath::Sqrt(fX*fX + fY*fY + fZ*fZ)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRho()
{
  return TMath::Sqrt(fX*fX + fY*fY)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPhiS()
{
  return TMath::ATan2(fY,fX);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRapidityS()
{
  return 0.5 * TMath::Log( (fT + fZ) / (fT - fZ) );
}

//________________________________________________________________________________________________________________
double ThermParticle::GetP()
{
  return TMath::Sqrt(fPx*fPx + fPy*fPy + fPz*fPz);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPt()
{
  return TMath::Sqrt(fPx*fPx + fPy*fPy);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetMt()
{
  return TMath::Sqrt(fMass*fMass + fPx*fPx + fPy*fPy);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPhiP()
{
  return TMath::ATan2(fPy,fPx);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRapidityP()
{
  return 0.5 * TMath::Log( (fE + fPz) / (fE - fPz) );
}

//________________________________________________________________________________________________________________
double ThermParticle::GetEtaP()
{
  double tP = GetP();
  return 0.5 * TMath::Log( (tP + fPz) / (tP - fPz) );
}

//________________________________________________________________________________________________________________
void ThermParticle::TransformToLCMS(double aBetaZ)
{
  double tmp;
  double tGammaZ = 1.0 / TMath::Sqrt(1.0 - aBetaZ*aBetaZ);

  tmp = tGammaZ * (fE  - aBetaZ * fPz);
  fPz  = tGammaZ * (fPz - aBetaZ * fE );
  fE   = tmp;
  
  tmp = tGammaZ * (fT  - aBetaZ * fZ );
  fZ   = tGammaZ * (fZ  - aBetaZ * fT );
  fT   = tmp;
}


//________________________________________________________________________________________________________________
void ThermParticle::TransformRotateZ(double aPhi)
{
  double tmp;
  double tCosPhi = TMath::Cos(aPhi);
  double tSinPhi = TMath::Sin(aPhi);
  
  tmp = ( fPx * tCosPhi + fPy * tSinPhi);
  fPy  = ( fPy * tCosPhi - fPx * tSinPhi);
  fPx  = tmp;
  
  tmp = ( fX  * tCosPhi + fY  * tSinPhi);
  fY   = ( fY  * tCosPhi - fX  * tSinPhi);
  fX   = tmp;
}


//________________________________________________________________________________________________________________
void ThermParticle::TransformToPRF(double aBetaT)
{
  double tmp;
  double tGammaT = 1.0 / TMath::Sqrt(1.0 - aBetaT*aBetaT);
  
  tmp = tGammaT * (fE  - aBetaT * fPx);
  fPx  = tGammaT * (fPx - aBetaT * fE);
  fE   = tmp;
  
  tmp = tGammaT * (fT  - aBetaT * fX);
  fX   = tGammaT * (fX  - aBetaT * fT);
  fT   = tmp;
}



















