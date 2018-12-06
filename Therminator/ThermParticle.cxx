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
ThermParticle::ThermParticle() :
  fPrimordial(false),
  fParticleOfInterest(false),
  fMass(0), 
  fT(0), fX(0), fY(0), fZ(0),
  fE(0), fPx(0), fPy(0), fPz(0),
  fDecayed(0), fPID(0), fFatherPID(0), fRootPID(0),
  fEID(0), fFatherEID(0), fEventID(0),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{

}

//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(ParticleCoor* aParticle) :
  fPrimordial(false),
  fParticleOfInterest(false),
  fMass(aParticle->mass), 
  fT(aParticle->t), fX(aParticle->x), fY(aParticle->y), fZ(aParticle->z),
  fE(aParticle->e), fPx(aParticle->px), fPy(aParticle->py), fPz(aParticle->pz),
  fDecayed(aParticle->decayed), fPID(aParticle->pid), fFatherPID(aParticle->fatherpid), fRootPID(aParticle->rootpid),
  fEID(aParticle->eid), fFatherEID(aParticle->fathereid), fEventID(aParticle->eventid),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{
  SetIsParticleOfInterest();
  if(fFatherEID == -1) fPrimordial = true;
}

//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(vector<double> &aVecFromTxt) :
  fPrimordial(false),
  fParticleOfInterest(false),
  fMass(0), 
  fT(0), fX(0), fY(0), fZ(0),
  fE(0), fPx(0), fPy(0), fPz(0),
  fDecayed(0), fPID(0), fFatherPID(0), fRootPID(0),
  fEID(0), fFatherEID(0), fEventID(0),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{
  assert(aVecFromTxt.size() == 27);

  fPrimordial = aVecFromTxt[0];
  fParticleOfInterest = aVecFromTxt[1];

  fMass = aVecFromTxt[2];

  fT = aVecFromTxt[3];
  fX = aVecFromTxt[4];
  fY = aVecFromTxt[5];
  fZ = aVecFromTxt[6];

  fE = aVecFromTxt[7];
  fPx = aVecFromTxt[8];
  fPy = aVecFromTxt[9];
  fPz = aVecFromTxt[10];

  fDecayed = aVecFromTxt[11];
  fPID = aVecFromTxt[12];
  fFatherPID = aVecFromTxt[13];
  fRootPID = aVecFromTxt[14];
  fEID = aVecFromTxt[15];
  fFatherEID = aVecFromTxt[16];
  fEventID = aVecFromTxt[17];

  fFatherMass = aVecFromTxt[18];

  fFatherT = aVecFromTxt[19];
  fFatherX = aVecFromTxt[20];
  fFatherY = aVecFromTxt[21];
  fFatherZ = aVecFromTxt[22];

  fFatherE = aVecFromTxt[23];
  fFatherPx = aVecFromTxt[24];
  fFatherPy = aVecFromTxt[25];
  fFatherPz = aVecFromTxt[26];

}



//________________________________________________________________________________________________________________
ThermParticle::ThermParticle(const ThermParticle& aParticle) :
  fPrimordial(aParticle.fPrimordial), fParticleOfInterest(aParticle.fParticleOfInterest), fMass(aParticle.fMass), 
  fT(aParticle.fT), fX(aParticle.fX), fY(aParticle.fY), fZ(aParticle.fZ),
  fE(aParticle.fE), fPx(aParticle.fPx), fPy(aParticle.fPy), fPz(aParticle.fPz),
  fDecayed(aParticle.fDecayed), fPID(aParticle.fPID), fFatherPID(aParticle.fFatherPID), fRootPID(aParticle.fRootPID),
  fEID(aParticle.fEID), fFatherEID(aParticle.fFatherEID), fEventID(aParticle.fEventID),

  fFatherMass(aParticle.fFatherMass), 
  fFatherT(aParticle.fFatherT), fFatherX(aParticle.fFatherX), fFatherY(aParticle.fFatherY), fFatherZ(aParticle.fFatherZ),
  fFatherE(aParticle.fFatherE), fFatherPx(aParticle.fFatherPx), fFatherPy(aParticle.fFatherPy), fFatherPz(aParticle.fFatherPz)
{

}

//________________________________________________________________________________________________________________
ThermParticle& ThermParticle::operator=(const ThermParticle& aParticle)
{
  if(this == &aParticle) return *this;

  fPrimordial = aParticle.fPrimordial;
  fParticleOfInterest = aParticle.fParticleOfInterest;
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

  fFatherMass = aParticle.fFatherMass; 
  fFatherT = aParticle.fFatherT;
  fFatherX = aParticle.fFatherX;
  fFatherY = aParticle.fFatherY;
  fFatherZ = aParticle.fFatherZ;
  fFatherE = aParticle.fFatherE;
  fFatherPx = aParticle.fFatherPx;
  fFatherPy = aParticle.fFatherPy;
  fFatherPz = aParticle.fFatherPz;

  return *this;
}

//________________________________________________________________________________________________________________
ThermParticle* ThermParticle::clone()
{
  return(new ThermParticle(*this));
}

//________________________________________________________________________________________________________________
ThermParticle::~ThermParticle()
{
//  cout << "ThermParticle object is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void ThermParticle::SetIsParticleOfInterest()
{
  if(fPID == 3122) fParticleOfInterest = true;        //Lam
  else if(fPID == -3122) fParticleOfInterest = true;  //ALam

  else if(fPID == 311) fParticleOfInterest = true;    //K0 (K0s = 310, but apparently there are no K0s)

  else if(fPID == 321) fParticleOfInterest = true;    //KchP
  else if(fPID == -321) fParticleOfInterest = true;   //KchM 

  else fParticleOfInterest = false;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetTau() const
{
  return TMath::Sqrt(fT*fT - fZ*fZ)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetR() const
{
  return TMath::Sqrt(fX*fX + fY*fY + fZ*fZ)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRho() const
{
  return TMath::Sqrt(fX*fX + fY*fY)*kHbarC;
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPhiS() const
{
  return TMath::ATan2(fY,fX);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRapidityS() const
{
  return 0.5 * TMath::Log( (fT + fZ) / (fT - fZ) );
}

//________________________________________________________________________________________________________________
double ThermParticle::GetP() const
{
  return TMath::Sqrt(fPx*fPx + fPy*fPy + fPz*fPz);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPt() const
{
  return TMath::Sqrt(fPx*fPx + fPy*fPy);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetMt() const
{
  return TMath::Sqrt(fMass*fMass + fPx*fPx + fPy*fPy);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetPhiP() const
{
  return TMath::ATan2(fPy,fPx);
}

//________________________________________________________________________________________________________________
double ThermParticle::GetRapidityP() const
{
  return 0.5 * TMath::Log( (fE + fPz) / (fE - fPz) );
}

//________________________________________________________________________________________________________________
double ThermParticle::GetEtaP() const
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
  //NOTE: Clockwise rotation!!! (to be consistent with ParticleCoor::TransformRotateZ
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


//________________________________________________________________________________________________________________
TLorentzVector ThermParticle::GetFourPosition() const
{
  return TLorentzVector(fX,fY,fZ,fT);
}


//________________________________________________________________________________________________________________
TLorentzVector ThermParticle::GetFourMomentum() const
{
  return TLorentzVector(fPx,fPy,fPz,fE);
}

//________________________________________________________________________________________________________________
TLorentzVector ThermParticle::GetFatherFourPosition() const
{
  return TLorentzVector(fFatherX,fFatherY,fFatherZ,fFatherT);
}


//________________________________________________________________________________________________________________
TLorentzVector ThermParticle::GetFatherFourMomentum() const
{
  return TLorentzVector(fFatherPx,fFatherPy,fFatherPz,fFatherE);
}



//________________________________________________________________________________________________________________
void ThermParticle::LoadFather(ThermParticle& aFather)
{
  int tPID = aFather.GetPID();
  int tEID = aFather.GetEID();

  //Just double check here...
  assert(tPID == fFatherPID);
  assert(tEID == fFatherEID);

  double tMass = aFather.GetMass();

  TLorentzVector tFourPosition = aFather.GetFourPosition();
    double tT = tFourPosition.T();
    double tX = tFourPosition.X();
    double tY = tFourPosition.Y();
    double tZ = tFourPosition.Z();

  TLorentzVector tFourMomentum = aFather.GetFourMomentum();
    double tE = tFourMomentum.E();
    double tPx = tFourMomentum.Px();
    double tPy = tFourMomentum.Py();
    double tPz = tFourMomentum.Pz();

  //-----------------------------------------------------------

  fFatherMass = tMass;

  fFatherT = tT; 
  fFatherX = tX; 
  fFatherY = tY; 
  fFatherZ = tZ;

  fFatherE = tE; 
  fFatherPx = tPx;
  fFatherPy = tPy;
  fFatherPz = tPz;

}

//________________________________________________________________________________________________________________
bool ThermParticle::PassKinematicCuts()
{
  if(abs(GetEtaP()) > 0.8) return false;

  if(fPID==kPDGKchP || fPID==kPDGKchM)
  {
    if(GetPt()<0.14 || GetPt()>1.5) return false;
  }

  return true;
}









