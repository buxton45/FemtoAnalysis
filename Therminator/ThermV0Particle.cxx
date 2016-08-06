///////////////////////////////////////////////////////////////////////////
// ThermV0Particle:                                                      //
///////////////////////////////////////////////////////////////////////////


#include "ThermV0Particle.h"

#ifdef __ROOT__
ClassImp(ThermV0Particle)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle() :
  ThermParticle(),
  fDaughter1Found(false),
  fDaughter2Found(false),
  fBothDaughtersFound(false),
  fGoodLambda(false),
  fDaughter1PID(0),
  fDaughter2PID(0),
  fDaughter1EID(0),
  fDaughter2EID(0),

  fDaughter1Mass(0), fDaughter1T(0), fDaughter1X(0), fDaughter1Y(0), fDaughter1Z(0), fDaughter1E(0), fDaughter1Px(0), fDaughter1Py(0), fDaughter1Pz(0),
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0)
{

}


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(ParticleCoor* aParticle) :
  ThermParticle(aParticle),
  fDaughter1Found(false),
  fDaughter2Found(false),
  fBothDaughtersFound(false),
  fGoodLambda(false),
  fDaughter1PID(),
  fDaughter2PID(0),
  fDaughter1EID(0),
  fDaughter2EID(0),

  fDaughter1Mass(0), fDaughter1T(0), fDaughter1X(0), fDaughter1Y(0), fDaughter1Z(0), fDaughter1E(0), fDaughter1Px(0), fDaughter1Py(0), fDaughter1Pz(0),
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0)
{

}


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(const ThermV0Particle& aParticle) :
  ThermParticle(aParticle),
  fDaughter1Found(aParticle.fDaughter1Found),
  fDaughter2Found(aParticle.fDaughter2Found),
  fBothDaughtersFound(aParticle.fBothDaughtersFound),
  fGoodLambda(aParticle.fGoodLambda),
  fDaughter1PID(aParticle.fDaughter1PID),
  fDaughter2PID(aParticle.fDaughter2PID),
  fDaughter1EID(aParticle.fDaughter1EID),
  fDaughter2EID(aParticle.fDaughter2EID),

  fDaughter1Mass(aParticle.fDaughter1Mass), 
  fDaughter1T(aParticle.fDaughter1T), fDaughter1X(aParticle.fDaughter1X), fDaughter1Y(aParticle.fDaughter1Y), fDaughter1Z(aParticle.fDaughter1Z),
  fDaughter1E(aParticle.fDaughter1E), fDaughter1Px(aParticle.fDaughter1Px), fDaughter1Py(aParticle.fDaughter1Py), fDaughter1Pz(aParticle.fDaughter1Pz),

  fDaughter2Mass(aParticle.fDaughter2Mass), 
  fDaughter2T(aParticle.fDaughter2T), fDaughter2X(aParticle.fDaughter2X), fDaughter2Y(aParticle.fDaughter2Y), fDaughter2Z(aParticle.fDaughter2Z),
  fDaughter2E(aParticle.fDaughter2E), fDaughter2Px(aParticle.fDaughter2Px), fDaughter2Py(aParticle.fDaughter2Py), fDaughter2Pz(aParticle.fDaughter2Pz)
{

}

//________________________________________________________________________________________________________________
ThermV0Particle& ThermV0Particle::operator=(const ThermV0Particle& aParticle)
{
  if(this == &aParticle) return *this;

  ThermParticle::operator=(aParticle);

  fDaughter1Found = aParticle.fDaughter1Found;
  fDaughter2Found = aParticle.fDaughter2Found;
  fBothDaughtersFound = aParticle.fBothDaughtersFound;
  fGoodLambda = aParticle.fGoodLambda;

  fDaughter1PID = aParticle.fDaughter1PID;
  fDaughter2PID = aParticle.fDaughter2PID;
  fDaughter1EID = aParticle.fDaughter1EID;
  fDaughter2EID = aParticle.fDaughter2EID;

  fDaughter1Mass = aParticle.fDaughter1Mass; 
  fDaughter1T = aParticle.fDaughter1T;
  fDaughter1X = aParticle.fDaughter1X;
  fDaughter1Y = aParticle.fDaughter1Y;
  fDaughter1Z = aParticle.fDaughter1Z;
  fDaughter1E = aParticle.fDaughter1E;
  fDaughter1Px = aParticle.fDaughter1Px;
  fDaughter1Py = aParticle.fDaughter1Py;
  fDaughter1Pz = aParticle.fDaughter1Pz;

  fDaughter2Mass = aParticle.fDaughter2Mass; 
  fDaughter2T = aParticle.fDaughter2T;
  fDaughter2X = aParticle.fDaughter2X;
  fDaughter2Y = aParticle.fDaughter2Y;
  fDaughter2Z = aParticle.fDaughter2Z;
  fDaughter2E = aParticle.fDaughter2E;
  fDaughter2Px = aParticle.fDaughter2Px;
  fDaughter2Py = aParticle.fDaughter2Py;
  fDaughter2Pz = aParticle.fDaughter2Pz;

  return *this;
}


//________________________________________________________________________________________________________________
ThermV0Particle* ThermV0Particle::clone()
{
  return(new ThermV0Particle(*this));
}


//________________________________________________________________________________________________________________
ThermV0Particle::~ThermV0Particle()
{
//  cout << "ThermV0Particle object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
void ThermV0Particle::LoadDaughter1(ThermParticle& aDaughter)
{
  assert(!fDaughter1Found);

  int tPID = aDaughter.GetPID();
  int tEID = aDaughter.GetEID();
  double tMass = aDaughter.GetMass();

  TLorentzVector* tFourPosition = aDaughter.GetFourPosition();
    double tT = tFourPosition->T();
    double tX = tFourPosition->X();
    double tY = tFourPosition->Y();
    double tZ = tFourPosition->Z();

  TLorentzVector* tFourMomentum = aDaughter.GetFourMomentum();
    double tE = tFourMomentum->E();
    double tPx = tFourMomentum->Px();
    double tPy = tFourMomentum->Py();
    double tPz = tFourMomentum->Pz();

  //-----------------------------------------------------------

  SetDaughter1PID(tPID);
  fDaughter1EID = tEID;

  fDaughter1Mass =tMass;

  fDaughter1T = tT; 
  fDaughter1X = tX; 
  fDaughter1Y = tY; 
  fDaughter1Z = tZ;

  fDaughter1E = tE; 
  fDaughter1Px = tPx;
  fDaughter1Py = tPy;
  fDaughter1Pz = tPz;

  fDaughter1Found = true;
}

//________________________________________________________________________________________________________________
void ThermV0Particle::LoadDaughter2(ThermParticle& aDaughter)
{
  assert(!fDaughter2Found);

  int tPID = aDaughter.GetPID();
  int tEID = aDaughter.GetEID();
  double tMass = aDaughter.GetMass();

  TLorentzVector* tFourPosition = aDaughter.GetFourPosition();
    double tT = tFourPosition->T();
    double tX = tFourPosition->X();
    double tY = tFourPosition->Y();
    double tZ = tFourPosition->Z();

  TLorentzVector* tFourMomentum = aDaughter.GetFourMomentum();
    double tE = tFourMomentum->E();
    double tPx = tFourMomentum->Px();
    double tPy = tFourMomentum->Py();
    double tPz = tFourMomentum->Pz();

  //-----------------------------------------------------------

  SetDaughter2PID(tPID);
  fDaughter2EID = tEID;

  fDaughter2Mass =tMass;

  fDaughter2T = tT; 
  fDaughter2X = tX; 
  fDaughter2Y = tY; 
  fDaughter2Z = tZ;

  fDaughter2E = tE; 
  fDaughter2Px = tPx;
  fDaughter2Py = tPy;
  fDaughter2Pz = tPz;

  fDaughter2Found = true;
}

//________________________________________________________________________________________________________________
void ThermV0Particle::LoadDaughter(ThermParticle& aDaughter)
{
if(fBothDaughtersFound)
{
cout << "aDaughter.GetPID() = " << aDaughter.GetPID() << endl;
cout << "PID = " << fPID << endl;
cout << "fDaughter1PID = " << fDaughter1PID << endl;
cout << "fDaughter2PID = " << fDaughter2PID << endl;
}

  assert(!fBothDaughtersFound);

  int tPID = aDaughter.GetPID();

  if(abs(tPID) == 111 || abs(tPID) == 2112)  //if daughter is pi0 or neutron
  {
    if(fDaughter1Found) LoadDaughter2(aDaughter);
    else LoadDaughter1(aDaughter);
  }

  else if(tPID > 0)
  {
    LoadDaughter1(aDaughter);
  }

  else if(tPID < 0)
  {
    LoadDaughter2(aDaughter);
  }

  else assert(0);

  if(fDaughter1Found && fDaughter2Found) fBothDaughtersFound = true;

}







