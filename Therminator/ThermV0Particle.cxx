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
  fGoodV0(false),
  fDaughter1PID(0),
  fDaughter2PID(0),
  fDaughter1EID(0),
  fDaughter2EID(0),

  fDaughter1Mass(0), fDaughter1T(0), fDaughter1X(0), fDaughter1Y(0), fDaughter1Z(0), fDaughter1E(0), fDaughter1Px(0), fDaughter1Py(0), fDaughter1Pz(0),
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{

}


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(ParticleCoor* aParticle) :
  ThermParticle(aParticle),
  fDaughter1Found(false),
  fDaughter2Found(false),
  fBothDaughtersFound(false),
  fGoodV0(false),
  fDaughter1PID(),
  fDaughter2PID(0),
  fDaughter1EID(0),
  fDaughter2EID(0),

  fDaughter1Mass(0), fDaughter1T(0), fDaughter1X(0), fDaughter1Y(0), fDaughter1Z(0), fDaughter1E(0), fDaughter1Px(0), fDaughter1Py(0), fDaughter1Pz(0),
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{

}


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(vector<double> &aVecFromTxt) :
  ThermParticle(),
  fDaughter1Found(false),
  fDaughter2Found(false),
  fBothDaughtersFound(false),
  fGoodV0(false),
  fDaughter1PID(0),
  fDaughter2PID(0),
  fDaughter1EID(0),
  fDaughter2EID(0),

  fDaughter1Mass(0), fDaughter1T(0), fDaughter1X(0), fDaughter1Y(0), fDaughter1Z(0), fDaughter1E(0), fDaughter1Px(0), fDaughter1Py(0), fDaughter1Pz(0),
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0),
  fFatherMass(0), fFatherT(0), fFatherX(0), fFatherY(0), fFatherZ(0), fFatherE(0), fFatherPx(0), fFatherPy(0), fFatherPz(0)
{
  assert(aVecFromTxt.size() == 53);
  //------ThermParticle
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

  //------ThermV0Particle
  fDaughter1Found = aVecFromTxt[18];
  fDaughter2Found = aVecFromTxt[19];
  fBothDaughtersFound = aVecFromTxt[20];

  fGoodV0 = aVecFromTxt[21];

  fDaughter1PID = aVecFromTxt[22];
  fDaughter2PID = aVecFromTxt[23];

  fDaughter1EID = aVecFromTxt[24];
  fDaughter2EID = aVecFromTxt[25];

  fDaughter1Mass = aVecFromTxt[26];

  fDaughter1T = aVecFromTxt[27];
  fDaughter1X = aVecFromTxt[28];
  fDaughter1Y = aVecFromTxt[29];
  fDaughter1Z = aVecFromTxt[30];

  fDaughter1E = aVecFromTxt[31];
  fDaughter1Px = aVecFromTxt[32];
  fDaughter1Py = aVecFromTxt[33];
  fDaughter1Pz = aVecFromTxt[34];

  fDaughter2Mass = aVecFromTxt[35];

  fDaughter2T = aVecFromTxt[36];
  fDaughter2X = aVecFromTxt[37];
  fDaughter2Y = aVecFromTxt[38];
  fDaughter2Z = aVecFromTxt[39];

  fDaughter2E = aVecFromTxt[40];
  fDaughter2Px = aVecFromTxt[41];
  fDaughter2Py = aVecFromTxt[42];
  fDaughter2Pz = aVecFromTxt[43];

  fFatherMass = aVecFromTxt[44];

  fFatherT = aVecFromTxt[45];
  fFatherX = aVecFromTxt[46];
  fFatherY = aVecFromTxt[47];
  fFatherZ = aVecFromTxt[48];

  fFatherE = aVecFromTxt[49];
  fFatherPx = aVecFromTxt[50];
  fFatherPy = aVecFromTxt[51];
  fFatherPz = aVecFromTxt[52];

}


//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(const ThermV0Particle& aParticle) :
  ThermParticle(aParticle),
  fDaughter1Found(aParticle.fDaughter1Found),
  fDaughter2Found(aParticle.fDaughter2Found),
  fBothDaughtersFound(aParticle.fBothDaughtersFound),
  fGoodV0(aParticle.fGoodV0),
  fDaughter1PID(aParticle.fDaughter1PID),
  fDaughter2PID(aParticle.fDaughter2PID),
  fDaughter1EID(aParticle.fDaughter1EID),
  fDaughter2EID(aParticle.fDaughter2EID),

  fDaughter1Mass(aParticle.fDaughter1Mass), 
  fDaughter1T(aParticle.fDaughter1T), fDaughter1X(aParticle.fDaughter1X), fDaughter1Y(aParticle.fDaughter1Y), fDaughter1Z(aParticle.fDaughter1Z),
  fDaughter1E(aParticle.fDaughter1E), fDaughter1Px(aParticle.fDaughter1Px), fDaughter1Py(aParticle.fDaughter1Py), fDaughter1Pz(aParticle.fDaughter1Pz),

  fDaughter2Mass(aParticle.fDaughter2Mass), 
  fDaughter2T(aParticle.fDaughter2T), fDaughter2X(aParticle.fDaughter2X), fDaughter2Y(aParticle.fDaughter2Y), fDaughter2Z(aParticle.fDaughter2Z),
  fDaughter2E(aParticle.fDaughter2E), fDaughter2Px(aParticle.fDaughter2Px), fDaughter2Py(aParticle.fDaughter2Py), fDaughter2Pz(aParticle.fDaughter2Pz),

  fFatherMass(aParticle.fFatherMass), 
  fFatherT(aParticle.fFatherT), fFatherX(aParticle.fFatherX), fFatherY(aParticle.fFatherY), fFatherZ(aParticle.fFatherZ),
  fFatherE(aParticle.fFatherE), fFatherPx(aParticle.fFatherPx), fFatherPy(aParticle.fFatherPy), fFatherPz(aParticle.fFatherPz)
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
  fGoodV0 = aParticle.fGoodV0;

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
//TODO clearly this class needs to be reconsidered
//maybe ThermV0Particle should not inherit from ThermParticle, but should rather have ThermParticle members representing
//the father and daughters?  For now, this works
bool ThermV0Particle::PassDaughterCuts(int aDaughter)
{
  int tPID;
  double tPx, tPy, tPz, tPt, tPMag, tEta;

  if(aDaughter == 1)
  {
    tPx = fDaughter1Px;
    tPy = fDaughter1Py;
    tPz = fDaughter1Pz;

    tPID = fDaughter1PID;
  }
  else if(aDaughter == 2)
  {
    tPx = fDaughter2Px;
    tPy = fDaughter2Py;
    tPz = fDaughter2Pz;

    tPID = fDaughter2PID;
  }
  else assert(0);

  tPt = sqrt(tPx*tPx + tPy*tPy);
  tPMag = sqrt(tPx*tPx + tPy*tPy + tPz*tPz);
  tEta = 0.5*log((tPMag+tPz)/(tPMag-tPz));

  if(abs(tEta) > 0.8) return false;

  if(tPID == kPDGPiP || tPID == kPDGPiM)
  {
    if(tPt < 0.16) return false;
  }
  else if(tPID == kPDGProt)
  {
    if(tPt < 0.5) return false;
  }
  else if(tPID == kPDGAntiProt)
  {
    if(tPt < 0.3) return false;
  }
  else return false;  //daughter are of wrong type.  Note, this does not mean reconstruction was bad,
		      // but is due to daughters being neutral (for instance, Lambda-> pi0 + n)

  return true;
}

//________________________________________________________________________________________________________________
bool ThermV0Particle::PassV0Cuts()
{
  if(abs(GetEtaP()) > 0.8) return false;

  if(fPID == kPDGLam || fPID == kPDGALam)
  {
    if(GetPt() < 0.4) return false;
  }
  else if(fPID == kPDGK0)
  {
    if(GetPt() < 0.2) return false;
  }
  else
  {
    cout << "V0 of wrong type for this analysis is selected.  Prepare for crash" << endl;
    assert(0);
  }

  if(!PassDaughterCuts(1)) return false;
  if(!PassDaughterCuts(2)) return false;

  return true;
}

//________________________________________________________________________________________________________________
void ThermV0Particle::LoadDaughter1(ThermParticle& aDaughter)
{
  assert(!fDaughter1Found);

  int tPID = aDaughter.GetPID();
  int tEID = aDaughter.GetEID();
  double tMass = aDaughter.GetMass();

  TLorentzVector tFourPosition = aDaughter.GetFourPosition();
    double tT = tFourPosition.T();
    double tX = tFourPosition.X();
    double tY = tFourPosition.Y();
    double tZ = tFourPosition.Z();

  TLorentzVector tFourMomentum = aDaughter.GetFourMomentum();
    double tE = tFourMomentum.E();
    double tPx = tFourMomentum.Px();
    double tPy = tFourMomentum.Py();
    double tPz = tFourMomentum.Pz();

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

  TLorentzVector tFourPosition = aDaughter.GetFourPosition();
    double tT = tFourPosition.T();
    double tX = tFourPosition.X();
    double tY = tFourPosition.Y();
    double tZ = tFourPosition.Z();

  TLorentzVector tFourMomentum = aDaughter.GetFourMomentum();
    double tE = tFourMomentum.E();
    double tPx = tFourMomentum.Px();
    double tPy = tFourMomentum.Py();
    double tPz = tFourMomentum.Pz();

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

  if(fDaughter1Found && fDaughter2Found)  //Note, fGoodV0 cannot be true for K0L because it has no daughters, so will never be set true
  {
    fBothDaughtersFound = true;
    fGoodV0 = PassV0Cuts();
  }

}


//________________________________________________________________________________________________________________
void ThermV0Particle::LoadFather(ThermParticle& aFather)
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





