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
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0)
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
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0)
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
  fDaughter2Mass(0), fDaughter2T(0), fDaughter2X(0), fDaughter2Y(0), fDaughter2Z(0), fDaughter2E(0), fDaughter2Px(0), fDaughter2Py(0), fDaughter2Pz(0)
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

  fFatherMass = aVecFromTxt[18];

  fFatherT = aVecFromTxt[19];
  fFatherX = aVecFromTxt[20];
  fFatherY = aVecFromTxt[21];
  fFatherZ = aVecFromTxt[22];

  fFatherE = aVecFromTxt[23];
  fFatherPx = aVecFromTxt[24];
  fFatherPy = aVecFromTxt[25];
  fFatherPz = aVecFromTxt[26];


  //------ThermV0Particle
  fDaughter1Found = aVecFromTxt[27];
  fDaughter2Found = aVecFromTxt[28];
  fBothDaughtersFound = aVecFromTxt[29];

  fGoodV0 = aVecFromTxt[30];

  fDaughter1PID = aVecFromTxt[31];
  fDaughter2PID = aVecFromTxt[32];

  fDaughter1EID = aVecFromTxt[33];
  fDaughter2EID = aVecFromTxt[34];

  fDaughter1Mass = aVecFromTxt[35];

  fDaughter1T = aVecFromTxt[36];
  fDaughter1X = aVecFromTxt[37];
  fDaughter1Y = aVecFromTxt[38];
  fDaughter1Z = aVecFromTxt[39];

  fDaughter1E = aVecFromTxt[40];
  fDaughter1Px = aVecFromTxt[41];
  fDaughter1Py = aVecFromTxt[42];
  fDaughter1Pz = aVecFromTxt[43];

  fDaughter2Mass = aVecFromTxt[44];

  fDaughter2T = aVecFromTxt[45];
  fDaughter2X = aVecFromTxt[46];
  fDaughter2Y = aVecFromTxt[47];
  fDaughter2Z = aVecFromTxt[48];

  fDaughter2E = aVecFromTxt[49];
  fDaughter2Px = aVecFromTxt[50];
  fDaughter2Py = aVecFromTxt[51];
  fDaughter2Pz = aVecFromTxt[52];
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
bool ThermV0Particle::DoubleCheckLamAttributes()
{
  if(!fGoodV0) {cout << "DoubleCheckLamAttributes Fail 1" << endl; return false;}
  if(fPID != kPDGLam) {cout << "DoubleCheckLamAttributes Fail 2" << endl; return false;}
  if(!fBothDaughtersFound) {cout << "DoubleCheckLamAttributes Fail 3" << endl; return false;}
  if(fDaughter1PID != kPDGProt) {cout << "DoubleCheckLamAttributes Fail 4" << endl; return false;}
  if(fDaughter2PID != kPDGPiM) {cout << "DoubleCheckLamAttributes Fail 5" << endl; return false;}

  return true;
}

//________________________________________________________________________________________________________________
bool ThermV0Particle::DoubleCheckALamAttributes()
{
  if(!fGoodV0) {cout << "DoubleCheckALamAttributes Fail 1" << endl; return false;}
  if(fPID != kPDGALam) {cout << "DoubleCheckALamAttributes Fail 2" << endl; return false;}
  if(!BothDaughtersFound()) {cout << "DoubleCheckALamAttributes Fail 3" << endl; return false;}
  if(fDaughter1PID != kPDGPiP) {cout << "DoubleCheckALamAttributes Fail 4" << endl; return false;}
  if(fDaughter2PID != kPDGAntiProt) {cout << "DoubleCheckALamAttributes Fail 5" << endl; return false;}

  return true;
}

//________________________________________________________________________________________________________________
bool ThermV0Particle::DoubleCheckK0Attributes()
{
  if(!fGoodV0) {cout << "DoubleCheckK0Attributes Fail 1" << endl; return false;}
  if(fPID != kPDGK0) {cout << "DoubleCheckK0Attributes Fail 2" << endl; return false;}
  if(!fBothDaughtersFound) {cout << "DoubleCheckK0Attributes Fail 3" << endl; return false;}
  if(fDaughter1PID != kPDGPiP) {cout << "DoubleCheckK0Attributes Fail 4" << endl; return false;}
  if(fDaughter2PID != kPDGPiM) {cout << "DoubleCheckK0Attributes Fail 5" << endl; return false;}

  return true;
}



//________________________________________________________________________________________________________________
bool ThermV0Particle::DoubleCheckV0Attributes()
{
  //------------------------------
  if(fDaughter1Mass==0) {cout << "DoubleCheckV0Attributes Fail 1" << endl; return false;}
  if(fDaughter1T==0 || fDaughter1X==0 || fDaughter1Y==0 ||fDaughter1Z==0) {cout << "DoubleCheckV0Attributes Fail 2" << endl; return false;}
  if(fDaughter1E==0 || fDaughter1Px==0 || fDaughter1Py==0 ||fDaughter1Pz==0) {cout << "DoubleCheckV0Attributes Fail 3" << endl; return false;}

  if(fDaughter2Mass==0) {cout << "DoubleCheckV0Attributes Fail 4" << endl; return false;}
  if(fDaughter2T==0 || fDaughter2X==0 || fDaughter2Y==0 ||fDaughter2Z==0) {cout << "DoubleCheckV0Attributes Fail 5" << endl; return false;}
  if(fDaughter2E==0 || fDaughter2Px==0 || fDaughter2Py==0 ||fDaughter2Pz==0) {cout << "DoubleCheckV0Attributes Fail 6" << endl; return false;}

  if(!fPrimordial)
  {
    if(fFatherT==0 || fFatherX==0 || fFatherY==0 ||fFatherZ==0) {cout << "DoubleCheckV0Attributes Fail 7" << endl; return false;}
    if(fFatherE==0 || fFatherPx==0 || fFatherPy==0 ||fFatherPz==0) {cout << "DoubleCheckV0Attributes Fail 8" << endl; return false;}
  }
  //------------------------------
  if(fPID == kPDGLam) return DoubleCheckLamAttributes();
  else if(fPID == kPDGALam) return DoubleCheckALamAttributes();
  else if(fPID == kPDGK0) return DoubleCheckK0Attributes();
  else assert(0);

}

