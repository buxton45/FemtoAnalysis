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
  fPosDaughterFound(false),
  fNegDaughterFound(false),
  fBothDaughtersFound(false),
  fGoodLambda(false),
  fPosDaughterPID(0),
  fNegDaughterPID(0),

  fMassPos(0), fTPos(0), fXPos(0), fYPos(0), fZPos(0), fEPos(0), fPxPos(0), fPyPos(0), fPzPos(0),
  fMassNeg(0), fTNeg(0), fXNeg(0), fYNeg(0), fZNeg(0), fENeg(0), fPxNeg(0), fPyNeg(0), fPzNeg(0)
{

}

//________________________________________________________________________________________________________________
ThermV0Particle::ThermV0Particle(const ThermV0Particle& aParticle) :
  ThermParticle(aParticle),
  fPosDaughterFound(aParticle.fPosDaughterFound),
  fNegDaughterFound(aParticle.fNegDaughterFound),
  fBothDaughtersFound(aParticle.fBothDaughtersFound),
  fGoodLambda(aParticle.fGoodLambda),
  fPosDaughterPID(aParticle.fPosDaughterPID),
  fNegDaughterPID(aParticle.fNegDaughterPID),

  fMassPos(aParticle.fMassPos), 
  fTPos(aParticle.fTPos), fXPos(aParticle.fXPos), fYPos(aParticle.fYPos), fZPos(aParticle.fZPos),
  fEPos(aParticle.fEPos), fPxPos(aParticle.fPxPos), fPyPos(aParticle.fPyPos), fPzPos(aParticle.fPzPos),

  fMassNeg(aParticle.fMassNeg), 
  fTNeg(aParticle.fTNeg), fXNeg(aParticle.fXNeg), fYNeg(aParticle.fYNeg), fZNeg(aParticle.fZNeg),
  fENeg(aParticle.fENeg), fPxNeg(aParticle.fPxNeg), fPyNeg(aParticle.fPyNeg), fPzNeg(aParticle.fPzNeg)
{

}

//________________________________________________________________________________________________________________
ThermV0Particle& ThermV0Particle::operator=(ThermV0Particle& aParticle)
{
  if(this == &aParticle) return *this;

  ThermParticle::operator=(aParticle);

  fPosDaughterFound = aParticle.fPosDaughterFound;
  fNegDaughterFound = aParticle.fNegDaughterFound;
  fBothDaughtersFound = aParticle.fBothDaughtersFound;
  fGoodLambda = aParticle.fGoodLambda;

  fPosDaughterPID = aParticle.fPosDaughterPID;
  fNegDaughterPID = aParticle.fNegDaughterPID;

  fMassPos = aParticle.fMassPos; 
  fTPos = aParticle.fTPos;
  fXPos = aParticle.fXPos;
  fYPos = aParticle.fYPos;
  fZPos = aParticle.fZPos;
  fEPos = aParticle.fEPos;
  fPxPos = aParticle.fPxPos;
  fPyPos = aParticle.fPyPos;
  fPzPos = aParticle.fPzPos;

  fMassNeg = aParticle.fMassNeg; 
  fTNeg = aParticle.fTNeg;
  fXNeg = aParticle.fXNeg;
  fYNeg = aParticle.fYNeg;
  fZNeg = aParticle.fZNeg;
  fENeg = aParticle.fENeg;
  fPxNeg = aParticle.fPxNeg;
  fPyNeg = aParticle.fPyNeg;
  fPzNeg = aParticle.fPzNeg;

  return *this;
}



//________________________________________________________________________________________________________________
ThermV0Particle::~ThermV0Particle()
{
  cout << "ThermV0Particle object is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void ThermV0Particle::LoadDaughter(ThermParticle* aDaughter)
{
  int tPID = aDaughter->GetPID();
  double tMass = aDaughter->GetMass();

  TLorentzVector* tFourPosition = aDaughter->GetFourPosition();
    double tT = tFourPosition->T();
    double tX = tFourPosition->X();
    double tY = tFourPosition->Y();
    double tZ = tFourPosition->Z();

  TLorentzVector* tFourMomentum = aDaughter->GetFourMomentum();
    double tE = tFourMomentum->E();
    double tPx = tFourMomentum->Px();
    double tPy = tFourMomentum->Py();
    double tPz = tFourMomentum->Pz();

  if(tPID > 0)
  {
    assert(!fPosDaughterFound);  //should not have already found positive daughter

    SetPosDaughterPID(tPID);

    fMassPos =tMass;

    fTPos = tT; 
    fXPos = tX; 
    fYPos = tY; 
    fZPos = tZ;

    fEPos = tE; 
    fPxPos = tPx;
    fPyPos = tPy;
    fPzPos = tPz;

    fPosDaughterFound = true;
  }

  else if(tPID < 0)
  {
    assert(!fNegDaughterFound);  //should not have already found negative daughter

    SetNegDaughterPID(tPID);

    fMassNeg =tMass;

    fTNeg = tT; 
    fXNeg = tX; 
    fYNeg = tY; 
    fZNeg = tZ;

    fENeg = tE; 
    fPxNeg = tPx;
    fPyNeg = tPy;
    fPzNeg = tPz;

    fNegDaughterFound = true;
  }

  else assert(0);

  if(fPosDaughterFound && fNegDaughterFound) fBothDaughtersFound = true;

}







