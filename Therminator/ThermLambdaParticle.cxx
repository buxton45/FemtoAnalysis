///////////////////////////////////////////////////////////////////////////
// ThermLambdaParticle:                                                  //
///////////////////////////////////////////////////////////////////////////


#include "ThermLambdaParticle.h"

#ifdef __ROOT__
ClassImp(ThermLambdaParticle)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermLambdaParticle::ThermLambdaParticle(ParticleType aType) :
  ThermParticle(aType),
  fPosDaughterPID(0),
  fNegDaughterPID(0),

  fMassPos(0), fTPos(0), fXPos(0), fYPos(0), fZPos(0), fEPos(0), fPxPos(0), fPyPos(0), fPzPos(0),
  fMassNeg(0), fTNeg(0), fXNeg(0), fYNeg(0), fZNeg(0), fENeg(0), fPxNeg(0), fPyNeg(0), fPzNeg(0)
{

}

//________________________________________________________________________________________________________________
ThermLambdaParticle::ThermLambdaParticle(const ThermLambdaParticle& aParticle) :
  ThermParticle(aParticle),
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
ThermLambdaParticle& ThermLambdaParticle::operator=(ThermLambdaParticle& aParticle)
{
  if(this == &aParticle) return *this;

  ThermParticle::operator=(aParticle);

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
ThermLambdaParticle::~ThermLambdaParticle()
{
  cout << "ThermLambdaParticle object is being deleted!!!" << endl;
}
