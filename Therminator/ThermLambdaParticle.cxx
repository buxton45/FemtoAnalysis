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
ThermLambdaParticle::ThermLambdaParticle() :
  ThermParticle(),
  fPosDaughterFound(false),
  fNegDaughterFound(false),
  fBothDaughtersFound(false),
  fGoodLambda(false),
  fPosDaughterPID(0),
  fNegDaughterPID(0),
  fPosDaughterPDGType(kPDGNull),
  fNegDaughterPDGType(kPDGNull),

  fMassPos(0), fTPos(0), fXPos(0), fYPos(0), fZPos(0), fEPos(0), fPxPos(0), fPyPos(0), fPzPos(0),
  fMassNeg(0), fTNeg(0), fXNeg(0), fYNeg(0), fZNeg(0), fENeg(0), fPxNeg(0), fPyNeg(0), fPzNeg(0)
{

}

//________________________________________________________________________________________________________________
ThermLambdaParticle::ThermLambdaParticle(const ThermLambdaParticle& aParticle) :
  ThermParticle(aParticle),
  fPosDaughterFound(aParticle.fPosDaughterFound),
  fNegDaughterFound(aParticle.fNegDaughterFound),
  fBothDaughtersFound(aParticle.fBothDaughtersFound),
  fGoodLambda(aParticle.fGoodLambda),
  fPosDaughterPID(aParticle.fPosDaughterPID),
  fNegDaughterPID(aParticle.fNegDaughterPID),
  fPosDaughterPDGType(aParticle.fPosDaughterPDGType),
  fNegDaughterPDGType(aParticle.fNegDaughterPDGType),

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

  fPosDaughterFound = aParticle.fPosDaughterFound;
  fNegDaughterFound = aParticle.fNegDaughterFound;
  fBothDaughtersFound = aParticle.fBothDaughtersFound;
  fGoodLambda = aParticle.fGoodLambda;

  fPosDaughterPID = aParticle.fPosDaughterPID;
  fNegDaughterPID = aParticle.fNegDaughterPID;

  fPosDaughterPDGType = aParticle.fPosDaughterPDGType;
  fNegDaughterPDGType = aParticle.fNegDaughterPDGType;

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


//________________________________________________________________________________________________________________
void ThermLambdaParticle::LoadDaughter(ThermParticle* aDaughter)
{
  int tPID = aDaughter->GetPID();
  ParticlePDGType tPDGType = aDaughter->GetParticlePDGType();
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

    SetPosDaughterPID(tPID); //also sets fPosDaughterPDGType

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

    SetNegDaughterPID(tPID); //also sets fNegDaughterPDGType

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







