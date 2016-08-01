///////////////////////////////////////////////////////////////////////////
// ThermLambdaParticle:                                                  //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMLAMBDAPARTICLE
#define THERMLAMBDAPARTICLE

#include "ThermParticle.h"
class ThermParticle;

class ThermLambdaParticle : public ThermParticle {

public:
  ThermLambdaParticle(ParticleType aType);
  ThermLambdaParticle(const ThermLambdaParticle& aParticle);
  ThermLambdaParticle& operator=(ThermLambdaParticle& aParticle);

  virtual ~ThermLambdaParticle();


  //inline
  void SetPosDaughterPID(int aPID);
  void SetNegDaughterPID(int aPID);

  int GetPosDaughterPID();
  int GetNegDaughterPID();

private:
  int fPosDaughterPID;
  int fNegDaughterPID;

  double fMassPos;
  double fTPos, fXPos, fYPos, fZPos;
  double fEPos, fPxPos, fPyPos, fPzPos;

  double fMassNeg;
  double fTNeg, fXNeg, fYNeg, fZNeg;
  double fENeg, fPxNeg, fPyNeg, fPzNeg;


#ifdef __ROOT__
  ClassDef(ThermLambdaParticle, 1)
#endif
};


//inline stuff
inline void ThermLambdaParticle::SetPosDaughterPID(int aPID) {fPosDaughterPID = aPID;}
inline void ThermLambdaParticle::SetNegDaughterPID(int aPID) {fNegDaughterPID = aPID;}

inline int ThermLambdaParticle::GetPosDaughterPID() {return fPosDaughterPID;}
inline int ThermLambdaParticle::GetNegDaughterPID() {return fNegDaughterPID;}

#endif
