///////////////////////////////////////////////////////////////////////////
// ThermV0Particle:                                                  //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMV0PARTICLE
#define THERMV0PARTICLE

#include "ThermParticle.h"
class ThermParticle;

class ThermV0Particle : public ThermParticle {

public:
  ThermV0Particle();
  ThermV0Particle(const ThermV0Particle& aParticle);
  ThermV0Particle& operator=(ThermV0Particle& aParticle);

  virtual ~ThermV0Particle();

  void LoadDaughter(ThermParticle* aDaughter);

  //inline

  void SetPosDaughterPID(int aPID);
  void SetNegDaughterPID(int aPID);
  int GetPosDaughterPID();
  int GetNegDaughterPID();

  bool BothDaughtersFound();

private:
  bool fPosDaughterFound;
  bool fNegDaughterFound;
  bool fBothDaughtersFound;

  bool fGoodLambda;

  int fPosDaughterPID;
  int fNegDaughterPID;

  double fMassPos;
  double fTPos, fXPos, fYPos, fZPos;
  double fEPos, fPxPos, fPyPos, fPzPos;

  double fMassNeg;
  double fTNeg, fXNeg, fYNeg, fZNeg;
  double fENeg, fPxNeg, fPyNeg, fPzNeg;


#ifdef __ROOT__
  ClassDef(ThermV0Particle, 1)
#endif
};


//inline stuff

inline void ThermV0Particle::SetPosDaughterPID(int aPID) {fPosDaughterPID = aPID;}
inline void ThermV0Particle::SetNegDaughterPID(int aPID) {fNegDaughterPID = aPID;}
inline int ThermV0Particle::GetPosDaughterPID() {return fPosDaughterPID;}
inline int ThermV0Particle::GetNegDaughterPID() {return fNegDaughterPID;}

inline bool ThermV0Particle::BothDaughtersFound() {return fBothDaughtersFound;}

#endif
