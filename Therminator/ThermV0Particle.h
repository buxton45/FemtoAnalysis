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
  void SetPosDaughterPDGType(int aPID);
  void SetNegDaughterPDGType(int aPID);
  ParticlePDGType GetPosDaughterPDGType();
  ParticlePDGType GetNegDaughterPDGType();

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

  ParticlePDGType fPosDaughterPDGType;
  ParticlePDGType fNegDaughterPDGType;

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
inline void ThermV0Particle::SetPosDaughterPDGType(int aPID) {fPosDaughterPDGType = GetPDGType(aPID);}
inline void ThermV0Particle::SetNegDaughterPDGType(int aPID) {fNegDaughterPDGType = GetPDGType(aPID);}
inline ParticlePDGType ThermV0Particle::GetPosDaughterPDGType() {return fPosDaughterPDGType;}
inline ParticlePDGType ThermV0Particle::GetNegDaughterPDGType() {return fNegDaughterPDGType;}

inline void ThermV0Particle::SetPosDaughterPID(int aPID) {fPosDaughterPID = aPID; SetPosDaughterPDGType(fPosDaughterPID);}
inline void ThermV0Particle::SetNegDaughterPID(int aPID) {fNegDaughterPID = aPID; SetPosDaughterPDGType(fNegDaughterPID);}
inline int ThermV0Particle::GetPosDaughterPID() {return fPosDaughterPID;}
inline int ThermV0Particle::GetNegDaughterPID() {return fNegDaughterPID;}

inline bool ThermV0Particle::BothDaughtersFound() {return fBothDaughtersFound;}

#endif
