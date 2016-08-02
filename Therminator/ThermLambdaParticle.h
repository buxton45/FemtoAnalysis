///////////////////////////////////////////////////////////////////////////
// ThermLambdaParticle:                                                  //
///////////////////////////////////////////////////////////////////////////

#ifndef THERMLAMBDAPARTICLE
#define THERMLAMBDAPARTICLE

#include "ThermParticle.h"
class ThermParticle;

class ThermLambdaParticle : public ThermParticle {

public:
  ThermLambdaParticle();
  ThermLambdaParticle(const ThermLambdaParticle& aParticle);
  ThermLambdaParticle& operator=(ThermLambdaParticle& aParticle);

  virtual ~ThermLambdaParticle();

  void LoadDaughter(ThermParticle* aDaughter);

  //inline
  void SetPosDaughterPDGType(int aPID);
  void SetNegDaughterPDGType(int aPID);

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
  ClassDef(ThermLambdaParticle, 1)
#endif
};


//inline stuff
inline void ThermLambdaParticle::SetPosDaughterPDGType(int aPID) {fPosDaughterPDGType = GetPDGType(aPID);}
inline void ThermLambdaParticle::SetNegDaughterPDGType(int aPID) {fNegDaughterPDGType = GetPDGType(aPID);}

inline void ThermLambdaParticle::SetPosDaughterPID(int aPID) {fPosDaughterPID = aPID; SetPosDaughterPDGType(fPosDaughterPID);}
inline void ThermLambdaParticle::SetNegDaughterPID(int aPID) {fNegDaughterPID = aPID; SetPosDaughterPDGType(fNegDaughterPID);}

inline int ThermLambdaParticle::GetPosDaughterPID() {return fPosDaughterPID;}
inline int ThermLambdaParticle::GetNegDaughterPID() {return fNegDaughterPID;}

inline bool ThermLambdaParticle::BothDaughtersFound() {return fBothDaughtersFound;}

#endif
