/*
 *  myAliFemtoAvgSepCorrFctn.h
 *
 */

#ifndef MYALIFEMTOAVGSEPCORRFCTN_H
#define MYALIFEMTOAVGSEPCORRFCTN_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

class myAliFemtoAvgSepCorrFctn : public AliFemtoCorrFctn {
public:
  enum PairType {kPosPos = 0, kPosNeg = 1, kNegPos = 2, kNegNeg = 3, kTrackPos = 4, kTrackNeg = 5, kTrackTrack = 6};
  enum V0Type {kLambda = 0, kAntiLambda = 1, kK0Short = 2};
  myAliFemtoAvgSepCorrFctn(const char* title, const int& nbins, const float& AvgSepLo, const float& AvgSepHi);
  myAliFemtoAvgSepCorrFctn(const myAliFemtoAvgSepCorrFctn& aCorrFctn);
  virtual ~myAliFemtoAvgSepCorrFctn();

  myAliFemtoAvgSepCorrFctn& operator=(const myAliFemtoAvgSepCorrFctn& aCorrFctn);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  //--used when both are V0s
  TH1D* NumeratorPosPos();
  TH1D* NumeratorPosNeg();
  TH1D* NumeratorNegPos();
  TH1D* NumeratorNegNeg();
  TH1D* DenominatorPosPos();
  TH1D* DenominatorPosNeg();
  TH1D* DenominatorNegPos();
  TH1D* DenominatorNegNeg();


  //--used when one V0 and one track
  TH1D* NumeratorTrackPos();
  TH1D* NumeratorTrackNeg();
  TH1D* DenominatorTrackPos();
  TH1D* DenominatorTrackNeg();

 //--used when both are tracks
  TH1D* NumeratorTrackTrack();
  TH1D* DenominatorTrackTrack();

  virtual TList* GetOutputList();
  void Write();

  double CalculateAvgSep(AliFemtoPair* pair, PairType fType);
  bool IsPionNSigma(float mom, float nsigmaTPCPi, float nsigmaTOFPi);
  bool IsProtonNSigma(float mom, float nsigmaTPCP, float nsigmaTOFP);
  bool IsK0Short(AliFemtoV0* aV0);

private:
  //--used when both are V0s
  TH1D* fNumeratorPosPos;          // numerator ++
  TH1D* fNumeratorPosNeg;          // numerator +-
  TH1D* fNumeratorNegPos;          // numerator -+
  TH1D* fNumeratorNegNeg;          // numerator --
  TH1D* fDenominatorPosPos;          // denominator ++
  TH1D* fDenominatorPosNeg;          // denominator +-
  TH1D* fDenominatorNegPos;          // denominator -+
  TH1D* fDenominatorNegNeg;          // denominator --

  //--used when one V0 and one track
  TH1D* fNumeratorTrackPos;          // numerator Track +
  TH1D* fNumeratorTrackNeg;          // numerator Track -
  TH1D* fDenominatorTrackPos;          // denominator Track +
  TH1D* fDenominatorTrackNeg;          // denominator Track -

 //--used when both are tracks
  TH1D* fNumeratorTrackTrack;          // numerator Track Track
  TH1D* fDenominatorTrackTrack;          // denominator Track Track


#ifdef __ROOT__
  ClassDef(myAliFemtoAvgSepCorrFctn, 1)
#endif
};

inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorPosPos(){return fNumeratorPosPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorPosNeg(){return fNumeratorPosNeg;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorNegPos(){return fNumeratorNegPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorNegNeg(){return fNumeratorNegNeg;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorPosPos(){return fDenominatorPosPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorPosNeg(){return fDenominatorPosNeg;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorNegPos(){return fDenominatorNegPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorNegNeg(){return fDenominatorNegNeg;}

inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorTrackPos(){return fNumeratorTrackPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorTrackNeg(){return fNumeratorTrackNeg;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorTrackPos(){return fDenominatorTrackPos;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorTrackNeg(){return fDenominatorTrackNeg;}

inline  TH1D* myAliFemtoAvgSepCorrFctn::NumeratorTrackTrack(){return fNumeratorTrackTrack;}
inline  TH1D* myAliFemtoAvgSepCorrFctn::DenominatorTrackTrack(){return fDenominatorTrackTrack;}

#endif
