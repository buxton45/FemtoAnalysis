/*
 *  myAliFemtoSepCorrFctns.h
 *
 */

//Instead of creating a TH2F Average Separation CF
//This will create a TH2F Separation CF for all 8 points used in the AverageCF calculation

#ifndef MYALIFEMTOSEPCORRFCTNS_H
#define MYALIFEMTOSEPCORRFCTNS_H

#include "TH1D.h"
#include "TH2F.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

class myAliFemtoSepCorrFctns : public AliFemtoCorrFctn {
public:
  enum PairType {kPosPos = 0, kPosNeg = 1, kNegPos = 2, kNegNeg = 3, kTrackPos = 4, kTrackNeg = 5, kTrackTrack = 6};
  enum V0Type {kLambda = 0, kAntiLambda = 1, kK0Short = 2};
  myAliFemtoSepCorrFctns(const char* title, const int& nbinsX, const float& XLo, const float& XHi, const int& nbinsY, const float& SepLo, const float& SepHi);
  myAliFemtoSepCorrFctns(const myAliFemtoSepCorrFctns& aCorrFctn);
  virtual ~myAliFemtoSepCorrFctns();

  myAliFemtoSepCorrFctns& operator=(const myAliFemtoSepCorrFctns& aCorrFctn);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  //--used when both are V0s
  TH2F* NumeratorPosPos();
  TH2F* NumeratorPosNeg();
  TH2F* NumeratorNegPos();
  TH2F* NumeratorNegNeg();
  TH2F* DenominatorPosPos();
  TH2F* DenominatorPosNeg();
  TH2F* DenominatorNegPos();
  TH2F* DenominatorNegNeg();


  //--used when one V0 and one track
  TH2F* NumeratorTrackPos();
  TH2F* NumeratorTrackNeg();
  TH2F* DenominatorTrackPos();
  TH2F* DenominatorTrackNeg();

 //--used when both are tracks
  TH2F* NumeratorTrackTrack();
  TH2F* DenominatorTrackTrack();

  virtual TList* GetOutputList();
  void Write();

  void FillSepHisto(AliFemtoPair* pair, PairType fType, TH2F* aHisto, double aWeight);
  bool IsPionNSigma(float mom, float nsigmaTPCPi, float nsigmaTOFPi);
  bool IsProtonNSigma(float mom, float nsigmaTPCP, float nsigmaTOFP);
  bool IsK0Short(AliFemtoV0* aV0);

private:
  //--used when both are V0s
  TH2F* fNumeratorPosPos;          // numerator ++
  TH2F* fNumeratorPosNeg;          // numerator +-
  TH2F* fNumeratorNegPos;          // numerator -+
  TH2F* fNumeratorNegNeg;          // numerator --
  TH2F* fDenominatorPosPos;          // denominator ++
  TH2F* fDenominatorPosNeg;          // denominator +-
  TH2F* fDenominatorNegPos;          // denominator -+
  TH2F* fDenominatorNegNeg;          // denominator --

  //--used when one V0 and one track
  TH2F* fNumeratorTrackPos;          // numerator Track +
  TH2F* fNumeratorTrackNeg;          // numerator Track -
  TH2F* fDenominatorTrackPos;          // denominator Track +
  TH2F* fDenominatorTrackNeg;          // denominator Track -

 //--used when both are tracks
  TH2F* fNumeratorTrackTrack;          // numerator Track Track
  TH2F* fDenominatorTrackTrack;          // denominator Track Track


#ifdef __ROOT__
  ClassDef(myAliFemtoSepCorrFctns, 1)
#endif
};

inline  TH2F* myAliFemtoSepCorrFctns::NumeratorPosPos(){return fNumeratorPosPos;}
inline  TH2F* myAliFemtoSepCorrFctns::NumeratorPosNeg(){return fNumeratorPosNeg;}
inline  TH2F* myAliFemtoSepCorrFctns::NumeratorNegPos(){return fNumeratorNegPos;}
inline  TH2F* myAliFemtoSepCorrFctns::NumeratorNegNeg(){return fNumeratorNegNeg;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorPosPos(){return fDenominatorPosPos;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorPosNeg(){return fDenominatorPosNeg;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorNegPos(){return fDenominatorNegPos;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorNegNeg(){return fDenominatorNegNeg;}

inline  TH2F* myAliFemtoSepCorrFctns::NumeratorTrackPos(){return fNumeratorTrackPos;}
inline  TH2F* myAliFemtoSepCorrFctns::NumeratorTrackNeg(){return fNumeratorTrackNeg;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorTrackPos(){return fDenominatorTrackPos;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorTrackNeg(){return fDenominatorTrackNeg;}

inline  TH2F* myAliFemtoSepCorrFctns::NumeratorTrackTrack(){return fNumeratorTrackTrack;}
inline  TH2F* myAliFemtoSepCorrFctns::DenominatorTrackTrack(){return fDenominatorTrackTrack;}

#endif
