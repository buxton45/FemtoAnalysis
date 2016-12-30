///////////////////////////////////////////////////////////////////////////
//                                                                       //
// AliFemtoESDTrackCut: A basic track cut that used information from     //
// ALICE ESD to accept or reject the track.                              //  
// Enables the selection on charge, transverse momentum, rapidity,       //
// pid probabilities, number of ITS and TPC clusters                     //
// Author: Marek Chojnacki (WUT), mchojnacki@knf.pw.edu.pl               //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

//  Purpose:  extend applicability of AliFemtoV0TrackCut class
//	Specifically:
//		1.  allow one to find K0s (not only lambdas)
//		2.  Change SetMinDaughtersToPrimVertex to accept two arguments,
//		    corresponding to cuts for the positive and negative daughters, separately (DONE in vAN-20141013)

#ifndef MYALIFEMTOV0TRACKCUT_H
#define MYALIFEMTOV0TRACKCUT_H

#include <TH1F.h>
#include "AliFemtoTrackCut.h"
#include "TVectorD.h"

class myAliFemtoV0TrackCut : public AliFemtoParticleCut 
{
  public:
  enum V0Type {kLambda = 0, kAntiLambda=1, kK0Short=2, kAll=99, kLambdaMC=101, kAntiLambdaMC=102};
  typedef enum V0Type AliFemtoV0Type;


  myAliFemtoV0TrackCut();
  virtual ~myAliFemtoV0TrackCut();

  virtual bool Pass(const AliFemtoV0* aV0);

  virtual AliFemtoString Report();
  virtual TList *ListSettings();
  virtual AliFemtoParticleType Type(){return hbtV0;}
  
  void SetInvariantMassLambda(double,double);
  void SetInvariantMassK0Short(double,double);
  void SetMinDaughtersToPrimVertex(double,double);
  void SetMaxDcaV0Daughters(double);
  void SetMaxDcaV0(double);
  void SetMaxCosPointingAngle(double);
  void SetMaxV0DecayLength(double);
  void SetParticleType(short);
  void SetEta(double);
  void SetPt(double,double);
  void SetEtaDaughters(float);
  void SetTPCnclsDaughters(int);
  void SetNdofDaughters(int);
  void SetStatusDaughters(unsigned long);
  void SetPtPosDaughter(float,float);
  void SetPtNegDaughter(float,float);
  void SetOnFlyStatus(bool);
  void SetMinAvgSeparation(double);

  //----n sigma----
  bool IsKaonTPCdEdxNSigma(float mom, float nsigmaK);
  bool IsKaonTOFNSigma(float mom, float nsigmaK);
  bool IsKaonNSigma(float mom, float nsigmaTPCK, float nsigmaTOFK);
  bool IsPionNSigma(float mom, float nsigmaTPCPi, float nsigmaTOFPi);
  bool IsProtonNSigma(float mom, float nsigmaTPCP, float nsigmaTOFP);

  //-----23/10/2014------------------
  void SetRemoveMisidentified(bool aRemove);
  void SetInvMassMisidentified(double aInvMassMin, double aInvMassMax);
  void SetMisIDHisto(char* title, const int& nbins, const float& aInvMassMin, const float& aInvMassMax);
  TH1F *GetMisIDHisto();
  void SetCalculatePurity(bool aCalc);
  void SetPurityRange(double aInvMassMin, double aInvMassMax);
  void SetPurityHisto(char* title, const int& nbins, const float& aInvMassMin, const float& aInvMassMax);
  TH1F *GetPurityHisto();
  
 private:   // here are the quantities I want to cut on...

  double            fInvMassLambdaMin;   //invariant mass lambda min
  double            fInvMassLambdaMax;   //invariant mass lambda max
  double            fInvMassK0ShortMin;   //invariant mass K0 min
  double            fInvMassK0ShortMax;   //invariant mass K0 max
  double            fMinDcaDaughterPosToVert; //DCA of positive daughter to primary vertex
  double            fMinDcaDaughterNegToVert; //DCA of negative daughter to primary vertex
  double            fMaxDcaV0Daughters;     //Max DCA of v0 daughters at Decay vertex
  double            fMaxDcaV0;
  double            fMaxDecayLength;
  
  double            fMaxCosPointingAngle;
  short             fParticleType; //0-lambda
  double            fEta;
  double            fPtMin;
  double            fPtMax;
  bool              fOnFlyStatus;

  float fMaxEtaDaughters;			            // Eta of positive daughter
  int   fTPCNclsDaughters;			            // No. of cls of pos daughter
  int   fNdofDaughters;			                    // No. of degrees of freedom of the pos. daughter track
  unsigned long fStatusDaughters;			    // Status (tpc refit, its refit...)
  float fPtMinPosDaughter;
  float fPtMaxPosDaughter;
  float fPtMinNegDaughter;
  float fPtMaxNegDaughter;
  double fMinAvgSepDaughters;

  //-----23/10/2014------------------
  bool fRemoveMisidentified;   //remove POI candidates that fulfill the misidentified mass hypothesis within a given Minv range
  double fInvMassMisidentifiedMin;
  double fInvMassMisidentifiedMax;
  TH1F *fMisIDHisto;
  bool fCalculatePurity;
  double fPurityInvMassMin;
  double fPurityInvMassMax;
  TH1F *fPurity;   //InvMass histogram to be used in purity calculation

#ifdef __ROOT__ 
  ClassDef(myAliFemtoV0TrackCut, 1)
#endif

};


#endif
