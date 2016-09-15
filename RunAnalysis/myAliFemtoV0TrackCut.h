/// \class AliFemtoV0TrackCut

#ifndef MYALIFEMTOV0TRACKCUT_H
#define MYALIFEMTOV0TRACKCUT_H

#include <TH1D.h>
#include "AliFemtoTrackCut.h"

//  Purpose:  extend applicability of AliFemtoV0TrackCut class
//	Specifically:
//		1.  allow one to find K0s (not only lambdas)
//		2.  Change SetMinDaughtersToPrimVertex to accept two arguments,
//		    corresponding to cuts for the positive and negative daughters, separately (DONE in vAN-20141013)


class myAliFemtoV0TrackCut : public AliFemtoParticleCut 
{
  public:
  enum V0Type {kLambda = 0, kAntiLambda=1, kK0s=2, kAll=99, kLambdaMC=101, kAntiLambdaMC=102};
  typedef enum V0Type AliFemtoV0Type;


  myAliFemtoV0TrackCut();
  myAliFemtoV0TrackCut(const myAliFemtoV0TrackCut& cut);  //copy constructor - 30 June 2015
  virtual ~myAliFemtoV0TrackCut();

  myAliFemtoV0TrackCut& operator=(const myAliFemtoV0TrackCut& cut);  //assignment operator - 30 June 2015

  virtual bool Pass(const AliFemtoV0* aV0);

  virtual AliFemtoString Report();
  virtual TList *ListSettings();
  virtual AliFemtoParticleType Type(){return hbtV0;}
  
  void SetInvariantMassLambda(double,double);
  void SetInvariantMassK0s(double,double);
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
  void SetUseSimpleMisIDCut(bool aUse);
  void SetInvMassReject(AliFemtoV0Type aV0Type, double aInvMassMin, double aInvMassMax);
  void SetBuildMisIDHistograms(bool aBuild);
  void SetMisIDHisto(AliFemtoV0Type aMisIDV0Type, const int& nbins, const float& aInvMassMin, const float& aInvMassMax);  //allows control over binning
  void SetDefaultMisIDHistos();
  TObjArray *GetMisIDHistos();

  void SetCalculatePurity(bool aCalc);
  void SetLooseInvMassCut(bool aUseCut, double aInvMassMin, double aInvMassMax);
  void SetPurityHisto(const char* title, const int& nbins, const float& aInvMassMin, const float& aInvMassMax);
  TH1D *GetPurityHisto();
  //-----03/11/2014------------------
  bool IsMisIDK0s(const AliFemtoV0* aV0);
  bool IsMisIDLambda(const AliFemtoV0* aV0);
  bool IsMisIDAntiLambda(const AliFemtoV0* aV0);
  //-----06/11/2014------------------
  void SetUseLooseInvMassCut(bool aUse);
  
 private:   // here are the quantities I want to cut on...

  double            fInvMassLambdaMin;   //invariant mass lambda min
  double            fInvMassLambdaMax;   //invariant mass lambda max
  double            fInvMassK0sMin;   //invariant mass K0 min
  double            fInvMassK0sMax;   //invariant mass K0 max
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
  //For more control over the misidentification cut, use AliFemtoV0TrackCutNSigmaFilter class
  bool fRemoveMisidentified;         // attempt to remove V0 candidates (K0s, Lambda and AntiLambda) which are misidentified
                                     //   i.e. check if Lambda or AntiLambda could be misidentified K0s,
                                     //        or if K0s could be misidentified Lambda or AntiLambda
                                     // Uses methods IsMisIDK0s, IsMisIDLambda, IsMisIDAntiLambda
  bool fUseSimpleMisIDCut;	     // If set to true, the misidentification cut is based soley off of the invariant mass
                                     // If set to false, it also considers the NSigma of the daughter particles when cutting
  bool fBuildMisIDHistograms;

  double fInvMassRejectLambdaMin, fInvMassRejectLambdaMax;
  double fInvMassRejectAntiLambdaMin, fInvMassRejectAntiLambdaMax;
  double fInvMassRejectK0sMin, fInvMassRejectK0sMax;
  TH1D *fLambdaMassOfMisIDV0;          // Mass assuming Lambda hypothesis for V0s rejected by misidentification cut
  TH1D *fAntiLambdaMassOfMisIDV0;      // Mass assuming AntiLambda hypothesis for V0s rejected by misidentification cut
  TH1D *fK0sMassOfMisIDV0;         // Mass asumming K0s hypothesis for V0s rejected by misidentification cut
  bool fCalculatePurity;
  double fLooseInvMassMin;
  double fLooseInvMassMax;
  TH1D *fPurity;   //InvMass histogram to be used in purity calculation
  //-----06/11/2014------------------
  bool fUseLooseInvMassCut;

#ifdef __ROOT__ 
  ClassDef(myAliFemtoV0TrackCut, 1)
#endif

};


#endif

