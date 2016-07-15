///////////////////////////////////////////////////////////////////////////
//                                                                       //
// myAliFemtoSepCorrFctns:                                             //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "myAliFemtoSepCorrFctns.h"
//#include "AliFemtoHisto.h"
#include <cstdio>

#ifdef __ROOT__
ClassImp(myAliFemtoSepCorrFctns)
#endif

//____________________________
myAliFemtoSepCorrFctns::myAliFemtoSepCorrFctns(const char* title, const int& nbinsX, const float& XLo, const float& XHi, const int& nbinsY, const float& SepLo, const float& SepHi):
  fNumeratorPosPos(0),
  fNumeratorPosNeg(0),
  fNumeratorNegPos(0),
  fNumeratorNegNeg(0),
  fNumeratorTrackPos(0),
  fNumeratorTrackNeg(0),
  fNumeratorTrackTrack(0),
  fDenominatorPosPos(0),
  fDenominatorPosNeg(0),
  fDenominatorNegPos(0),
  fDenominatorNegNeg(0),
  fDenominatorTrackPos(0),
  fDenominatorTrackNeg(0),
  fDenominatorTrackTrack(0)

{
  // set up numerators
  char tTitNumPosPos[101] = "NumPosPos";
  strncat(tTitNumPosPos,title, 100);
  fNumeratorPosPos = new TH2F(tTitNumPosPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumPosNeg[101] = "NumPosNeg";
  strncat(tTitNumPosNeg,title, 100);
  fNumeratorPosNeg = new TH2F(tTitNumPosNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumNegPos[101] = "NumNegPos";
  strncat(tTitNumNegPos,title, 100);
  fNumeratorNegPos = new TH2F(tTitNumNegPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumNegNeg[101] = "NumNegNeg";
  strncat(tTitNumNegNeg,title, 100);
  fNumeratorNegNeg = new TH2F(tTitNumNegNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumTrackPos[101] = "NumTrackPos";
  strncat(tTitNumTrackPos,title, 100);
  fNumeratorTrackPos = new TH2F(tTitNumTrackPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumTrackNeg[101] = "NumTrackNeg";
  strncat(tTitNumTrackNeg,title, 100);
  fNumeratorTrackNeg = new TH2F(tTitNumTrackNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitNumTrackTrack[101] = "NumTrackTrack";
  strncat(tTitNumTrackTrack,title, 100);
  fNumeratorTrackTrack = new TH2F(tTitNumTrackTrack,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);

  // set up denominators
  char tTitDenPosPos[101] = "DenPosPos";
  strncat(tTitDenPosPos,title, 100);
  fDenominatorPosPos = new TH2F(tTitDenPosPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenPosNeg[101] = "DenPosNeg";
  strncat(tTitDenPosNeg,title, 100);
  fDenominatorPosNeg = new TH2F(tTitDenPosNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenNegPos[101] = "DenNegPos";
  strncat(tTitDenNegPos,title, 100);
  fDenominatorNegPos = new TH2F(tTitDenNegPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenNegNeg[101] = "DenNegNeg";
  strncat(tTitDenNegNeg,title, 100);
  fDenominatorNegNeg = new TH2F(tTitDenNegNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenTrackPos[101] = "DenTrackPos";
  strncat(tTitDenTrackPos,title, 100);
  fDenominatorTrackPos = new TH2F(tTitDenTrackPos,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenTrackNeg[101] = "DenTrackNeg";
  strncat(tTitDenTrackNeg,title, 100);
  fDenominatorTrackNeg = new TH2F(tTitDenTrackNeg,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);
  char tTitDenTrackTrack[101] = "DenTrackTrack";
  strncat(tTitDenTrackTrack,title, 100);
  fDenominatorTrackTrack = new TH2F(tTitDenTrackTrack,title,nbinsX,XLo,XHi,nbinsY,SepLo,SepHi);


  // this next bit is unfortunately needed so that we can have many histos of same "title"
  // it is neccessary if we typedef TH2F to TH2F (which we do)
  //fNumerator->SetDirectory(0);
  //fDenominator->SetDirectory(0);

  // to enable error bar calculation...
  fNumeratorPosPos->Sumw2();
  fNumeratorPosNeg->Sumw2();
  fNumeratorNegPos->Sumw2();
  fNumeratorNegNeg->Sumw2();
  fNumeratorTrackPos->Sumw2();
  fNumeratorTrackNeg->Sumw2();
  fNumeratorTrackTrack->Sumw2();

  fDenominatorPosPos->Sumw2();
  fDenominatorPosNeg->Sumw2();
  fDenominatorNegPos->Sumw2();
  fDenominatorNegNeg->Sumw2();
  fDenominatorTrackPos->Sumw2();
  fDenominatorTrackNeg->Sumw2();
  fDenominatorTrackTrack->Sumw2();

}

//____________________________
myAliFemtoSepCorrFctns::myAliFemtoSepCorrFctns(const myAliFemtoSepCorrFctns& aCorrFctn) :
  AliFemtoCorrFctn(),
  fNumeratorPosPos(0),
  fNumeratorPosNeg(0),
  fNumeratorNegPos(0),
  fNumeratorNegNeg(0),
  fNumeratorTrackPos(0),
  fNumeratorTrackNeg(0),
  fNumeratorTrackTrack(0),
  fDenominatorPosPos(0),
  fDenominatorPosNeg(0),
  fDenominatorNegPos(0),
  fDenominatorNegNeg(0),
  fDenominatorTrackPos(0),
  fDenominatorTrackNeg(0),
  fDenominatorTrackTrack(0)

{
  // copy constructor
  fNumeratorPosPos = new TH2F(*aCorrFctn.fNumeratorPosPos);
  fNumeratorPosNeg = new TH2F(*aCorrFctn.fNumeratorPosNeg);
  fNumeratorNegPos = new TH2F(*aCorrFctn.fNumeratorNegPos);
  fNumeratorNegNeg = new TH2F(*aCorrFctn.fNumeratorNegNeg);
  fNumeratorTrackPos = new TH2F(*aCorrFctn.fNumeratorTrackPos);
  fNumeratorTrackNeg = new TH2F(*aCorrFctn.fNumeratorTrackNeg);
  fNumeratorTrackTrack = new TH2F(*aCorrFctn.fNumeratorTrackTrack);

  fDenominatorPosPos = new TH2F(*aCorrFctn.fDenominatorPosPos);
  fDenominatorPosNeg = new TH2F(*aCorrFctn.fDenominatorPosNeg);
  fDenominatorNegPos = new TH2F(*aCorrFctn.fDenominatorNegPos);
  fDenominatorNegNeg = new TH2F(*aCorrFctn.fDenominatorNegNeg);
  fDenominatorTrackPos = new TH2F(*aCorrFctn.fDenominatorTrackPos);
  fDenominatorTrackNeg = new TH2F(*aCorrFctn.fDenominatorTrackNeg);
  fDenominatorTrackTrack = new TH2F(*aCorrFctn.fDenominatorTrackTrack);
}
//____________________________
myAliFemtoSepCorrFctns::~myAliFemtoSepCorrFctns(){
  // destructor
  delete fNumeratorPosPos;
  delete fNumeratorPosNeg;
  delete fNumeratorNegPos;
  delete fNumeratorNegNeg;
  delete fNumeratorTrackPos;
  delete fNumeratorTrackNeg;
  delete fNumeratorTrackTrack;

  delete fDenominatorPosPos;
  delete fDenominatorPosNeg;
  delete fDenominatorNegPos;
  delete fDenominatorNegNeg;
  delete fDenominatorTrackPos;
  delete fDenominatorTrackNeg;
  delete fDenominatorTrackTrack;
}
//_________________________
myAliFemtoSepCorrFctns& myAliFemtoSepCorrFctns::operator=(const myAliFemtoSepCorrFctns& aCorrFctn)
{
  // assignment operator
  if (this == &aCorrFctn)
    return *this;

  if (fNumeratorPosPos) delete fNumeratorPosPos;
  fNumeratorPosPos = new TH2F(*aCorrFctn.fNumeratorPosPos);
  if (fNumeratorPosNeg) delete fNumeratorPosNeg;
  fNumeratorPosNeg = new TH2F(*aCorrFctn.fNumeratorPosNeg);
  if (fNumeratorNegPos) delete fNumeratorNegPos;
  fNumeratorNegPos = new TH2F(*aCorrFctn.fNumeratorNegPos);
  if (fNumeratorNegNeg) delete fNumeratorNegNeg;
  fNumeratorNegNeg = new TH2F(*aCorrFctn.fNumeratorNegNeg);
  if (fNumeratorTrackPos) delete fNumeratorTrackPos;
  fNumeratorTrackPos = new TH2F(*aCorrFctn.fNumeratorTrackPos);
  if (fNumeratorTrackNeg) delete fNumeratorTrackNeg;
  fNumeratorTrackNeg = new TH2F(*aCorrFctn.fNumeratorTrackNeg);
  if (fNumeratorTrackTrack) delete fNumeratorTrackTrack;
  fNumeratorTrackTrack = new TH2F(*aCorrFctn.fNumeratorTrackTrack);

  if (fDenominatorPosPos) delete fDenominatorPosPos;
  fDenominatorPosPos = new TH2F(*aCorrFctn.fDenominatorPosPos);
  if (fDenominatorPosNeg) delete fDenominatorPosNeg;
  fDenominatorPosNeg = new TH2F(*aCorrFctn.fDenominatorPosNeg);
  if (fDenominatorNegPos) delete fDenominatorNegPos;
  fDenominatorNegPos = new TH2F(*aCorrFctn.fDenominatorNegPos);
  if (fDenominatorNegNeg) delete fDenominatorNegNeg;
  fDenominatorNegNeg = new TH2F(*aCorrFctn.fDenominatorNegNeg);
  if (fDenominatorTrackPos) delete fDenominatorTrackPos;
  fDenominatorTrackPos = new TH2F(*aCorrFctn.fDenominatorTrackPos);
  if (fDenominatorTrackNeg) delete fDenominatorTrackNeg;
  fDenominatorTrackNeg = new TH2F(*aCorrFctn.fDenominatorTrackNeg);
  if (fDenominatorTrackTrack) delete fDenominatorTrackTrack;
  fDenominatorTrackTrack = new TH2F(*aCorrFctn.fDenominatorTrackTrack);

  return *this;
}

//_________________________
void myAliFemtoSepCorrFctns::Finish(){
  // here is where we should normalize, fit, etc...
  // we should NOT Draw() the histos (as I had done it below),
  // since we want to insulate ourselves from root at this level
  // of the code.  Do it instead at root command line with browser.
  //  fNumerator->Draw();
  //fDenominator->Draw();
}

//____________________________
AliFemtoString myAliFemtoSepCorrFctns::Report(){
  // construct report
  string stemp = "Sep Correlation Function Report:\n";
  char ctemp[100];
  snprintf(ctemp , 100, "Number of entries in fNumeratorPosPos:\t%E\n",fNumeratorPosPos->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in fDenominatorPosPos:\t%E\n",fDenominatorPosPos->GetEntries());
  stemp += ctemp;
  //  stemp += mCoulombWeight->Report();
  AliFemtoString returnThis = stemp;
  return returnThis;
}
//____________________________
void myAliFemtoSepCorrFctns::AddRealPair(AliFemtoPair* pair){
  // add true pair
  if(fPairCut)
    if (!fPairCut->Pass(pair)) return;

  double weight = 1.0;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      FillSepHisto(pair,kPosPos,fNumeratorPosPos,weight);
      FillSepHisto(pair,kPosNeg,fNumeratorPosNeg,weight);
      FillSepHisto(pair,kNegPos,fNumeratorNegPos,weight);
      FillSepHisto(pair,kNegNeg,fNumeratorNegNeg,weight);
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      FillSepHisto(pair,kTrackPos,fNumeratorTrackPos,weight);
      FillSepHisto(pair,kTrackNeg,fNumeratorTrackNeg,weight);
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      FillSepHisto(pair,kTrackTrack,fNumeratorTrackTrack,weight);
    } 
}

//____________________________
void myAliFemtoSepCorrFctns::AddMixedPair(AliFemtoPair* pair){
  // add mixed (background) pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;
  
  double weight = 1.0;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      FillSepHisto(pair,kPosPos,fDenominatorPosPos,weight);
      FillSepHisto(pair,kPosNeg,fDenominatorPosNeg,weight);
      FillSepHisto(pair,kNegPos,fDenominatorNegPos,weight);
      FillSepHisto(pair,kNegNeg,fDenominatorNegNeg,weight);
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      FillSepHisto(pair,kTrackPos,fDenominatorTrackPos,weight);
      FillSepHisto(pair,kTrackNeg,fDenominatorTrackNeg,weight);
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      FillSepHisto(pair,kTrackTrack,fDenominatorTrackTrack,weight);
    }
}
//____________________________
void myAliFemtoSepCorrFctns::Write(){
  // Write out neccessary objects
  fNumeratorPosPos->Write();
  fNumeratorPosNeg->Write();
  fNumeratorNegPos->Write();
  fNumeratorNegNeg->Write();
  fNumeratorTrackPos->Write();
  fNumeratorTrackNeg->Write();
  fNumeratorTrackTrack->Write();

  fDenominatorPosPos->Write();
  fDenominatorPosNeg->Write();
  fDenominatorNegPos->Write();
  fDenominatorNegNeg->Write();
  fDenominatorTrackPos->Write();
  fDenominatorTrackNeg->Write();
  fDenominatorTrackTrack->Write();
}
//______________________________
TList* myAliFemtoSepCorrFctns::GetOutputList()
{
  // Prepare the list of objects to be written to the output
  TList *tOutputList = new TList();

  tOutputList->Add(fNumeratorPosPos);
  tOutputList->Add(fNumeratorPosNeg);
  tOutputList->Add(fNumeratorNegPos);
  tOutputList->Add(fNumeratorNegNeg);
  tOutputList->Add(fNumeratorTrackPos);
  tOutputList->Add(fNumeratorTrackNeg);
  tOutputList->Add(fNumeratorTrackTrack);

  tOutputList->Add(fDenominatorPosPos);
  tOutputList->Add(fDenominatorPosNeg);
  tOutputList->Add(fDenominatorNegPos);
  tOutputList->Add(fDenominatorNegNeg);
  tOutputList->Add(fDenominatorTrackPos);
  tOutputList->Add(fDenominatorTrackNeg);
  tOutputList->Add(fDenominatorTrackTrack);

  return tOutputList;
}

//______________________________
void myAliFemtoSepCorrFctns::FillSepHisto(AliFemtoPair* pair, PairType fType, TH2F* aHisto, double aWeight)
{
  //Note:  Previously, I thought there was an ambiguity in kPosNeg and kNegPos Cfs, but there is not.
  //       Track1 always corresponds to the (Anti)Lambda, and Track1 to the K0Short
  //       If I want to ensure this further (checked locally already), use old version of this program
  //       and fix IsK0Short to use the full capabilities of K0Short finded in myAliFemtoV0TrackCut,
  //       not simply checking IsPionNSigma on positive and negative daughters

  double fSep;
  AliFemtoThreeVector first, second, tmp1, tmp2;

  for(int i=0; i<8; i++)
  {
    fSep = 0.;

    if(fType == kPosPos)  //POSITIVE (A)Lam daughter and POSITIVE K0s daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointPos(i);
    }
    else if(fType == kPosNeg)  //POSITIVE (A)Lam daughter and NEGATIVE K0s daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointNeg(i);
    }
    else if(fType == kNegPos)  //NEGATIVE (A)Lam daughter and POSITIVE K0s daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointPos(i);
    }
    else if(fType == kNegNeg)  //NEGATIVE (A)Lam daughter and NEGATIVE K0s daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointNeg(i);
    }

    else if(fType == kTrackPos)  //TRACK (i.e. KchP or KchM) with POSITIVE (A)Lam daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
      tmp2 = pair->Track2()->Track()->NominalTpcPoint(i);
    }
    else if(fType == kTrackNeg)  //TRACK (i.e. KchP or KchM) with NEGATIVE (A)Lam daughter
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      tmp2 = pair->Track2()->Track()->NominalTpcPoint(i);
    }

    else if(fType == kTrackTrack)  //TRACK-TRACK for completeness
    {
      tmp1 = pair->Track1()->Track()->NominalTpcPoint(i);
      tmp2 = pair->Track2()->Track()->NominalTpcPoint(i);
    }

    first.SetX((double)(tmp1.x()));
    first.SetY((double)(tmp1.y()));
    first.SetZ((double)(tmp1.z()));

    second.SetX((double)(tmp2.x()));
    second.SetY((double)(tmp2.y()));
    second.SetZ((double)(tmp2.z()));

    fSep = TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));

    aHisto->Fill(i,fSep,aWeight);

  }

}
