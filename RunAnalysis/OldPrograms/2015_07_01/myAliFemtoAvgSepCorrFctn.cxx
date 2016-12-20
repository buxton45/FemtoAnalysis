///////////////////////////////////////////////////////////////////////////
//                                                                       //
// myAliFemtoAvgSepCorrFctn:                                             //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

//NOTE:  this program does not work correctly.  First, there is no need for an ambiguity resolution,
//  as discussed in the code.  I always set up (A)Lam to be my first particle (Track1) and my K0Short to be
//  my second particle (Track2).  Therefore, IsK0Short(pair->Track1()->V0()) should always return false.
//  However, as the functin IsK0Short is now (only checking IsPionNSignma), some (Anti)Lambdas pass this cut,
//  so IsK0Short(pair->Track1()->V0()) sometimes returns true, which is bad.
//  The result is a mixing of AvgSep cfs.  For example, p(Lam)-pi-(K0) cf will be diluted with pi-(Lam)-pi+(K0) cf

//  This issue is resolved when using the full power of the K0Short finder from myAliFemtoV0TrackCut
//  i.e., IsK0Short should test more than just IsPionNSigma on the positive and negative daughters, it should also 
//  check IsMisIDLambda and IsMisIDAntiLambda, and also should ensure the mass of the particle in question falls within
//  the accepted range of Minv.
//  Nonetheless, this fix is not necessary.

//  There is no need to ever check IsK0Short.  There is no ambiguity, and taking out this check will speed up the code.
//  The reason this ambiguity resolution was put in place was because I thought Track1 could sometimes be (A)Lam and sometimes
//  be a K0Short, which is NOT true! (unless, of course, I set K0Short as particlecut1 and Lam as particlecut2, which I should not do

#include "myAliFemtoAvgSepCorrFctn.h"
//#include "AliFemtoHisto.h"
#include <cstdio>

#ifdef __ROOT__
ClassImp(myAliFemtoAvgSepCorrFctn)
#endif

//____________________________
myAliFemtoAvgSepCorrFctn::myAliFemtoAvgSepCorrFctn(const char* title, const int& nbins, const float& AvgSepLo, const float& AvgSepHi):
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
  fNumeratorPosPos = new TH1D(tTitNumPosPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumPosNeg[101] = "NumPosNeg";
  strncat(tTitNumPosNeg,title, 100);
  fNumeratorPosNeg = new TH1D(tTitNumPosNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumNegPos[101] = "NumNegPos";
  strncat(tTitNumNegPos,title, 100);
  fNumeratorNegPos = new TH1D(tTitNumNegPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumNegNeg[101] = "NumNegNeg";
  strncat(tTitNumNegNeg,title, 100);
  fNumeratorNegNeg = new TH1D(tTitNumNegNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumTrackPos[101] = "NumTrackPos";
  strncat(tTitNumTrackPos,title, 100);
  fNumeratorTrackPos = new TH1D(tTitNumTrackPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumTrackNeg[101] = "NumTrackNeg";
  strncat(tTitNumTrackNeg,title, 100);
  fNumeratorTrackNeg = new TH1D(tTitNumTrackNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitNumTrackTrack[101] = "NumTrackTrack";
  strncat(tTitNumTrackTrack,title, 100);
  fNumeratorTrackTrack = new TH1D(tTitNumTrackTrack,title,nbins,AvgSepLo,AvgSepHi);

  // set up denominators
  char tTitDenPosPos[101] = "DenPosPos";
  strncat(tTitDenPosPos,title, 100);
  fDenominatorPosPos = new TH1D(tTitDenPosPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenPosNeg[101] = "DenPosNeg";
  strncat(tTitDenPosNeg,title, 100);
  fDenominatorPosNeg = new TH1D(tTitDenPosNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenNegPos[101] = "DenNegPos";
  strncat(tTitDenNegPos,title, 100);
  fDenominatorNegPos = new TH1D(tTitDenNegPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenNegNeg[101] = "DenNegNeg";
  strncat(tTitDenNegNeg,title, 100);
  fDenominatorNegNeg = new TH1D(tTitDenNegNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenTrackPos[101] = "DenTrackPos";
  strncat(tTitDenTrackPos,title, 100);
  fDenominatorTrackPos = new TH1D(tTitDenTrackPos,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenTrackNeg[101] = "DenTrackNeg";
  strncat(tTitDenTrackNeg,title, 100);
  fDenominatorTrackNeg = new TH1D(tTitDenTrackNeg,title,nbins,AvgSepLo,AvgSepHi);
  char tTitDenTrackTrack[101] = "DenTrackTrack";
  strncat(tTitDenTrackTrack,title, 100);
  fDenominatorTrackTrack = new TH1D(tTitDenTrackTrack,title,nbins,AvgSepLo,AvgSepHi);


  // this next bit is unfortunately needed so that we can have many histos of same "title"
  // it is neccessary if we typedef TH1D to TH1d (which we do)
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
myAliFemtoAvgSepCorrFctn::myAliFemtoAvgSepCorrFctn(const myAliFemtoAvgSepCorrFctn& aCorrFctn) :
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
  fNumeratorPosPos = new TH1D(*aCorrFctn.fNumeratorPosPos);
  fNumeratorPosNeg = new TH1D(*aCorrFctn.fNumeratorPosNeg);
  fNumeratorNegPos = new TH1D(*aCorrFctn.fNumeratorNegPos);
  fNumeratorNegNeg = new TH1D(*aCorrFctn.fNumeratorNegNeg);
  fNumeratorTrackPos = new TH1D(*aCorrFctn.fNumeratorTrackPos);
  fNumeratorTrackNeg = new TH1D(*aCorrFctn.fNumeratorTrackNeg);
  fNumeratorTrackTrack = new TH1D(*aCorrFctn.fNumeratorTrackTrack);

  fDenominatorPosPos = new TH1D(*aCorrFctn.fDenominatorPosPos);
  fDenominatorPosNeg = new TH1D(*aCorrFctn.fDenominatorPosNeg);
  fDenominatorNegPos = new TH1D(*aCorrFctn.fDenominatorNegPos);
  fDenominatorNegNeg = new TH1D(*aCorrFctn.fDenominatorNegNeg);
  fDenominatorTrackPos = new TH1D(*aCorrFctn.fDenominatorTrackPos);
  fDenominatorTrackNeg = new TH1D(*aCorrFctn.fDenominatorTrackNeg);
  fDenominatorTrackTrack = new TH1D(*aCorrFctn.fDenominatorTrackTrack);
}
//____________________________
myAliFemtoAvgSepCorrFctn::~myAliFemtoAvgSepCorrFctn(){
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
myAliFemtoAvgSepCorrFctn& myAliFemtoAvgSepCorrFctn::operator=(const myAliFemtoAvgSepCorrFctn& aCorrFctn)
{
  // assignment operator
  if (this == &aCorrFctn)
    return *this;

  if (fNumeratorPosPos) delete fNumeratorPosPos;
  fNumeratorPosPos = new TH1D(*aCorrFctn.fNumeratorPosPos);
  if (fNumeratorPosNeg) delete fNumeratorPosNeg;
  fNumeratorPosNeg = new TH1D(*aCorrFctn.fNumeratorPosNeg);
  if (fNumeratorNegPos) delete fNumeratorNegPos;
  fNumeratorNegPos = new TH1D(*aCorrFctn.fNumeratorNegPos);
  if (fNumeratorNegNeg) delete fNumeratorNegNeg;
  fNumeratorNegNeg = new TH1D(*aCorrFctn.fNumeratorNegNeg);
  if (fNumeratorTrackPos) delete fNumeratorTrackPos;
  fNumeratorTrackPos = new TH1D(*aCorrFctn.fNumeratorTrackPos);
  if (fNumeratorTrackNeg) delete fNumeratorTrackNeg;
  fNumeratorTrackNeg = new TH1D(*aCorrFctn.fNumeratorTrackNeg);
  if (fNumeratorTrackTrack) delete fNumeratorTrackTrack;
  fNumeratorTrackTrack = new TH1D(*aCorrFctn.fNumeratorTrackTrack);

  if (fDenominatorPosPos) delete fDenominatorPosPos;
  fDenominatorPosPos = new TH1D(*aCorrFctn.fDenominatorPosPos);
  if (fDenominatorPosNeg) delete fDenominatorPosNeg;
  fDenominatorPosNeg = new TH1D(*aCorrFctn.fDenominatorPosNeg);
  if (fDenominatorNegPos) delete fDenominatorNegPos;
  fDenominatorNegPos = new TH1D(*aCorrFctn.fDenominatorNegPos);
  if (fDenominatorNegNeg) delete fDenominatorNegNeg;
  fDenominatorNegNeg = new TH1D(*aCorrFctn.fDenominatorNegNeg);
  if (fDenominatorTrackPos) delete fDenominatorTrackPos;
  fDenominatorTrackPos = new TH1D(*aCorrFctn.fDenominatorTrackPos);
  if (fDenominatorTrackNeg) delete fDenominatorTrackNeg;
  fDenominatorTrackNeg = new TH1D(*aCorrFctn.fDenominatorTrackNeg);
  if (fDenominatorTrackTrack) delete fDenominatorTrackTrack;
  fDenominatorTrackTrack = new TH1D(*aCorrFctn.fDenominatorTrackTrack);

  return *this;
}

//_________________________
void myAliFemtoAvgSepCorrFctn::Finish(){
  // here is where we should normalize, fit, etc...
  // we should NOT Draw() the histos (as I had done it below),
  // since we want to insulate ourselves from root at this level
  // of the code.  Do it instead at root command line with browser.
  //  fNumerator->Draw();
  //fDenominator->Draw();
}

//____________________________
AliFemtoString myAliFemtoAvgSepCorrFctn::Report(){
  // construct report
  string stemp = "AvgSep Correlation Function Report:\n";
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
void myAliFemtoAvgSepCorrFctn::AddRealPair(AliFemtoPair* pair){
  // add true pair
  if(fPairCut)
    if (!fPairCut->Pass(pair)) return;
  
  //--used when both are V0s
  double fAvgSepPosPos=0.;
  double fAvgSepPosNeg=0.;
  double fAvgSepNegPos=0.;
  double fAvgSepNegNeg=0.;
  
  //--used when one V0 and one track
  double fAvgSepTrackPos=0.;
  double fAvgSepTrackNeg=0.;
  
  //--used when both are tracks
  double fAvgSepTrackTrack=0.;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      fAvgSepPosPos = CalculateAvgSep(pair,kPosPos);
        fNumeratorPosPos->Fill(fAvgSepPosPos);
      fAvgSepPosNeg = CalculateAvgSep(pair,kPosNeg);
        fNumeratorPosNeg->Fill(fAvgSepPosNeg);
      fAvgSepNegPos = CalculateAvgSep(pair,kNegPos);
        fNumeratorNegPos->Fill(fAvgSepNegPos);
      fAvgSepNegNeg = CalculateAvgSep(pair,kNegNeg);
        fNumeratorNegNeg->Fill(fAvgSepNegNeg);
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      fAvgSepTrackPos = CalculateAvgSep(pair,kTrackPos);
        fNumeratorTrackPos->Fill(fAvgSepTrackPos);
      fAvgSepTrackNeg = CalculateAvgSep(pair,kTrackNeg);
        fNumeratorTrackNeg->Fill(fAvgSepTrackNeg);
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      fAvgSepTrackTrack = CalculateAvgSep(pair,kTrackTrack);
        fNumeratorTrackTrack->Fill(fAvgSepTrackTrack);
    } 
}

//____________________________
void myAliFemtoAvgSepCorrFctn::AddMixedPair(AliFemtoPair* pair){
  // add mixed (background) pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;
  
  double weight = 1.0;
  
  //--used when both are V0s
  double fAvgSepPosPos=0.;
  double fAvgSepPosNeg=0.;
  double fAvgSepNegPos=0.;
  double fAvgSepNegNeg=0.;
  
  //--used when one V0 and one track
  double fAvgSepTrackPos=0.;
  double fAvgSepTrackNeg=0.;
  
  //--used when both are tracks
  double fAvgSepTrackTrack=0.;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      fAvgSepPosPos = CalculateAvgSep(pair,kPosPos);
        fDenominatorPosPos->Fill(fAvgSepPosPos,weight);
      fAvgSepPosNeg = CalculateAvgSep(pair,kPosNeg);
        fDenominatorPosNeg->Fill(fAvgSepPosNeg,weight);
      fAvgSepNegPos = CalculateAvgSep(pair,kNegPos);
        fDenominatorNegPos->Fill(fAvgSepNegPos,weight);
      fAvgSepNegNeg = CalculateAvgSep(pair,kNegNeg);
        fDenominatorNegNeg->Fill(fAvgSepNegNeg,weight);
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      fAvgSepTrackPos = CalculateAvgSep(pair,kTrackPos);
        fDenominatorTrackPos->Fill(fAvgSepTrackPos,weight);
      fAvgSepTrackNeg = CalculateAvgSep(pair,kTrackNeg);
        fDenominatorTrackNeg->Fill(fAvgSepTrackNeg,weight);
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      fAvgSepTrackTrack = CalculateAvgSep(pair,kTrackTrack);
        fDenominatorTrackTrack->Fill(fAvgSepTrackTrack,weight);
    }
}
//____________________________
void myAliFemtoAvgSepCorrFctn::Write(){
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
TList* myAliFemtoAvgSepCorrFctn::GetOutputList()
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
double myAliFemtoAvgSepCorrFctn::CalculateAvgSep(AliFemtoPair* pair, PairType fType)
{
  double fAvgSep = 0.;
  AliFemtoThreeVector first, second, tmp1, tmp2;

  for(int i=0; i<8; i++)
  {
    if(fType == kPosPos)  //no ambiguity
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointPos(i);
    }
    else if(fType == kPosNeg)  //ambiguity!  Always want POSITIVE (anti)Lambda daughter and NEGATIVE K0s daughter
    {
      if( IsK0Short(pair->Track1()->V0()) )
      {
	tmp1 = pair->Track2()->V0()->NominalTpcPointPos(i);
	tmp2 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      }
      else if( IsK0Short(pair->Track2()->V0()) )
      {
	tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
	tmp2 = pair->Track2()->V0()->NominalTpcPointNeg(i);
      }
      else
      {
	cout << "CAUTION (1)!!!!!!" << endl;
	cout << "NEITHER PARTICLE FOUND TO BE K0SHORT!!!!!!!!!!!" << endl;
      }
    }
    else if(fType == kNegPos)  //ambiguity!  Always want NEGATIVE (anti)Lambda daughter and POSITIVE K0s daughter
    {
      if( IsK0Short(pair->Track1()->V0()) )
      {
	tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
	tmp2 = pair->Track2()->V0()->NominalTpcPointNeg(i);
      }
      else if( IsK0Short(pair->Track2()->V0()) )
      {
	tmp1 = pair->Track2()->V0()->NominalTpcPointPos(i);
	tmp2 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      }
      else
      {
	cout << "CAUTION (2)!!!!!!" << endl;
	cout << "NEITHER PARTICLE FOUND TO BE K0SHORT!!!!!!!!!!!" << endl;
      }
    }
    else if(fType == kNegNeg)  //no ambiguity
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      tmp2 = pair->Track2()->V0()->NominalTpcPointNeg(i);
    }
    else if(fType == kTrackPos)  //no ambiguity
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointPos(i);
      tmp2 = pair->Track2()->Track()->NominalTpcPoint(i);
    }
    else if(fType == kTrackNeg)  //no ambiguity
    {
      tmp1 = pair->Track1()->V0()->NominalTpcPointNeg(i);
      tmp2 = pair->Track2()->Track()->NominalTpcPoint(i);
    }
    else if(fType == kTrackTrack)  //no ambiguity
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

    fAvgSep += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));

  }

  fAvgSep /= 8;
  return fAvgSep;

}

//______________________________
bool myAliFemtoAvgSepCorrFctn::IsPionNSigma(float mom, float nsigmaTPCPi, float nsigmaTOFPi)
{
  if(mom<0.8)
    {
      if(TMath::Abs(nsigmaTPCPi)<3.0) return true;
    }
  else
    {
      if(nsigmaTOFPi<-999.)
	{
	  if(TMath::Abs(nsigmaTPCPi)<3.0) return true;
	}
      else
	{
	  if(TMath::Abs(nsigmaTPCPi)<3.0 && TMath::Abs(nsigmaTOFPi)<3.0) return true;
	}
    }
  return false; 
}

//______________________________
bool myAliFemtoAvgSepCorrFctn::IsProtonNSigma(float mom, float nsigmaTPCP, float nsigmaTOFP)
{
  if(mom<0.8)
    {
      if(TMath::Abs(nsigmaTPCP)<3.0) return true;
    }
  else
    {
      if(nsigmaTOFP<-999.)
	{
	  if(TMath::Abs(nsigmaTPCP)<3.0) return true;
	}
      else
	{
	  if(TMath::Abs(nsigmaTPCP)<3.0 && TMath::Abs(nsigmaTOFP)<3.0) return true;
	}
    }
  return false;
}

//______________________________
bool myAliFemtoAvgSepCorrFctn::IsK0Short(AliFemtoV0* aV0)
{
  //Looking for K0s = pip + pim
  if(IsPionNSigma(aV0->PtPos(), aV0->PosNSigmaTPCPi(), aV0->PosNSigmaTOFPi())) //is positive daughter a pip?
  {
    if(IsPionNSigma(aV0->PtNeg(), aV0->NegNSigmaTPCPi(), aV0->NegNSigmaTOFPi()))  //and is negative daughter a pim?
    {
      return true;
    }
  }
  return false;

}
