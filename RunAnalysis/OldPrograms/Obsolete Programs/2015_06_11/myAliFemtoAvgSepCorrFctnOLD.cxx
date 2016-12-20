///////////////////////////////////////////////////////////////////////////
//                                                                       //
// myAliFemtoAvgSepCorrFctn:                                             //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

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
  
  AliFemtoThreeVector first, second, tmp;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      //-------------  (++)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointPos(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepPosPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepPosPos /= 8;
      fNumeratorPosPos->Fill(fAvgSepPosPos);
      
      //-------------  (+-)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointNeg(i);
	  //cout<<"X neg: "<<tmp.x()<<endl;
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepPosNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepPosNeg /= 8;
      fNumeratorPosNeg->Fill(fAvgSepPosNeg);
      
      
      //-------------  (-+)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointPos(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepNegPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepNegPos /= 8;
      fNumeratorNegPos->Fill(fAvgSepNegPos);
      
      //-------------  (--)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointNeg(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepNegNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepNegNeg /= 8;
      fNumeratorNegNeg->Fill(fAvgSepNegNeg);
      
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      
      //-------------  (Track +)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackPos /= 8;
      fNumeratorTrackPos->Fill(fAvgSepTrackPos);
      
      
      //-------------  (Track -)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackNeg /= 8;
      fNumeratorTrackNeg->Fill(fAvgSepTrackNeg);
      
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->Track()->NominalTpcPoint(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackTrack += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackTrack /= 8;
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
  
  AliFemtoThreeVector first, second, tmp;
  
  if(pair->Track1()->V0() && pair->Track2()->V0()) //both particles are V0s
    {
      //-------------  (++)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointPos(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepPosPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepPosPos /= 8;
      fDenominatorPosPos->Fill(fAvgSepPosPos,weight);
      
      //-------------  (+-)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointNeg(i);
	  //cout<<"X neg: "<<tmp.x()<<endl;
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepPosNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepPosNeg /= 8;
      fDenominatorPosNeg->Fill(fAvgSepPosNeg,weight);
      
      
      //-------------  (-+)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointPos(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepNegPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepNegPos /= 8;
      fDenominatorNegPos->Fill(fAvgSepNegPos,weight);
      
      //-------------  (--)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->V0()->NominalTpcPointNeg(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepNegNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepNegNeg /= 8;
      fDenominatorNegNeg->Fill(fAvgSepNegNeg,weight);
      
    }
  
  if(pair->Track1()->V0() && pair->Track2()->Track()) //first particle is V0 and second is track
    {
      
      //-------------  (Track +)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointPos(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackPos += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackPos /= 8;
      fDenominatorTrackPos->Fill(fAvgSepTrackPos,weight);
      
      
      //-------------  (Track -)  -------------
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->V0()->NominalTpcPointNeg(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackNeg += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackNeg /= 8;
      fDenominatorTrackNeg->Fill(fAvgSepTrackNeg,weight);
      
    }
  
  
  if(pair->Track1()->Track() && pair->Track2()->Track()) //both particles are tracks
    {
      for(int i=0; i<8 ;i++)
	{
	  tmp = pair->Track1()->Track()->NominalTpcPoint(i);
	  //cout<<"X pos: "<<tmp.x()<<endl;
	  first.SetX((double)(tmp.x()));
	  first.SetY((double)tmp.y());
	  first.SetZ((double)tmp.z());
	  
	  tmp = pair->Track2()->Track()->NominalTpcPoint(i);
	  second.SetX((double)tmp.x());
	  second.SetY((double)tmp.y());
	  second.SetZ((double)tmp.z()); 
	  
	  fAvgSepTrackTrack += TMath::Sqrt(((double)first.x()-(double)second.x())*((double)first.x()-(double)second.x())+((double)first.y()-(double)second.y())*((double)first.y()-second.y())+((double)first.z()-(double)second.z())*((double)first.z()-(double)second.z()));
	}
      fAvgSepTrackTrack /= 8;
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
