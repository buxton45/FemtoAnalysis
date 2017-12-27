#include "Interpolate.h"

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  std::clock_t start = std::clock();

//-----------------------------------------------------------------------------

  TString tFileLocationNtupleBase = "~/Analysis/K0Lam/Results_cXicKch_20170423/Results_cXicKch_20170423";

  Interpolate *tInt = new Interpolate();

  tInt->BuildPairKStar3dVec(tFileLocationNtupleBase,kAXiKchP,k0010,kBp2,16,0.,0.16);
  tInt->MakeOtherArrays("~/Analysis/MathematicaNumericalIntegration/InterpHistsRepulsive");
  vector<vector<double> > myPairs2d = tInt->BuildPairs();
  vector<vector<double> > myGTildeReal = tInt->ReturnGTildeReal();


//--------------------------
  auto ta = std::chrono::high_resolution_clock::now();

  vector<double> mySerialResults = tInt->RunBilinearInterpolateSerial(myPairs2d,myGTildeReal);

  auto tb = std::chrono::high_resolution_clock::now();
  auto intab = std::chrono::duration_cast<std::chrono::microseconds>(tb-ta);
  cout << "Total time serial = " << intab.count() << " microseconds" << endl;

//--------------------------


//--------------------------
  auto t1 = std::chrono::high_resolution_clock::now();

  vector<double> myResults = tInt->RunBilinearInterpolateParallel(myPairs2d);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto int12 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
  cout << "Total time parallel = " << int12.count() << " microseconds" << endl;

//--------------------------

  for(int i=0; i<10000; i++) 
  {
/*
    cout << "i = " << i << endl;
    cout << "mySerialResults[i] = " << mySerialResults[i] << endl;
    cout << "myResults[i]       = " << myResults[i] << endl << endl;
*/
    if(abs(myResults[i]-mySerialResults[i]) > 0.0000001) 
    {
      cout << "DISCREPANCY in bin " << i << "!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << "myResults = " << myResults[i] << endl;
      cout << "mySerialResults = " << mySerialResults[i] << endl << endl;
    }
  }




//-------------------------------------------------------------------------------
  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Finished program in " << duration << " seconds" << endl;


//  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
