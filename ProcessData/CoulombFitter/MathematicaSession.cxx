///////////////////////////////////////////////////////////////////////////
// MathematicaSession:                                                   //
//  Do Mathematica magic in c++ programs!                                //
///////////////////////////////////////////////////////////////////////////

#include "MathematicaSession.h"

#ifdef __ROOT__
ClassImp(MathematicaSession)
#endif


//-------------------------------------------
//global stuff


  WSENV ep;
  WSLINK lp;

/*
static void deinit( void)
{
	if( ep) WSDeinitialize( ep);
}

static void closelink( void)
{
	if( lp) WSClose( lp);
}
*/
//-------------------------------------------

//________________________________________________________________________
MathematicaSession::MathematicaSession()
{
/*
  ep = (WSENV)0;
  lp = (WSLINK)0;

  int argc = 4;
  char *argv[5] = {"-linkname", "math -mathlink", "-linkmode", "launch", NULL};
  init_and_openlink( argc, argv);
*/
  InitializeSession();

}

//________________________________________________________________________
MathematicaSession::~MathematicaSession()
{
/*
  WSPutFunction( lp, "Exit", 0);
  WSEndPacket(lp);
  WSClose(lp);
*/
  EndSession();
}


//________________________________________________________________________
void MathematicaSession::InitializeSession()
{
  ep = (WSENV)0;
  lp = (WSLINK)0;

  int argc = 4;
  char *argv[5] = {"-linkname", "math -mathlink", "-linkmode", "launch", NULL};
  init_and_openlink( argc, argv);
}

//________________________________________________________________________
void MathematicaSession::EndSession()
{
  WSPutFunction( lp, "Exit", 0);
  WSEndPacket(lp);
  WSClose(lp);
}


//________________________________________________________________________
void MathematicaSession::deinit( void)
{
  if( ep) WSDeinitialize( ep);
}



//________________________________________________________________________
void MathematicaSession::closelink( void)
{
  if( lp) WSClose( lp);
}




//________________________________________________________________________
void MathematicaSession::init_and_openlink( int argc, char* argv[])
{
  #if WSINTERFACE >= 3
    int err;
  #else
    long err;
  #endif /* WSINTERFACE >= 3 */

  ep =  WSInitialize( (WSParametersPointer)0);
  if( ep == (WSENV)0) exit(1);
  atexit( deinit);

  #if WSINTERFACE < 3
    lp = WSOpenArgv( ep, argv, argv + argc, &err);
  #else
    lp = WSOpenArgcArgv( ep, argc, argv, &err);
  #endif

  if(lp == (WSLINK)0) exit(2);
  atexit( closelink);	
}



//________________________________________________________________________
void MathematicaSession::discardResult(WSLINK lp) 
{
  int pkt;
  while( (pkt = WSNextPacket( lp), pkt) && pkt != RETURNPKT) 
  {
    WSNewPacket( lp);
    if (WSError( lp)) error( lp);
  }
  WSNewPacket(lp);
}



//________________________________________________________________________
void MathematicaSession::error( WSLINK lp)
{
  if( WSError( lp)){
    fprintf( stderr, "Error detected by WSTP: %s.\n",WSErrorMessage(lp));
  }
  else
  {
    fprintf( stderr, "Error detected by this program.\n");
  }
  exit(3);
}


//________________________________________________________________________
complex<double> MathematicaSession::GetGamma(complex<double> aCmplx)
{
/*
  WSENV ep = (WSENV)0;
  WSLINK lp = (WSLINK)0;
*/

  #if WSINTERFACE >= 4
    int len;
  #else
    long len;
  #endif

  int pkt; //return value from WSNextPacket
           // = integer code corresponding to the packet type

  double tRe = real(aCmplx);
  double tIm = imag(aCmplx);

  double tGammaReal, tGammaImag;

  const char *tStringPtr;

  WSPutFunction(lp, "EvaluatePacket", 1);
    WSPutFunction(lp, "ToExpression", 1);
      WSPutString(lp, "MyGamma[aRe_,aIm_] := Gamma[Complex[aRe,aIm]]");
  WSEndPacket(lp);
  discardResult(lp);


  WSPutFunction( lp, "EvaluatePacket", 1);
    WSPutFunction( lp, "MyGamma", 2);
      WSPutDouble(lp,tRe);
      WSPutDouble(lp,tIm);
  WSEndPacket( lp);


  /* skip any packets before the first ReturnPacket */
  while( (pkt = WSNextPacket( lp), pkt) && pkt != RETURNPKT) 
  {
    WSNewPacket( lp);
    if (WSError( lp)) error( lp);
  }


  if ( WSGetFunction( lp, &tStringPtr, &len) && len==2)
  {
    WSGetDouble( lp, &tGammaReal);
    WSGetDouble( lp, &tGammaImag);

    if (WSError( lp)) error( lp);

    WSReleaseSymbol(lp,tStringPtr);  //WSGetFunction allocated memory, this disowns the memory
  }

//  WSPutFunction( lp, "Exit", 0);

  complex<double> tReturnCmplx (tGammaReal,tGammaImag);
  return tReturnCmplx;
}


//________________________________________________________________________
complex<double> MathematicaSession::GetDiGamma(complex<double> aCmplx)
{
  #if WSINTERFACE >= 4
    int len;
  #else
    long len;
  #endif


  int pkt; //return value from WSNextPacket
           // = integer code corresponding to the packet type

  double tRe = real(aCmplx);
  double tIm = imag(aCmplx);

  double tDiGammaReal, tDiGammaImag;

  const char *tStringPtr;

  WSPutFunction(lp, "EvaluatePacket", 1);
    WSPutFunction(lp, "ToExpression", 1);
      WSPutString(lp, "DiGamma[aRe_,aIm_] := PolyGamma[Complex[aRe,aIm]]");
  WSEndPacket(lp);
  discardResult(lp);


  WSPutFunction( lp, "EvaluatePacket", 1);
    WSPutFunction( lp, "DiGamma", 2);
      WSPutDouble(lp,tRe);
      WSPutDouble(lp,tIm);
  WSEndPacket( lp);


  /* skip any packets before the first ReturnPacket */
  while( (pkt = WSNextPacket( lp), pkt) && pkt != RETURNPKT) 
  {
    WSNewPacket( lp);
    if (WSError( lp)) error( lp);
  }


  if ( WSGetFunction( lp, &tStringPtr, &len) && len==2)
  {
    WSGetDouble( lp, &tDiGammaReal);
    WSGetDouble( lp, &tDiGammaImag);

    if (WSError( lp)) error( lp);

    WSReleaseSymbol(lp,tStringPtr);  //WSGetFunction allocated memory, this disowns the memory
  }

//  WSPutFunction( lp, "Exit", 0);

  complex<double> tReturnCmplx (tDiGammaReal,tDiGammaImag);
  return tReturnCmplx;
}


//________________________________________________________________________
complex<double> MathematicaSession::GetHyperGeo1F1(complex<double> aA, complex<double> aB, complex<double> aZ)
{
  #if WSINTERFACE >= 4
    int len;
  #else
    long len;
  #endif


  int pkt; //return value from WSNextPacket
           // = integer code corresponding to the packet type

  double tReA = real(aA);
  double tImA = imag(aA);

  double tReB = real(aB);
  double tImB = imag(aB);

  double tReZ = real(aZ);
  double tImZ = imag(aZ);

  double tHyperGeoReal, tHyperGeoImag;

  const char *tStringPtr;

  WSPutFunction(lp, "EvaluatePacket", 1);
    WSPutFunction(lp, "ToExpression", 1);
      WSPutString(lp, "HyperGeo1F1[aReA_,aImA_,aReB_,aImB_,aReZ_,aImZ_] := Hypergeometric1F1[Complex[aReA,aImA],Complex[aReB,aImB],Complex[aReZ,aImZ]]");
  WSEndPacket(lp);
  discardResult(lp);


  WSPutFunction( lp, "EvaluatePacket", 1);
    WSPutFunction( lp, "HyperGeo1F1", 6);
      WSPutDouble(lp,tReA);
      WSPutDouble(lp,tImA);
      WSPutDouble(lp,tReB);
      WSPutDouble(lp,tImB);
      WSPutDouble(lp,tReZ);
      WSPutDouble(lp,tImZ);
  WSEndPacket( lp);


  /* skip any packets before the first ReturnPacket */
  while( (pkt = WSNextPacket( lp), pkt) && pkt != RETURNPKT) 
  {
    WSNewPacket( lp);
    if (WSError( lp)) error( lp);
  }


  if ( WSGetFunction( lp, &tStringPtr, &len) && len==2)
  {
    WSGetDouble( lp, &tHyperGeoReal);
    WSGetDouble( lp, &tHyperGeoImag);

    if (WSError( lp)) error( lp);

    WSReleaseSymbol(lp,tStringPtr);  //WSGetFunction allocated memory, this disowns the memory
  }

//  WSPutFunction( lp, "Exit", 0);

  complex<double> tReturnCmplx (tHyperGeoReal,tHyperGeoImag);
  return tReturnCmplx;
}


//________________________________________________________________________
complex<double> MathematicaSession::GetCoulombHpmFunction(double aPhaseShift, double aEta, double aRho, HFuncType aPlusOrMinus, double aL)
{
  #if WSINTERFACE >= 4
    int len;
  #else
    long len;
  #endif


  int pkt; //return value from WSNextPacket
           // = integer code corresponding to the packet type

  double tWhittWReal, tWhittWImag;

  const char *tStringPtr;
  
  WSPutFunction(lp, "EvaluatePacket", 1);
    WSPutFunction(lp, "ToExpression", 1);
  if(aPlusOrMinus == kHPlus) WSPutString(lp, "WhittW[tEta_,tRho_,tL_] := WhittakerW[Complex[0.,-tEta],tL+0.5 ,Complex[0.,-2.*tRho]]");
  else WSPutString(lp, "WhittW[tEta_,tRho_,tL_] := WhittakerW[Complex[0.,tEta],tL+0.5 ,Complex[0.,2.*tRho]]");
  WSEndPacket(lp);
  discardResult(lp);


  WSPutFunction(lp, "EvaluatePacket", 1);
    WSPutFunction(lp, "WhittW", 3);
      WSPutDouble(lp,aEta);
      WSPutDouble(lp,aRho);
      WSPutDouble(lp,aL);  //WSPutInteger instead?
  WSEndPacket(lp);

  /* skip any packets before the first ReturnPacket */
  while( (pkt = WSNextPacket( lp), pkt) && pkt != RETURNPKT) 
  {
    WSNewPacket( lp);
    if (WSError( lp)) error( lp);
  }


  if ( WSGetFunction( lp, &tStringPtr, &len) && len==2)
  {
    WSGetDouble( lp, &tWhittWReal);
    WSGetDouble( lp, &tWhittWImag);

    if(WSError(lp))  //TODO
    {
      if(std::abs(tWhittWReal) < std::numeric_limits< double >::min())
      {
        tWhittWReal = 0.;
        WSClearError(lp);
        WSNewPacket(lp);
      }
      else if(std::abs(tWhittWImag) < std::numeric_limits< double >::min())
      {
        tWhittWImag = 0.;
        WSClearError(lp);
        WSNewPacket(lp);
      }
      else error(lp);
    }

    WSReleaseSymbol(lp,tStringPtr);  //WSGetFunction allocated memory, this disowns the memory
  }


  complex<double> tWhittW (tWhittWReal,tWhittWImag);

  complex<double> tExpArgument;
  double tExpArgReal, tExpArgImag;
  tExpArgReal = M_PI_2*aEta; //M_PI_2 = pi/2, defined in math.h
  tExpArgImag = aPhaseShift - M_PI_2*aL;

  if(aPlusOrMinus == kHPlus)
  {
    tExpArgument = complex<double>(tExpArgReal,tExpArgImag);
  }
  else
  {
    tExpArgImag *= -1;
    tExpArgument = complex<double>(tExpArgReal,tExpArgImag);
  }

  complex<double> tReturnHFunc = std::exp(tExpArgument)*tWhittW;

  return tReturnHFunc;
}

//________________________________________________________________________
complex<double> MathematicaSession::GetCoulombRegularFWaveFunction(double aPhaseShift, double aEta, double aRho, double aL)
{
  complex<double> tHFuncPlus = GetCoulombHpmFunction(aPhaseShift,aEta,aRho,kHPlus,aL);
  complex<double> tHFuncMinus = GetCoulombHpmFunction(aPhaseShift,aEta,aRho,kHMinus,aL);

  complex<double> tImI (0.,1.);
  complex<double> tReturnF = (1./(2.*tImI))*(tHFuncPlus - tHFuncMinus);
  return tReturnF;
}


//________________________________________________________________________
complex<double> MathematicaSession::GetCoulombSingularGWaveFunction(double aPhaseShift, double aEta, double aRho, double aL)
{
  complex<double> tHFuncPlus = GetCoulombHpmFunction(aPhaseShift,aEta,aRho,kHPlus,aL);
  complex<double> tHFuncMinus = GetCoulombHpmFunction(aPhaseShift,aEta,aRho,kHMinus,aL);

  complex<double> tReturnG = (1./2.)*(tHFuncPlus + tHFuncMinus);
  return tReturnG;
}

