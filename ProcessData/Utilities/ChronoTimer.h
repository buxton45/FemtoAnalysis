#ifndef CHRONOTIMER_H
#define CHRONOTIMER_H

#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

enum ClockType {kMin=0, kSec=1, kMilli=2, kMicro=3, kNano=4};

struct ChronoTimer
{
  std::chrono::high_resolution_clock::time_point fStart;
  std::chrono::high_resolution_clock::time_point fEnd;
  std::chrono::duration<double, std::nano> fInterval;
  ClockType fClockType;

  //-------------------------------
  ChronoTimer(ClockType aClockType=kMilli)
  {
    fClockType = aClockType;
  }

  //-------------------------------
  ~ChronoTimer()
  {

  }

  //-------------------------------
  void Start()
  {
    fStart = std::chrono::high_resolution_clock::now();
  }

  //-------------------------------
  void Stop()
  {
    fEnd = std::chrono::high_resolution_clock::now();
  }

  //-------------------------------
  double GetInterval()
  {
    fInterval = fEnd-fStart;
    double tReturnInterval = fInterval.count();

    if(fClockType==kMin)
    {
      tReturnInterval *= pow(10,-9);
      tReturnInterval /= 60.;
    }
    else if(fClockType==kSec) tReturnInterval *= pow(10,-9);
    else if(fClockType==kMilli) tReturnInterval *= pow(10,-6);
    else if(fClockType==kMicro) tReturnInterval *= pow(10,-3);

    return tReturnInterval;
  }

  //-------------------------------
  void PrintInterval()
  {
    cout << "Interval = " << GetInterval();

    if(fClockType==kMin) cout << " minutes" << endl;
    else if(fClockType==kSec) cout << " seconds" << endl;
    else if(fClockType==kMilli) cout << " milliseconds" << endl;
    else if(fClockType==kMicro) cout << " microseconds" << endl;
    else if(fClockType==kNano) cout << " nanoseconds" << endl;
    else cout << " ERROR!" << endl;
  }


};

#endif
