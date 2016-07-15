#!/usr/bin/env python3

from argparse import ArgumentParser
import asyncio

parser = ArgumentParser()

parser.add_argument('job_ids', nargs='+', help='A list of masterjobs to watch.')
parser.add_argument('-t','--time', help='Time (in minutes) to repeat the action', default='5', type=float)
args = parser.parse_args()

print(args)

@asyncio.coroutine
def kill_job(jid):
  prefix = "[kill_job %s] " % (jid)
  aliargs = ['aliensh', '-c', "masterjob %s -status DONE kill && exit" % (jid)]
  i = 0
  while True:
    print (prefix, i)
    cmd = asyncio.create_subprocess_exec(*aliargs, stdout = asyncio.subprocess.PIPE)
    proc = yield from cmd
    data = yield from proc.stdout.read()
    print (prefix, "===DATA====\n", data.decode())
    yield from asyncio.sleep(args.time * 60)
    i+=1

@asyncio.coroutine
def kill_main():
  killers = asyncio.JoinableQueue()
  for jid in args.job_ids:
    yield from killers.put(asyncio.async(kill_job(jid)))
  yield from killers.join()

loop = asyncio.get_event_loop()
try:
  loop.run_until_complete(kill_main())
  #loop.call_soon(kill_main())
  loop.run_forever()
finally:
  loop.close()
  
print ("Done!")
