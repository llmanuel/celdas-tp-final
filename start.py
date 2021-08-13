#!/usr/bin/env python

import sys, getopt
from DQNAgent import Agent

def main(argv):
  mode = "t"
  try:
    opts, args = getopt.getopt(argv, "m:", "mode=")
  except getopt.GetoptError:
    print("options are train(t) or play(p). start.py -m <t or p> or --mode <t or p>")
  for opt, arg in opts:
    if opt in ("-m", "--mode"):
      mode = arg

  Agent().run(mode)

if __name__ == "__main__":
  main(sys.argv[1:])
