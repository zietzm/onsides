There are similarly three main scripts here, and, as with Japan, the second one calls a bunch of other scripts.

Download seems to be working now. We'll have to run it for real, with an actual internet connection and see if it misses too much...

Again, script 2 calls a bunch of other scripts.
For some reason, instead of just importing and calling a python function, he's calling the python interpreter from python using os.system.
This is a non-workable hack if you're using modern tooling.
We will need to go through all of those scripts separately and refactor them to be directly callable.

Ok, first one should be good, but I'm blocked by not having Java installed lol
