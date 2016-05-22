A tool suite to rip open source hydrophonic streams and identify whale vocalizations.

Requirements
============
* Unix OS
* Python 3.5
* scipy, numpy, scikit-learn
* streamripper
* mpg123
* postgres

Setup for training
==================
	$ mkdir -p train/whale/mp3
	$ bash getorca.sh
	$ bash mp32wav.sh
	$ createuser whalesong
	$ createdb whalesong
	$ python migratedb.py
