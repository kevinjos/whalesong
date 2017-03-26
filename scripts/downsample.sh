#!/bin/bash

INFILES=../train/whale/wav/*
OUTDIR=../train/whale/wavds
for f in $INFILES
do
	outfile=${f##./*/}
	sox $f $OUTDIR/$outfile -G rate -v 16384
done
