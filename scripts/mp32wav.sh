#!/bin/bash

FILES=../train/whale/mp3/*
for fin in $FILES
do
	fout=${fin/mp3/wav}
	mpg123 $fin -w $fout
done
