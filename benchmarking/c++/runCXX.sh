#!/bin/bash

#Change me!
gurlsData=~/gurls/data

dir=$gurlsData/toy
# dir=$gurlsData/titanic
# dir=$gurlsData/big
# dir=$gurlsData/reallyBig

./primal $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv
./dual $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv
./gaussian $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv

rm *.bin
