#!/bin/bash

dir=../../data/toy
julia julia.jl $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv

dir=../../data/titanic
julia julia.jl $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv

dir=../../data/big
julia julia.jl $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv

dir=../../data/reallyBig
julia julia.jl $dir/xTrain.csv $dir/xTest.csv $dir/yTrain.csv $dir/yTest.csv


