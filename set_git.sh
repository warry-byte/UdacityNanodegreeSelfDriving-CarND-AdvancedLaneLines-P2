#!/bin/bash

# Make sure you have the latest version of the repo

echo setting up git

git config --global user.name "warry-byte"
git config --global user.email "antoun.aj@gmail.com"
git remote set-url origin git@github.com:warry-byte/UdacityNanodegreeSelfDriving-CarND-AdvancedLaneLines-P2.git
echo

echo Please verify remote:
git remote -v
echo