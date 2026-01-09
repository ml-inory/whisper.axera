#!/bin/bash
if [ ! -d ax650n_bsp_sdk ]; then
  echo "clone ax650 bsp to ax650n_bsp_sdk, please wait..."
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
fi

if [ ! -d ax620e_bsp_sdk ]; then
echo "clone ax620e bsp to ax620e_bsp_sdk, please wait..."
  git clone https://github.com/AXERA-TECH/ax620e_bsp_sdk.git --depth=1
fi