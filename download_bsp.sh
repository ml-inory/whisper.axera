#!/bin/bash
if [ ! -d ax650n_bsp_sdk ]; then
  echo "\e[32mTips: \n   follow docs/how_to_speed_up_submodule_init.md if you this process too slow\e[0m\n"

  echo "clone ax650 bsp to ax650n_bsp_sdk, please wait..."
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git
fi