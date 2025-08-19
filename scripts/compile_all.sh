#!/usr/bin/env bash

this_dir=$(dirname "$0")

echo ""
echo "********build fps************"
cd $this_dir/../core/csrc/fps/
rm -rf build
python setup.py


echo ""
echo "********build flow************"
cd ../flow/
rm -rf build/
python setup.py build_ext --inplace


echo ""
echo "********build ransac_voting************"
cd ../ransac_voting
rm -rf build
python setup.py build_ext --inplace


echo ""
echo "********build uncertainty pnp************no"
cd ../uncertainty_pnp
# sh build_ceres.sh
rm -rf build/
# sudo apt-get update
# sudo apt-get install -y libceres-dev libgoogle-glog-dev libgflags-dev
# mkdir -p lib
# ln -sf /usr/lib/libceres.so      ./lib/libceres.so
# ln -sf /usr/lib/x86_64-linux-gnu/libglog.so       ./lib/libglog.so
# ln -sf /usr/lib/x86_64-linux-gnu/libgflags.so     ./lib/libgflags.so  # 若需要
python setup.py build_ext --inplace


echo ""
echo "********build torch_nndistance (chamfer distance)************"
cd ../torch_nndistance
rm -rf build
python setup.py build_ext --inplace


echo ""
echo "********build cpp egl renderer************"
cd ../../../lib/egl_renderer/
rm -rf build/
python setup.py build_ext --inplace
