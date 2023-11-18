set -xe

rm -rf cutlass.origin

git clone https://github.com/NVIDIA/cutlass.git --depth=1 cutlass.origin

rm -rf cutlass
mkdir -p cutlass
cp -r cutlass.origin/include cutlass

git -C cutlass.origin/ rev-parse HEAD > cutlass/readme

rm -rf cutlass.origin
