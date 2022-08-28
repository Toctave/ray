compiler=gcc
flags="-O3 -g -Wall -Wextra -Wno-missing-braces"

mkdir -p build
cd build
# $compiler $flags -c ../src/image.c -o img.o
$compiler $flags ../src/main.c img.o -o ../ray -lm -lpthread
