# Harmonic Coordinates

## Compile

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

## Usage

    ./hc_bin ../data/woody-lo.off ../data/cage.off [../data/interior.off]

The last argument is optional. It's the interior control vertices data.

Press `1' to use Harmonic Coordinates

Press `2' to use Mean Value Coordinates (if use interior control then this is not available)
