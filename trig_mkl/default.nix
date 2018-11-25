with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "trig-mkl-env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = with python3.pkgs; [
    pip cython
    numpy
    (callPackage ../misc/nix/mkl.nix {})
  ];
}
