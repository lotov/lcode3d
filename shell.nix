{ pkgs ? import <nixpkgs> {} }:

let
  cupy = pkgs.python3Packages.buildPythonPackage rec {
    pname = "cupy";
    version = "6.0.0rc1";
    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "0pbw872f4m4jck4z114xdgs084ah5vin4chki9j6b17ybvx9jnxw";
    };
    propagatedBuildInputs = with pkgs; [
      cudatoolkit cudnn linuxPackages.nvidia_x11 nccl
    ] ++ (with pkgs.python3Packages; [
      fastrlock numpy six wheel
    ]);
    doCheck = false;
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      cupy
      matplotlib
      numba
      numpy
      scipy
      sphinx sphinx_rtd_theme
    ]))
    cloc
    python3Packages.flake8
    nvtop
  ];

  NUMBA_FORCE_CUDA_CC = "6.1";
  NUMBA_WARNINGS = 1;
  NUMBAPRO_NVVM = "${pkgs.cudatoolkit}/nvvm/lib64/libnvvm.so";
  NUMBAPRO_LIBDEVICE = "${pkgs.cudatoolkit}/nvvm/libdevice";
  NUMBAPRO_CUDA_DRIVER = "${pkgs.linuxPackages.nvidia_x11}/lib/libcuda.so";

  #OMP_NUM_THREADS = 1;
  #NUMBA_NUM_THREADS = 1;
  #MKL_NUM_THREADS = 1;
}
