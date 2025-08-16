{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glib
      pkgs.xorg.libX11
    ];
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
  };
}