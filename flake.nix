{
  description = "Application for a personal assistant";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
      in
      {
        packages = {
          myapp = mkPoetryApplication { projectDir = self; };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.myapp ];
          # we need blas/lapack libraries for scipy dependency
          # added this from source: https://discourse.nixos.org/t/problem-installing-scipy-with-poetry-install-a-nix-shell/17830/4
          packages = with pkgs; [
            # switching to python 3.10 because of failing scipy install- https://github.com/cython/cython/pull/4428#issuecomment-1682593857
            python310
            poetry
            gfortran
            pkg-config
            lapack-reference
          ];

          shellHook = ''
            export BLAS="${pkgs.lapack-reference}/lib/libblas.dylib"
            export LAPACK="${pkgs.lapack-reference}/lib/liblapack.dylib"
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib"
          '';
        };
      });
}
