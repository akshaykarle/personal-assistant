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
        nativeBuildInputs = with pkgs; [
          stdenv
          python310
          poetry
          tesseract # needed by unstructured.io to parse pdf and image files
          poppler_utils # needed by unstructured.io to parse pdf files
          pandoc # needed by unstructured.io to parse EPUBs, RTFs and Open Office docs
        ];
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; })
          mkPoetryApplication;
      in {
        inherit nativeBuildInputs;

        packages = {
          myapp = mkPoetryApplication {
            projectDir = self;
            python = pkgs.python310;
          };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          # inputsFrom = [ self.packages.${system}.myapp ];
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          packages = nativeBuildInputs;
        };
      });
}
