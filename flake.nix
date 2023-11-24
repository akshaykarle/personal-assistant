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
          myapp = mkPoetryApplication {
            projectDir = self;
            python = pkgs.python311;
          };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          # inputsFrom = [ self.packages.${system}.myapp ];
          packages = with pkgs; [
            python311
            poetry
            tesseract # needed by unstructured.io to parse pdf and image files
            poppler_utils # needed by unstructured.io to parse pdf files
            pandoc # needed by unstructured.io to parse EPUBs, RTFs and Open Office docs
          ];
        };
      });
}
