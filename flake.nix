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
        inherit (pkgs.stdenv) isAarch32 isAarch64 isDarwin;

        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        nativeBuildInputs = with pkgs; [
          stdenv
          python310
          poetry
        ];
        buildInputs = with pkgs; [
          tesseract # needed by unstructured.io to parse pdf and image files
          poppler_utils # needed by unstructured.io to parse pdf files
          pandoc # needed by unstructured.io to parse EPUBs, RTFs and Open Office docs
        ];
        osSpecific = with pkgs; (
          if isAarch64 && isDarwin then
            with pkgs.darwin.apple_sdk_11_0.frameworks; [
              Accelerate
              MetalKit
            ]
          else if isAarch32 && isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else if isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else
            [symlinkJoin {
              # HACK(Green-Sky): nix currently has issues with cmake findcudatoolkit
              # see https://github.com/NixOS/nixpkgs/issues/224291
              # copied from jaxlib
              name = "${cudaPackages.cudatoolkit.name}-merged";
              paths = [
                cudaPackages.cudatoolkit.lib
                cudaPackages.cudatoolkit.out
              ] ++ lib.optionals (lib.versionOlder cudaPackages.cudatoolkit.version "11") [
                # for some reason some of the required libs are in the targets/x86_64-linux
                # directory; not sure why but this works around it
                "${cudaPackages.cudatoolkit}/targets/${system}"
              ];
            }]
        );
        cmakeFlags = if isDarwin then [] else ["-DLLAMA_CUBLAS=ON"];
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; })
          mkPoetryApplication;
      in {
        inherit nativeBuildInputs buildInputs osSpecific cmakeFlags;

        packages = {
          myapp = mkPoetryApplication {
            projectDir = self;
            python = pkgs.python310;
          };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          # inputsFrom = [ self.packages.${system}.myapp ];
          packages = nativeBuildInputs ++ buildInputs ++ osSpecific;
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
      });
}
