{
  description = "GPU bugs analysis environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        
        pythonPackages = pkgs.python311Packages;
        pythonWithPackages = pythonPackages.python.withPackages (ps: with ps; [
          json5
          matplotlib
          numpy
          openai
          pandas
          requests
          result
          scipy
          seaborn
          scikit-learn
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonWithPackages
          ];

          shellHook = ''
            echo "Python environment ready with analysis packages"
          '';
        };
      });
}