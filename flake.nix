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
          requests
          result
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonWithPackages
          ];

          shellHook = ''
            echo "Python version: $(python --version)"
            python -c "import json5; print(f'  - json5 {json5.__version__}')"
            python -c "import matplotlib; print(f'  - matplotlib {matplotlib.__version__}')"
            python -c "import numpy; print(f'  - numpy {numpy.__version__}')"
            python -c "import openai; print(f'  - openai {openai.__version__}')"
            python -c "import requests; print(f'  - requests {requests.__version__}')"
            python -c "import result; print(f'  - result {result.__version__}')"
          '';
        };
      });
}