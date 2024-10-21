# fling-py

## Usage

To run the project, first build the dependencies with the following commands in the root directory:

```
sudo nixos-rebuild switch -I nixos-config=configuration.nix
poetry install
```

Then, run this command to actually run the project:

```
poetry run python main.py
```
