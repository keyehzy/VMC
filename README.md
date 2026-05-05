# VMC

Variational Monte Carlo experiments in C++.

## Development

The project uses CMake and a small wrapper script for the common development
cycle: format, build, then test.

```bash
tools/make.sh
```

Useful commands:

```bash
tools/make.sh format
tools/make.sh build
tools/make.sh test
tools/make.sh clean
```

The default build type is `Debug`. Override it with `BUILD_TYPE` when needed:

```bash
BUILD_TYPE=Release tools/make.sh test
```
