#!/usr/bin/env python3
"""Install one exact dependency profile from requirements.lock with this Python."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

if sys.version_info < (3, 11):
    raise SystemExit("The dependency installer requires Python 3.11 or newer")

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with (ROOT / "requirements.lock").open("rb") as stream:
        profiles = tomllib.load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(profiles))
    parser.add_argument("--show", action="store_true",
                        help="Print the selected pins and platform information without installing")
    args = parser.parse_args()
    profile = profiles[args.profile]
    options = profile.get("pip_options", [])
    requirements = profile["requirements"]
    if args.show:
        print(f"# Python {profile['python']}; CUDA {profile['cuda']}")
        print(f"# Platform: {profile['platform']}")
        if options:
            print(" ".join(options))
        print("\n".join(requirements))
        return
    if sys.prefix == sys.base_prefix:
        parser.error("Run this script with the Python interpreter of a virtual environment")
    if platform.python_implementation() != "CPython" or platform.python_version() != profile["python"]:
        parser.error(f"This profile requires CPython {profile['python']}; "
                     f"current interpreter is {platform.python_implementation()} {platform.python_version()}")
    if args.profile == "supervised-cu128":
        libc, version = platform.libc_ver()
        try:
            libc_version = tuple(int(part) for part in version.split("."))
        except ValueError:
            libc_version = ()
        if (sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}
                or sysconfig.get_config_var("Py_GIL_DISABLED")
                or libc != "glibc" or libc_version < (2, 28)):
            parser.error("supervised-cu128 requires Linux x86_64, glibc >= 2.28 and CPython with GIL")
    elif args.profile.startswith("llm-") and sys.platform not in {"linux", "darwin"}:
        parser.error("The LLM profiles are pinned for Linux and macOS")
    result = subprocess.run([sys.executable, "-m", "pip", "install",
                             *options, *requirements], check=False)
    if result.returncode == 0:
        result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
