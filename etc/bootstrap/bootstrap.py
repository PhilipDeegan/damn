#!/usr/bin/env python3
"""damn's bootstrap: what this machine and build can do, as PHARE's tools/config, one probe binary
per capability so each passes or fails alone (a probe that does not build, exits non zero,
aborts or times out: not available)

    python3 etc/bootstrap/bootstrap.py                        # the system's MPI / HDF5, CPU
    python3 etc/bootstrap/bootstrap.py --mpi-home /opt/hpc/mpi --gpu cuda
                                                  # that MPI, the HDF5 beside it, CUDA (clang)
    python3 etc/bootstrap/bootstrap.py -x res/mkn/hpc         # an mkn settings file of your own
    python3 etc/bootstrap/bootstrap.py --only mpi hdf5        # some areas

Toolchain (discover.py): MPI, HDF5 and the GPU toolkits are found first (options, environment,
system), nothing built; without -x an mkn settings file is written for them,
.damn/bootstrap/toolchain.yaml, for the probes and for later builds (-x <that file>).

Probes: etc/bootstrap/<area>/<name>.cpp, a first line comment
"// bootstrap probe ...: ranks=<n> profile=<mkn profile> [mpirun=<args,comma,separated>]
[needs=<fact,comma,separated>]", needs: toolchain facts (discover.facts: hdf5_parallel,
mpi_cuda_aware, ...) without which it is skipped, not built,
stdout "key=value" lines. Each is built with mkn (-p <profile> -M <probe>: the build's
compilers, flags and find paths, copied to .damn/bootstrap/bin), run with
mpirun [<mpirun args>] -n <ranks> (--mpirun) in its own directory under .damn/bootstrap/run, its
output kept in .damn/bootstrap/<area>_<name>.log. mpirun's args place the ranks, e.g.
mpirun=--map-by,numa,--bind-to,numa: a machine that cannot (one memory domain) fails the probe.
Writes

    inc/damn/core/def/_gen_bootstrap.hpp   DAMN_HAVE_<AREA>_<NAME> 1 / 0 per probe,
                                           damn::bootstrap::toolchain::<fact> typed constexpr
                                           (what discover.py found: gpu, mpi_impl, ...),
                                           damn::bootstrap::<area>_<name>_<key> strings, and
                                           every fact in damn::bootstrap::entries (config():
                                           a map, e.g. for python, as PHARE's build_config)
    .damn/bootstrap/bootstrap.json         the same map: "<area>_<name>": "1" / "0",
                                           "<area>_<name>_<key>": value, "settings"

The header is this machine's: not committed (.gitignore); damn/core/def/bootstrap.hpp includes it
if there, every DAMN_HAVE_* 0 otherwise. Run from the damn root. What it says the build can do,
a run still checks for itself (another node, another thread level)

A project built on damn has probes of its own (plamn: its kernels' costs) and runs them with this
code: its etc/bootstrap/bootstrap.py calls main(Project(root, name, header)), writing its own
header (<NAME>_HAVE_*, <name>::bootstrap) and .<name>/bootstrap
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Project:
    """whose probes: its root, its probes (etc/bootstrap/<area>/<name>.cpp, built with its
    mkn.yaml's profiles), its generated header (relative to root), its name: the macro prefix
    (<NAME>_HAVE_*), namespace (<name>::bootstrap) and output (.<name>/bootstrap)"""

    def __init__(self, root, name, header):
        self.root = Path(root)
        self.name = name
        self.probes = self.root / "etc" / "bootstrap"
        self.out = self.root / f".{name}" / "bootstrap"
        self.header = self.root / header
        self.guard = f"{name.upper()}_CORE_DEF_GEN_BOOTSTRAP_HPP"


DAMN = Project(ROOT, "damn", "inc/damn/core/def/_gen_bootstrap.hpp")


def probes(project, only):
    for cpp in sorted(project.probes.glob("*/*.cpp")):
        area = cpp.parent.name
        if only and area not in only:
            continue
        meta = dict(re.findall(r"(\w+)=(\S+)", cpp.read_text().splitlines()[0]))
        yield dict(
            area=area,
            name=cpp.stem,
            path=cpp,
            ranks=int(meta.get("ranks", 1)),
            profile=meta.get("profile", "bootstrap"),
            mpirun=[x for x in meta.get("mpirun", "").split(",") if x],
            needs=[x for x in meta.get("needs", "").split(",") if x],
        )


def build(project, p, args, log):
    """the probe as the profile's binary (bin/<profile>), copied out before the next one builds
    over it: no mkn -N needed"""
    out = project.root / "bin" / p["profile"]
    shutil.rmtree(out, ignore_errors=True)  # no stale binary of another probe
    cmd = [
        "mkn",
        "build",
        "-p",
        p["profile"],
        "-M",
        str(p["path"].relative_to(project.root)),
    ]
    cmd += ["-O", "2", "-x", args.settings]
    r = subprocess.run(cmd, cwd=project.root, capture_output=True, text=True)
    log.write(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}\n")
    exes = [f for f in out.glob("*") if f.is_file() and f.stat().st_mode & 0o111]
    if r.returncode or not exes:
        return None
    exe = project.out / "bin" / f"{p['area']}_{p['name']}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exes[0], exe)
    return exe


def run(project, p, exe, args, log, env):
    work = project.out / "run" / f"{p['area']}_{p['name']}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cmd = [args.mpirun, *p["mpirun"], "-n", str(p["ranks"]), str(exe)]
    try:
        r = subprocess.run(
            cmd, cwd=work, capture_output=True, text=True, timeout=args.timeout, env=env
        )
    except subprocess.TimeoutExpired as e:
        log.write(
            f"$ {' '.join(cmd)}\ntimed out after {args.timeout}s\n{e.stdout or ''}\n"
        )
        return False, {}, "timed out"
    log.write(f"$ {' '.join(cmd)}\nexit {r.returncode}\n{r.stdout}{r.stderr}\n")
    values = {}
    for line in r.stdout.splitlines():  # every rank prints: the first of each key
        key, eq, value = line.partition("=")
        if eq and re.fullmatch(r"\w+", key):
            values.setdefault(key, value.strip())
    shutil.rmtree(work, ignore_errors=True)
    return r.returncode == 0, values, f"exit {r.returncode}"


def entries(results, settings, tool):
    """every fact as "<area>_<name>" (1 / 0) and "<area>_<name>_<key>" (its values), settings,
    and the toolchain's as "toolchain_<fact>" """
    out = [("settings", settings)]
    out += [
        (f"toolchain_{k}", str(int(v)) if isinstance(v, bool) else str(v))
        for k, v in tool.items()
    ]
    for r in results:
        key = f"{r['area']}_{r['name']}"
        out.append((key, str(int(r["ok"]))))
        out += [(f"{key}_{k}", v) for k, v in r["values"].items()]
    return out


def cpp(v):
    """a fact's C++ type and literal"""
    if isinstance(v, bool):
        return "bool", "true" if v else "false"
    if isinstance(v, int):
        return "int", str(v)
    return (
        "std::string_view",
        '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"',
    )


def header(project, results, settings, tool):
    quote = (
        lambda v: '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    )  # noqa: E731
    table = entries(results, settings, tool)
    prefix, ns = project.name.upper(), f"{project.name}::bootstrap"
    out = [
        "// generated by etc/bootstrap/bootstrap.py ("
        + settings
        + "): this machine's and build's",
        f"// capabilities, see {project.name}/core/def/bootstrap.hpp. Not committed: rerun to "
        "update",
        f"#ifndef {project.guard}",
        f"#define {project.guard}",
        "",
        "#include <array>",
        "#include <string_view>",
        "#include <utility>",
        "",
        f"#define {prefix}_HAVE_BOOTSTRAP 1",
    ]
    for r in results:
        out.append(
            f"#define {prefix}_HAVE_{r['area'].upper()}_{r['name'].upper()} {int(r['ok'])}"
        )
    out += ["", f"namespace {ns}::toolchain", "{"]
    out += ["// what discover.py found (etc/bootstrap/discover.py: facts)"]
    for k, v in tool.items():
        t, lit = cpp(v)
        out.append(f"inline constexpr {t} {k} = {lit};")
    out += [f"}} // namespace {ns}::toolchain"]
    out += ["", f"namespace {ns}", "{"]
    for r in results:
        for k, v in r["values"].items():
            out.append(
                f'inline constexpr std::string_view {r["area"]}_{r["name"]}_{k} = {quote(v)};'
            )
    out += [
        "",
        f"// every fact, as bootstrap.json: {ns}::config() makes a map of it",
        f"inline constexpr std::array<std::pair<std::string_view, std::string_view>, {len(table)}> entries{{{{",
    ]
    out += [f"    {{{quote(k)}, {quote(v)}}}," for k, v in table]
    out += [
        "}};",
        f"}} // namespace {ns}",
        "",
        f"#endif // {project.guard}",
        "",
    ]
    return "\n".join(out)


def main(project=DAMN, argv=None):
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument(
        "-x",
        "--settings",
        help="mkn settings (compilers, find); default: generated, see below",
    )
    a.add_argument(
        "--mpirun", help="the settings' MPI's launcher (default: the found MPI's)"
    )
    a.add_argument("--timeout", type=float, default=60)
    a.add_argument("--only", nargs="*", help="areas (mpi, hdf5, cuda, ...)")
    discover.add_arguments(a)
    args = a.parse_args(argv)

    project.out.mkdir(parents=True, exist_ok=True)
    found = discover.discover(args)
    tool = discover.facts(found)
    print(discover.report(found))
    if not args.settings:
        out = Path(args.out_settings or project.out / "toolchain.yaml").resolve()
        command = " ".join(["etc/bootstrap/bootstrap.py", *(argv or sys.argv[1:])])
        out.write_text(discover.settings(found, args, command))
        args.settings = str(out)
        print(f"settings {out}")
    if not args.mpirun:
        args.mpirun = found["mpi"].get("mpirun", "mpirun")
    env = discover.run_env(found)
    print()

    results = []
    for p in probes(project, args.only):
        key = f"{p['area']}_{p['name']}"
        missing = [n for n in p["needs"] if not tool.get(n)]
        if missing:
            results.append(
                dict(
                    area=p["area"],
                    name=p["name"],
                    ok=False,
                    values={},
                    why=f"skipped: no {', '.join(missing)}",
                )
            )
            print(f"{'no':<4} {key:<22} skipped: no {', '.join(missing)}")
            continue
        with open(project.out / f"{key}.log", "w") as log:
            exe = build(project, p, args, log)
            ok, values, why = (
                run(project, p, exe, args, log, env)
                if exe
                else (False, {}, "build failed")
            )
        results.append(
            dict(area=p["area"], name=p["name"], ok=ok, values=values, why=why)
        )
        detail = ", ".join(f"{k}={v}" for k, v in values.items())
        print(
            f"{'yes' if ok else 'no ':<4} {key:<22} {detail if ok else why + ('; ' + detail if detail else '')}"
        )
        sys.stdout.flush()

    project.header.write_text(header(project, results, args.settings, tool))
    (project.out / "bootstrap.json").write_text(
        json.dumps(dict(entries(results, args.settings, tool)), indent=1)
    )
    print(
        f"wrote {project.header.relative_to(project.root)}, "
        f"{(project.out / 'bootstrap.json').relative_to(project.root)}"
    )


if __name__ == "__main__":
    main()
