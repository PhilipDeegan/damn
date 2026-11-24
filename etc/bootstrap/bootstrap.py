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
system), nothing built. Each run has its own directory, .damn/bootstrap/<name> (--name; default
cpu, cuda or rocm: configurations side by side); without -x an mkn settings file is written
there, toolchain.yaml, for the probes and for later builds: -x .damn/bootstrap/<name>/toolchain.yaml
(its inc: has the generated header). With your own -x: -x <it> -f .damn/bootstrap/<name>/inc.

Probes: etc/bootstrap/<area>/<name>.cpp, a leading comment with "ranks=<n> profile=<mkn profile>
[mpirun=<args,comma,separated>] [needs=<fact,comma,separated>]" (on any of its lines: keep each
short, clang-format reflows long ones), needs: toolchain facts (discover.facts: hdf5_parallel,
mpi_cuda_aware, ...) without which it is skipped, not built,
stdout "key=value" lines. Each is built with mkn (-p <profile> -M <probe>: the build's
compilers, flags and find paths; the probes' own project, etc/bootstrap/mkn.yaml, mkn -C;
copied to <run>/bin), run with
mpirun [<mpirun args>] -n <ranks> (--mpirun) in its own directory under <run>/run, its
output kept in <run>/<area>_<name>.log. mpirun's args place the ranks, e.g.
mpirun=--map-by,numa,--bind-to,numa: a machine that cannot (one memory domain) fails the probe.
Writes

    <run>/inc/damn/core/def/_gen_bootstrap.hpp
                                           compile time: DAMN_HAVE_<AREA>_<NAME> 1 / 0 per
                                           probe (for #if), damn::bootstrap::<Area>::<name>
                                           bools (each probe passed) and Toolchain (discover.py's
                                           flags, gpu); run time: damn::bootstrap::entries, every
                                           fact a string by path ("mpi/exchange_node/latency_s",
                                           "toolchain/mpi/version"), read with damn::bon::get<T>
                                           or as a tree (damn::bon::tree)
    <run>/bootstrap.json                   the same facts, nested by path

The header is this machine's and toolchain's, outside the source tree: damn/core/def/bootstrap.hpp
includes it when the build's include paths have it, else the committed damn/core/def/bootstrap_defaults.hpp (every capability off, made by
--defaults: rerun it when probes or toolchain flags change). Run from the damn root. What it
says the build can do, a run still checks for itself (another node, another thread level)

A project built on damn has probes of its own (plamn: its kernels' costs) and runs them with this
code: its etc/bootstrap/bootstrap.py calls main(Project(root, name, header)), writing its own
header (<NAME>_HAVE_*, <name>::bootstrap) under .<name>/bootstrap/<run>
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
        self.base = self.root / f".{name}" / "bootstrap"
        self.include = Path(header)  # as included: <name>/core/def/_gen_bootstrap.hpp
        self.at(self.base)
        self.guard = f"{name.upper()}_CORE_DEF_GEN_BOOTSTRAP_HPP"
        # the committed header for builds without a bootstrap, beside the generated one
        self.defaults = (
            self.root / "inc" / self.include.parent / "bootstrap_defaults.hpp"
        )
        self.defaults_guard = f"{name.upper()}_CORE_DEF_BOOTSTRAP_DEFAULTS_HPP"

    def at(self, out):
        """a run's directory: its logs, binaries, toolchain.yaml, bootstrap.json and inc/ (the
        generated header, outside the source tree: the toolchain's inc, or mkn -f <inc>)
        """
        self.out = Path(out)
        self.inc = self.out / "inc"
        self.header = self.inc / self.include


DAMN = Project(ROOT, "damn", "damn/core/def/_gen_bootstrap.hpp")


def metadata(cpp):
    """a probe's ranks / profile / mpirun / needs, from its leading // comment block (any line:
    clang-format may reflow them)"""
    block = []
    for line in cpp.read_text().splitlines():
        if not line.startswith("//"):
            break
        block.append(line)
    keys = r"\b(ranks|profile|mpirun|needs)=(\S+)"
    return dict(re.findall(keys, "\n".join(block)))


def probes(project, only):
    for cpp in sorted(project.probes.glob("*/*.cpp")):
        area = cpp.parent.name
        if only and area not in only:
            continue
        meta = metadata(cpp)
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
    """the probe as the profile's binary, built in the probes' own project (etc/bootstrap/mkn.yaml,
    mkn -C: binaries in etc/bootstrap/bin/<profile>), copied out before the next one builds over
    it: no mkn -N needed"""
    out = project.probes / "bin" / p["profile"]
    shutil.rmtree(out, ignore_errors=True)  # no stale binary of another probe
    cmd = ["mkn", "build", "-C", str(project.probes / "mkn.yaml"), "-p", p["profile"]]
    cmd += [
        "-M",
        str(p["path"].relative_to(project.probes)),
        "-O",
        "2",
        "-x",
        args.settings,
    ]
    r = subprocess.run(cmd, cwd=project.probes, capture_output=True, text=True)
    log.write(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}\n")
    exes = [f for f in out.glob("*") if f.is_file() and f.stat().st_mode & 0o111]
    if r.returncode or not exes:
        return None
    exe = project.out / "bin" / f"{p['area']}_{p['name']}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exes[0], exe)
    return exe


def settings_path(x):
    """-x as an absolute path when it names a file from here (the probes build from
    etc/bootstrap), else as given (mkn looks in its settings directory)"""
    for f in (Path(x), Path(f"{x}.yaml"), Path(f"{x}.yml")):
        if f.is_file():
            return str(f.resolve().with_suffix(""))
    return x


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
    """every fact as a string by path: "settings", "toolchain/<area>/<key>" (discover.py's),
    "<area>/<probe>/ok" (1 / 0) and "<area>/<probe>/<key>" (its output)"""
    out = [("settings", settings)]
    for k, v in tool.items():
        area, _, key = k.partition("_")
        path = f"toolchain/{area}/{key}" if key else f"toolchain/{area}"
        out.append((path, str(int(v)) if isinstance(v, bool) else str(v)))
    for r in results:
        out.append((f"{r['area']}/{r['name']}/ok", str(int(r["ok"]))))
        out += [(f"{r['area']}/{r['name']}/{k}", v) for k, v in r["values"].items()]
    return out


def nested(table):
    """entries as nested dicts (bootstrap.json), as damn::bon::tree"""
    root = {}
    for path, value in table:
        *parents, leaf = path.split("/")
        node = root
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return root


def raw(v):
    """a C++ raw string literal of v"""
    return f'R"bs({v})bs"'


def area_struct(area):
    return area[:1].upper() + area[1:]


def header(project, results, settings, tool, defaults=False):
    """the generated header (or, defaults, the committed one for builds without a bootstrap):
    macros for #if, a struct per area of compile time facts (bools; Toolchain's gpu), every fact
    as a string in entries"""
    table = [] if defaults else entries(results, settings, tool)
    prefix, ns = project.name.upper(), f"{project.name}::bootstrap"
    guard = project.defaults_guard if defaults else project.guard
    what = (
        "every capability off, for builds without a bootstrap: regenerate (--defaults) when "
        "probes or toolchain flags change"
        if defaults
        else f"this machine's and build's ({settings}). Not committed: rerun to update"
    )
    out = [
        f"// generated by etc/bootstrap/bootstrap.py{' --defaults' if defaults else ''}: {what}",
        f"// see {project.name}/core/def/bootstrap.hpp",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <array>",
        "#include <string_view>",
        "#include <utility>",
        "",
        f"#define {prefix}_HAVE_BOOTSTRAP {0 if defaults else 1}",
    ]
    for r in results:
        out.append(
            f"#define {prefix}_HAVE_{r['area'].upper()}_{r['name'].upper()} {int(r['ok'])}"
        )
    out += ["", f"namespace {ns}", "{"]
    out += ["// what discover.py found", "struct Toolchain", "{"]
    for k, v in tool.items():
        if isinstance(v, bool):
            out.append(f"    static constexpr bool {k} = {'true' if v else 'false'};")
    out.append(f"    static constexpr std::string_view gpu = {raw(tool['gpu'])};")
    out.append("};")
    areas = {}
    for r in results:
        areas.setdefault(r["area"], []).append(r)
    for area, rs in areas.items():
        out += [
            "",
            f"// etc/bootstrap/{area}: each probe passed",
            f"struct {area_struct(area)}",
            "{",
        ]
        out += [
            f"    static constexpr bool {r['name']} = {'true' if r['ok'] else 'false'};"
            for r in rs
        ]
        out.append("};")
    out += [
        "",
        '// every fact as a string by path ("settings", "toolchain/<area>/<key>",',
        '//  "<area>/<probe>/ok", "<area>/<probe>/<key>"): damn::bon::get / damn::bon::tree',
        f"inline constexpr std::array<std::pair<std::string_view, std::string_view>, {len(table)}>"
        + (" entries{};" if not table else " entries{{"),
    ]
    if table:
        out += [f"    {{{raw(k)}, {raw(v)}}}," for k, v in table]
        out.append("}};")
    out += [f"}} // namespace {ns}", "", f"#endif // {guard}", ""]
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
    a.add_argument(
        "--name",
        help="this run's directory, .<project>/bootstrap/<name> (default: cpu, cuda or rocm)",
    )
    a.add_argument("--only", nargs="*", help="areas (mpi, hdf5, cuda, ...)")
    a.add_argument(
        "--defaults",
        action="store_true",
        help="write the committed header for builds without a bootstrap (all off), nothing run",
    )
    discover.add_arguments(a)
    args = a.parse_args(argv)

    if args.defaults:
        results = [
            dict(area=p["area"], name=p["name"], ok=False, values={}, why="")
            for p in probes(project, None)
        ]
        project.defaults.write_text(
            header(project, results, "", discover.no_facts(), defaults=True)
        )
        print(f"wrote {project.defaults.relative_to(project.root)}")
        return

    found = discover.discover(args)
    tool = discover.facts(found)
    name = args.name or ("cpu" if found["gpu"] == "none" else found["gpu"])
    project.at(project.base / name)
    project.out.mkdir(parents=True, exist_ok=True)
    project.header.parent.mkdir(parents=True, exist_ok=True)  # mkn: an inc: must exist
    print(discover.report(found))
    generated = not args.settings
    if generated:
        out = (project.out / "toolchain.yaml").resolve()
        command = " ".join(["etc/bootstrap/bootstrap.py", *(argv or sys.argv[1:])])
        out.write_text(discover.settings(found, args, command, [project.inc.resolve()]))
        args.settings = str(out)
        print(f"settings {out}")
    args.settings = settings_path(args.settings)
    if not args.mpirun:
        args.mpirun = found["mpi"].get("mpirun", "mpirun")
    env = discover.run_env(found)
    print()

    results = []
    for p in probes(project, None):  # every probe in the header: its structs complete
        key = f"{p['area']}_{p['name']}"
        if args.only and p["area"] not in args.only:
            results.append(
                dict(area=p["area"], name=p["name"], ok=False, values={}, why="--only")
            )
            continue
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

    project.header.parent.mkdir(parents=True, exist_ok=True)
    project.header.write_text(header(project, results, args.settings, tool))
    (project.out / "bootstrap.json").write_text(
        json.dumps(nested(entries(results, args.settings, tool)), indent=1)
    )
    print(
        f"wrote {project.header.relative_to(project.root)}, "
        f"{(project.out / 'bootstrap.json').relative_to(project.root)}"
    )
    inc = project.inc.resolve()
    print(
        f"build with: -x {args.settings}"
        if generated
        else f"build with: -x {args.settings} -f {inc}"
    )


if __name__ == "__main__":
    main()
