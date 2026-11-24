"""bootstrap's inputs: where MPI, HDF5 and a GPU toolkit are (options, then their environment
variables, then the system's), what each is (read from its files and wrappers, nothing built),
and an mkn settings file for that toolchain when none is given (-x)

    --mpi-home DIR      (MPI_HOME, MPI_ROOT; else the mpicxx on PATH)   DIR/bin/mpicxx, mpirun
    --hdf5-root DIR     (HDF5_ROOT, HDF5_DIR; else the HDF5 built with that MPI: DIR/../hdf5,
                         then the system's)
    --gpu [cuda|rocm|auto]      (absent: a CPU build; bare --gpu: auto, a toolkit with a device,
                                 else CPU)
    --compiler gcc|clang|nvcc|hipcc   the family (default: gcc for CPU, clang for CUDA, hipcc
                                 for ROCm; nvcc runs over the MPI wrapper, -ccbin, and does
                                 not build damn's device code yet: damn#68); --cxx its binary
    --cuda-home DIR     (CUDA_HOME, CUDA_PATH; else /usr/local/cuda)  --cuda-arch sm_120 (else
                         nvptx-arch's first device)
    --rocm-path DIR     (ROCM_PATH; else /opt/rocm)  --gpu-targets gfx942 (else amdgpu-arch's)
    --cxx COMPILER      the MPI wrapper's compiler (OMPI_CXX / MPICH_CXX): g++ for none, clang++
                         for cuda, hipcc for rocm by default

so `--mpi-home /opt/hpc/mpi --gpu cuda` is the CUDA aware MPI with the HDF5 next to it, and no
option at all the system's MPI and HDF5 for a CPU build
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ARCH = os.uname().machine

# the system's parallel HDF5 (Debian / Ubuntu, Fedora / RHEL's openmpi flavour), (include, lib)
SYSTEM_HDF5 = [
    ("/opt/hdf5/include", "/opt/hdf5/lib"),
    ("/usr/include/hdf5/openmpi", f"/usr/lib/{ARCH}-linux-gnu/hdf5/openmpi"),
    (f"/usr/include/openmpi-{ARCH}", "/usr/lib64/openmpi/lib"),
]


# compiler families and the targets they build for
COMPILERS = dict(gcc=("none",), clang=("none", "cuda"), nvcc=("cuda",), hipcc=("rocm",))


def add_arguments(a):
    g = a.add_argument_group(
        "toolchain (when no -x is given, a settings file is made of it)"
    )
    g.add_argument("--mpi-home", help="MPI install prefix (bin/mpicxx)")
    g.add_argument("--mpi-include-dir", help="override the wrapper's include dirs")
    g.add_argument("--mpi-library-dir", help="override the wrapper's library dirs")
    g.add_argument("--hdf5-root", help="HDF5 install prefix (include, lib)")
    g.add_argument("--hdf5-include-dir")
    g.add_argument("--hdf5-library-dir")
    g.add_argument(
        "--gpu",
        nargs="?",
        const="auto",
        default="none",
        choices=["none", "cpu", "cuda", "rocm", "auto"],
        help="absent: a CPU build; bare --gpu: auto (a toolkit with a device, else CPU)",
    )
    g.add_argument(
        "--compiler",
        choices=sorted(COMPILERS),
        help="compiler family (default: gcc for CPU, clang for CUDA, hipcc for ROCm)",
    )
    g.add_argument("--cuda-home")
    g.add_argument("--cuda-arch", help="e.g. sm_120")
    g.add_argument("--rocm-path")
    g.add_argument("--gpu-targets", help="e.g. gfx942")
    g.add_argument(
        "--cxx",
        help="the compiler binary under the MPI wrapper (nvcc: its host compiler)",
    )


def sh(*cmd, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def first(*values):
    return next((v for v in values if v), None)


def same_dir(a, b):
    return Path(a).resolve() == Path(b).resolve()


def defines(header):
    """#define NAME VALUE lines of a header (H5pubconf.h, mpi-ext.h), "" for a bare #define"""
    if not Path(header).is_file():
        return {}
    return dict(
        re.findall(r"^#define\s+(\w+)[ \t]*(.*?)\s*$", Path(header).read_text(), re.M)
    )


def find_mpi(args):
    home = first(args.mpi_home, os.environ.get("MPI_HOME"), os.environ.get("MPI_ROOT"))
    if home:
        cxx = Path(home) / "bin" / "mpicxx"
    else:
        which = shutil.which("mpicxx")
        if not which:
            return dict(found=False, why="no mpicxx on PATH, no --mpi-home")
        cxx = Path(which)
        home = str(cxx.parent.parent)
    if not cxx.exists():
        return dict(found=False, why=f"no {cxx}")
    bindir = cxx.parent
    version = sh(str(cxx), "--showme:version")  # Open MPI
    if version:
        impl = "OMPI"
        inc = (sh(str(cxx), "--showme:incdirs") or "").split()
        lib = (sh(str(cxx), "--showme:libdirs") or "").split()
        m = re.search(
            r"\d+(\.\d+)+", version
        )  # "mpicxx: Open MPI 5.0.7 (Language: C++)"
        version = m.group(0) if m else version
    else:  # MPICH and its derivatives: -show prints the compile line
        show = sh(str(cxx), "-show") or ""
        impl = "MPICH" if show else "unknown"
        inc = re.findall(r"-I(\S+)", show)
        lib = re.findall(r"-L(\S+)", show)
        version = first(sh(str(bindir / "mpichversion")), "unknown")
    inc = args.mpi_include_dir.split(":") if args.mpi_include_dir else inc
    lib = args.mpi_library_dir.split(":") if args.mpi_library_dir else lib
    ext = {}
    for d in inc:  # Open MPI: mpi-ext.h includes one header per extension
        for h in Path(d).glob("**/mpiext_*.h") if Path(d).is_dir() else []:
            ext.update(defines(h))
        ext.update(defines(Path(d) / "mpi-ext.h"))
    return dict(
        found=True,
        home=str(Path(home).resolve()),
        bin=str(bindir),
        cxx=str(cxx),
        mpirun=str(bindir / "mpirun"),
        impl=impl,
        version=version,
        inc=inc,
        lib=lib,
        cuda_aware=ext.get("MPIX_CUDA_AWARE_SUPPORT") == "1",
        rocm_aware=ext.get("MPIX_ROCM_AWARE_SUPPORT") == "1",
    )


def hdf5_at(inc, lib):
    conf = defines(Path(inc) / "H5pubconf.h")
    if not conf:
        return None
    built = ""
    settings = Path(lib) / "libhdf5.settings"
    if settings.is_file():
        m = re.search(r"^\s*C Compiler:\s*(\S+)", settings.read_text(), re.M)
        built = m.group(1) if m else ""
    return dict(
        found=True,
        inc=str(inc),
        lib=str(lib),
        version=conf.get("H5_VERSION", "").strip('"'),
        parallel="H5_HAVE_PARALLEL" in conf,
        subfiling="H5_HAVE_SUBFILING_VFD" in conf,
        built_with=built,  # its MPI compiler (libhdf5.settings), to match ours
    )


def find_hdf5(args, mpi):
    def matches(h):  # built with this MPI (its mpicc in the same bin dir)
        return (
            mpi.get("found")
            and h["built_with"]
            and same_dir(Path(h["built_with"]).parent, mpi["bin"])
        )

    root = first(
        args.hdf5_root, os.environ.get("HDF5_ROOT"), os.environ.get("HDF5_DIR")
    )
    if args.hdf5_include_dir or root:
        h = hdf5_at(
            first(args.hdf5_include_dir, root and f"{root}/include"),
            first(args.hdf5_library_dir, root and f"{root}/lib"),
        )
        if not h:
            return dict(
                found=False, why=f"no H5pubconf.h under {root or args.hdf5_include_dir}"
            )
        h["matches_mpi"] = bool(matches(h))
        return h
    candidates = []
    if mpi.get(
        "found"
    ):  # an HDF5 installed beside the MPI (/opt/hpc/{mpi,hdf5}), or in it
        up = Path(mpi["home"]).parent
        candidates += [(up / "hdf5" / "include", up / "hdf5" / "lib")]
        candidates += [(Path(mpi["home"]) / "include", Path(mpi["home"]) / "lib")]
    candidates += SYSTEM_HDF5
    found = [h for h in (hdf5_at(i, l) for i, l in candidates) if h]
    if not found:
        return dict(found=False, why="no HDF5 found (--hdf5-root)")
    h = next((h for h in found if matches(h)), found[0])
    h["matches_mpi"] = bool(matches(h))
    return h


def llvm_tool(name):
    """clang's offload arch tools: on PATH, else beside the newest /usr/lib/llvm-*"""
    which = shutil.which(name)
    if which:
        return which
    tools = sorted(
        Path("/usr/lib").glob(f"llvm-*/bin/{name}"),
        key=lambda p: int(re.sub(r"\D", "", p.parts[-3]) or 0),
    )
    return str(tools[-1]) if tools else None


def find_cuda(args):
    home = first(
        args.cuda_home,
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    )
    version_json = Path(home) / "version.json"
    if not (Path(home) / "include" / "cuda_runtime.h").exists():
        return dict(found=False, why=f"no CUDA toolkit at {home}")
    version = ""
    if version_json.is_file():
        version = (
            json.loads(version_json.read_text()).get("cuda", {}).get("version", "")
        )
    devices = []
    tool = llvm_tool("nvptx-arch")
    if tool:
        devices = (sh(tool) or "").split()
    arch = first(args.cuda_arch, devices and devices[0])
    target = Path(home) / "targets" / f"{ARCH}-linux"
    return dict(
        found=True,
        home=str(Path(home).resolve()),
        version=version,
        devices=devices,
        arch=arch or "",
        cccl=str(target / "include" / "cccl"),
        lib=str(target / "lib") if (target / "lib").is_dir() else f"{home}/lib64",
    )


def find_rocm(args):
    home = first(args.rocm_path, os.environ.get("ROCM_PATH"), "/opt/rocm")
    if not (Path(home) / "bin" / "hipcc").exists():
        return dict(found=False, why=f"no hipcc under {home}")
    devices = []
    tool = first(str(Path(home) / "bin" / "amdgpu-arch"), llvm_tool("amdgpu-arch"))
    if tool and Path(tool).exists():
        devices = sorted(set((sh(tool) or "").split()))
    version = Path(home) / ".info" / "version"
    return dict(
        found=True,
        home=str(Path(home).resolve()),
        version=version.read_text().strip() if version.is_file() else "",
        devices=devices,
        arch=first(args.gpu_targets, devices and devices[0]) or "",
    )


def discover(args):
    mpi = find_mpi(args)
    found = dict(
        mpi=mpi, hdf5=find_hdf5(args, mpi), cuda=find_cuda(args), rocm=find_rocm(args)
    )
    gpu = "none" if args.gpu == "cpu" else args.gpu
    if gpu == "auto":
        gpu = next(
            (k for k in ("cuda", "rocm") if found[k]["found"] and found[k]["devices"]),
            "none",
        )
    found["gpu"] = gpu
    found["compiler"] = choose_compiler(args, gpu, found["cuda"])
    return found


def choose_compiler(args, gpu, cuda):
    """the compiler family for the target (--compiler, else gcc for CPU, clang for CUDA, hipcc
    for ROCm) and its binary. No nvcc fallback without a clang++: damn's device code builds with
    clang only (damn#68); --compiler nvcc to try anyway"""
    family = args.compiler or dict(none="gcc", cuda="clang", rocm="hipcc")[gpu]
    if gpu not in COMPILERS[family]:
        target = "CPU" if gpu == "none" else gpu
        return dict(
            found=False, family=family, why=f"{family} does not build for {target}"
        )
    if family == "nvcc":
        nvcc = Path(cuda.get("home", ""), "bin", "nvcc")
        if not nvcc.exists():
            return dict(found=False, family=family, why=f"no {nvcc}")
        host = args.cxx or "g++"  # nvcc's host compiler, under the MPI wrapper (-ccbin)
        return dict(found=True, family=family, cxx=host, driver=str(nvcc))
    cxx = args.cxx or dict(gcc="g++", clang="clang++", hipcc="hipcc")[family]
    if not shutil.which(cxx):
        why = f"no {cxx} on PATH"
        if gpu == "cuda" and family == "clang":
            why += " (damn's CUDA code needs clang: nvcc rejects its device lambdas, damn#68;"
            why += " --compiler nvcc to try anyway)"
        return dict(found=False, family=family, why=why)
    return dict(found=True, family=family, cxx=cxx, driver="")


# the toolchain's compile time flags (the header's Toolchain struct): each always there, false
#  when its area was not found
FLAGS = dict(
    mpi=["found", "cuda_aware", "rocm_aware"],
    hdf5=["found", "parallel", "subfiling", "matches_mpi"],
    cuda=["found"],
    rocm=["found"],
)


def no_facts():
    """the facts of no toolchain: every flag false, no GPU (the defaults header)"""
    return dict(
        gpu="none", **{f"{a}_{k}": False for a, ks in FLAGS.items() for k in ks}
    )


def facts(found):
    """what was found, flat and typed ("<area>_<key>": bool / int / str), for the header"""
    out = no_facts()
    out["gpu"] = found["gpu"]
    for area in ("mpi", "hdf5", "cuda", "rocm", "compiler"):
        for k, v in found[area].items():
            if k in ("inc", "lib") and isinstance(v, list):
                v = ":".join(v)
            if k == "devices":
                out[f"{area}_device_count"] = len(v)
                continue
            if isinstance(v, (bool, int, str)):
                out[f"{area}_{k}"] = v
    return out


def report(found):
    lines = []
    m, h = found["mpi"], found["hdf5"]
    lines.append(
        f"mpi   {m['impl']} {m['version']} ({m['cxx']}), cuda aware {m['cuda_aware']}, "
        f"rocm aware {m['rocm_aware']}"
        if m["found"]
        else f"mpi   none: {m['why']}"
    )
    lines.append(
        f"hdf5  {h['version']} ({h['inc']}), parallel {h['parallel']}, subfiling "
        f"{h['subfiling']}"
        + ("" if h["matches_mpi"] else f", NOT built with this MPI ({h['built_with']})")
        if h["found"]
        else f"hdf5  none: {h['why']}"
    )
    for k in ("cuda", "rocm"):
        g = found[k]
        lines.append(
            f"{k:<5} {g['version']} ({g['home']}), devices {' '.join(g['devices']) or 'none'}"
            if g["found"]
            else f"{k:<5} none: {g['why']}"
        )
    lines.append(f"gpu   {found['gpu']}")
    cc = found["compiler"]
    lines.append(
        f"cxx   {cc['family']} ({cc['cxx']}{', ' + cc['driver'] if cc['driver'] else ''})"
        if cc["found"]
        else f"cxx   {cc['family']}: {cc['why']}"
    )
    if found["gpu"] == "cuda" and m["found"] and not m["cuda_aware"]:
        lines.append(
            "      (this MPI is not CUDA aware: device buffers are staged on the host)"
        )
    return "\n".join(lines)


def settings(found, args, command, incs=()):
    """an mkn settings file for the found toolchain: the MPI wrapper over the GPU's compiler,
    HDF5's find paths, run paths (-Wl,-rpath) for libraries outside the system's: binaries and
    python modules run without LD_LIBRARY_PATH; incs: include dirs every build gets (the
    generated header's)"""
    m, h, gpu = found["mpi"], found["hdf5"], found["gpu"]
    if not m["found"]:
        raise SystemExit(f"bootstrap: {m['why']} (or give -x)")
    if gpu in ("cuda", "rocm") and not found[gpu]["found"]:
        raise SystemExit(f"bootstrap: --gpu {gpu}: {found[gpu]['why']}")
    cc = found["compiler"]
    if not cc["found"]:
        raise SystemExit(f"bootstrap: --compiler {cc['family']}: {cc['why']}")
    cxx = cc["cxx"]
    driver = m["cxx"]  # what mkn runs: the MPI wrapper, or nvcc over it (-ccbin)
    system = lambda d: str(d).startswith(("/usr/", "/lib"))  # noqa: E731
    runpath = [d for d in [*m["lib"], h.get("lib", "")] if d and not system(d)]
    rpath = " ".join(f"-Wl,-rpath,{d}" for d in runpath)
    common = "-std=c++20 -fPIC -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX"
    env = [f"PATH={m['bin']}:${{PATH}}"]  # its mpirun; libraries through the run paths
    inc = path = ""
    if gpu == "cuda" and cc["family"] == "nvcc":
        c = found["cuda"]
        if not c["arch"]:
            raise SystemExit("bootstrap: --gpu cuda: no device found, give --cuda-arch")
        driver = f"{cc['driver']} -ccbin {m['cxx']}"
        flags = (
            f"-std=c++20 -x cu -arch={c['arch']} --extended-lambda --expt-relaxed-constexpr"
            " -Xcompiler -fPIC -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -DMKN_GPU_CUDA=1"
            " -DMKN_GPU_ROCM=0 -DDAMN_HAVE_MKN_GPU=1 -DDAMN_HAVE_GPU_HARDWARE=1"
        )
        link = "-lcudart " + " ".join(f"-Xlinker -rpath={d}" for d in runpath)
    elif gpu == "cuda":
        c = found["cuda"]
        if not c["arch"]:
            raise SystemExit("bootstrap: --gpu cuda: no device found, give --cuda-arch")
        flags = (
            f"{common} -fsized-deallocation -Wno-unknown-cuda-version"
            f" -Xclang -fcuda-allow-variadic-functions -x cuda --cuda-path={c['home']}"
            f" --cuda-gpu-arch={c['arch']} -DMKN_GPU_CUDA=1 -DMKN_GPU_ROCM=0"
            " -DDAMN_HAVE_MKN_GPU=1 -DDAMN_HAVE_GPU_HARDWARE=1"
        )
        link = f"-L{c['lib']} -lcudart {rpath}"
        inc, path = c["cccl"], c["lib"]
    elif gpu == "rocm":
        r = found["rocm"]
        arch = f" --offload-arch={r['arch']}" if r["arch"] else ""
        flags = (
            f"{common} -fsized-deallocation{arch} -DMKN_GPU_ROCM=1 -DMKN_GPU_CUDA=0"
            " -DDAMN_HAVE_MKN_GPU=1 -DDAMN_HAVE_GPU_HARDWARE=1 -DDAMN_HAVE_THRUST=0"
        )
        link = f"-lamdhip64 {rpath}"
        env = [f"PATH={m['bin']}:{r['home']}/bin:${{PATH}}", *env[1:]]
    else:
        flags = f"{common} ${{cpu.thrust}}"
        link = rpath
    env += [f"OMPI_CXX={cxx}", f"MPICH_CXX={cxx}"]
    out = [
        f"# generated by {command}",
        "# this machine's toolchain (see etc/bootstrap/discover.py): not committed, rerun to update",
        "#",
        *[f"#   {line}" for line in report(found).splitlines()],
        "",
        "super: settings # ~/.maiken/settings.yaml: the repositories",
        "",
        "property:",
        f"  sys.arch: {ARCH}",
        # host thrust (damn's tpp/cccl), as res/mkn/base.yaml: CPU builds only
        "  cpu.thrust: >-",
        "    -Itpp/cccl/thrust -Itpp/cccl/libcudacxx/include -Itpp/cccl/cub",
        "    -Itpp/damn/tpp/cccl/thrust -Itpp/damn/tpp/cccl/libcudacxx/include"
        " -Itpp/damn/tpp/cccl/cub",
        f"  cxx.flags: {flags}",
        *(
            [f"  lnk.flags: {link.strip()}"] if link.strip() else []
        ),  # empty: "null" to mkn
        "",
        "env: |",
        *[f"  {e}" for e in env],
        "",
    ]
    incs = [str(i) for i in incs] + ([inc] if inc else [])
    if incs:
        out += ["inc: " + incs[0], *[f"  {i}" for i in incs[1:]]]
    if path:
        out += [f"path: {path}"]
    out += [""]
    if h.get("found"):
        out += ["find:", "  hdf5:", f"    inc: {h['inc']}", f"    lib: {h['lib']}", ""]
    if (
        driver == m["cxx"]
    ):  # the wrapper takes gcc style flags (nvcc as mkn.gpu's: no mask)
        out += ["compiler:", "  mask:", "    g++:", f"      {m['cxx']}", ""]
    out += [
        "file:",
        "  - type: cpp:cxx:cc",
        "    archiver: ar -cr",
        f"    compiler: {driver} ${{cxx.flags}}",
        f"    linker: {driver}" + (" ${lnk.flags}" if link.strip() else ""),
        "",
    ]
    return "\n".join(out)


def run_env(found):
    """the probes' runs: the found MPI's libraries first (as the settings' env for mkn's own)"""
    m, h = found["mpi"], found["hdf5"]
    env = dict(os.environ)
    if m.get("found"):
        env["PATH"] = f"{m['bin']}:{env.get('PATH', '')}"
        libs = [*m["lib"], h.get("lib", "")]
        libs += [env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = ":".join(
            d for d in libs if d
        )  # no empty entry: not "."
    return env
