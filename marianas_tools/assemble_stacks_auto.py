#!/usr/bin/env python3
"""
Assemble microscope TIFF exports into a single OME-TIFF per dataset group.

Files are named like:
  <BASE>_Z00_T01_C0.tif
and BASE is everything before _Z.._T.._C...

Key complication: each .tif can itself be a Z-stack (multi-page TIFF), and filename Z/T/C
may be constant (e.g., Z00 always), meaning that axis might live inside the TIFF.

Heuristic:
- If an axis varies across filenames -> that axis is "across files".
- If an axis does NOT vary across filenames -> it is either size 1 or lives "inside file".
- We probe one TIFF to infer internal stack shape and map it to (Z,Y,X) or (Z,C,Y,X) etc.

Output array order: (T, Z, Y, X, C)  [axes="TZYXC"]
Default is dry-run; use --save to write.

Requires:
  pip install tifffile numpy
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile as tiff

__all__ = ["assemble_marianas_stack"]

TIFF_RE = re.compile(
    r"^(?P<base>.+)_Z(?P<z>\d+)_T(?P<t>\d+)_C(?P<c>\d+)\.(?P<ext>tif|tiff)$",
    re.IGNORECASE,
)

BAD_SUFFIXES = {".log", ".xml"}  # we ignore these


@dataclass(frozen=True)
class FrameKey:
    t: int
    z: int
    c: int


@dataclass
class GroupInfo:
    base: str
    frames: Dict[FrameKey, Path]
    z_vals: List[int]
    t_vals: List[int]
    c_vals: List[int]


@dataclass
class InternalShape:
    raw_shape: Tuple[int, ...]
    # interpreted as (t_in, z_in, c_in, y, x) with some = 1 if absent
    t_in: int
    z_in: int
    c_in: int
    y: int
    x: int
    dtype: np.dtype


@dataclass
class Plan:
    base: str
    across_t: bool
    across_z: bool
    across_c: bool
    nt: int
    nz: int
    nc: int
    yx: Tuple[int, int]
    dtype: np.dtype
    n_files: int
    missing_keys: List[FrameKey]
    out_path: Path
    note: str


def parse_one(path: Path) -> Optional[Tuple[str, FrameKey]]:
    m = TIFF_RE.match(path.name)
    if not m:
        return None
    base = m.group("base")
    return base, FrameKey(t=int(m.group("t")), z=int(m.group("z")), c=int(m.group("c")))


def collect_groups(input_dir: Path, recursive: bool = False) -> Dict[str, GroupInfo]:
    it = input_dir.rglob("*") if recursive else input_dir.glob("*")
    groups: Dict[str, Dict[FrameKey, Path]] = {}

    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() in BAD_SUFFIXES:
            continue
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue

        parsed = parse_one(p)
        if parsed is None:
            continue
        base, key = parsed
        groups.setdefault(base, {})
        if key in groups[base]:
            raise RuntimeError(f"Duplicate frame for {base} at {key}: {p}")
        groups[base][key] = p

    out: Dict[str, GroupInfo] = {}
    for base, frames in groups.items():
        z_vals = sorted({k.z for k in frames})
        t_vals = sorted({k.t for k in frames})
        c_vals = sorted({k.c for k in frames})
        out[base] = GroupInfo(base=base, frames=frames, z_vals=z_vals, t_vals=t_vals, c_vals=c_vals)
    return out


def infer_internal_shape(
    path: Path,
    file_nz: Optional[int],
    file_nt: Optional[int],
    file_nc: Optional[int],
) -> InternalShape:
    """
    Read one TIFF and interpret its array shape.

    Common cases:
      2D: (Y, X)
      3D: (Z, Y, X)  or (T, Y, X) (ambiguous; we assume Z by default)
      4D: (Z, C, Y, X) or (T, Z, Y, X) etc.
      5D: (T, Z, C, Y, X) etc.

    We let user disambiguate with --file-nz/--file-nt/--file-nc:
    - If provided, we reshape the leading dimension(s) accordingly.
    """
    arr = tiff.imread(str(path))
    raw_shape = tuple(arr.shape)
    dtype = arr.dtype

    if arr.ndim < 2:
        raise ValueError(f"Unexpected TIFF ndim={arr.ndim} for {path}")

    y, x = arr.shape[-2], arr.shape[-1]
    leading = arr.shape[:-2]  # could be empty, or (Z,), or (Z,C), etc.

    # If user gives explicit internal dims, honor them strictly.
    if file_nz is not None or file_nt is not None or file_nc is not None:
        # Determine total leading elements
        lead_n = int(np.prod(leading)) if leading else 1

        t_in = file_nt if file_nt is not None else 1
        z_in = file_nz if file_nz is not None else 1
        c_in = file_nc if file_nc is not None else 1

        if t_in * z_in * c_in != lead_n:
            raise ValueError(
                f"Internal-dim override mismatch for {path.name}: "
                f"leading product={lead_n} but file_nt*file_nz*file_nc={t_in*z_in*c_in} "
                f"(t={t_in}, z={z_in}, c={c_in}). "
                f"Raw shape={raw_shape}"
            )

        return InternalShape(
            raw_shape=raw_shape,
            t_in=t_in,
            z_in=z_in,
            c_in=c_in,
            y=y,
            x=x,
            dtype=np.dtype(dtype),
        )

    # No explicit override: heuristic mapping
    if len(leading) == 0:
        return InternalShape(raw_shape=raw_shape, t_in=1, z_in=1, c_in=1, y=y, x=x, dtype=np.dtype(dtype))

    if len(leading) == 1:
        # Most common: Z-stack stored as (Z, Y, X)
        return InternalShape(raw_shape=raw_shape, t_in=1, z_in=leading[0], c_in=1, y=y, x=x, dtype=np.dtype(dtype))

    if len(leading) == 2:
        # Ambiguous: (Z,C,Y,X) vs (T,Z,Y,X) etc.
        # We'll assume (Z,C,Y,X) if second dim is small-ish (<=4 typically),
        # otherwise treat as (T,Z,Y,X) with C=1.
        a, b = leading
        if b <= 4:
            return InternalShape(raw_shape=raw_shape, t_in=1, z_in=a, c_in=b, y=y, x=x, dtype=np.dtype(dtype))
        else:
            return InternalShape(raw_shape=raw_shape, t_in=a, z_in=b, c_in=1, y=y, x=x, dtype=np.dtype(dtype))

    if len(leading) == 3:
        # Assume (T, Z, C, Y, X) if last of leading is small-ish
        a, b, c = leading
        if c <= 4:
            return InternalShape(raw_shape=raw_shape, t_in=a, z_in=b, c_in=c, y=y, x=x, dtype=np.dtype(dtype))
        else:
            # otherwise (T, Z, ?, Y, X) with C=1 and fold last into Z
            return InternalShape(raw_shape=raw_shape, t_in=a, z_in=b * c, c_in=1, y=y, x=x, dtype=np.dtype(dtype))

    # len(leading) >= 4: flatten all but YX into (t,z,c) as best we can
    lead_prod = int(np.prod(leading))
    # assume it's pure Z if huge
    return InternalShape(raw_shape=raw_shape, t_in=1, z_in=lead_prod, c_in=1, y=y, x=x, dtype=np.dtype(dtype))


def plan_group(
    g: GroupInfo,
    out_dir: Path,
    nz_override: Optional[int],
    nt_override: Optional[int],
    nc_override: Optional[int],
    file_nz: Optional[int],
    file_nt: Optional[int],
    file_nc: Optional[int],
    out_ext: str,
) -> Tuple[Plan, InternalShape]:
    # Determine which axes vary across filenames
    across_z = len(g.z_vals) > 1
    across_t = len(g.t_vals) > 1
    across_c = len(g.c_vals) > 1

    # Probe internal TIFF shape from a representative file (first)
    rep_path = next(iter(g.frames.values()))
    internal = infer_internal_shape(rep_path, file_nz=file_nz, file_nt=file_nt, file_nc=file_nc)

    # Decide final dims from combination of "across files" + "inside file"
    # For each axis:
    #   if across axis -> size = count across filenames (unless override)
    #   else -> size = internal axis size (unless override), else 1
    nt = nt_override if nt_override is not None else (len(g.t_vals) if across_t else internal.t_in)
    nz = nz_override if nz_override is not None else (len(g.z_vals) if across_z else internal.z_in)
    nc = nc_override if nc_override is not None else (len(g.c_vals) if across_c else internal.c_in)

    # Note to help you verify the inference
    note = (
        f"Across-files axes: T={'yes' if across_t else 'no'} "
        f"Z={'yes' if across_z else 'no'} "
        f"C={'yes' if across_c else 'no'}; "
        f"internal inferred (t,z,c)=({internal.t_in},{internal.z_in},{internal.c_in}) from {rep_path.name}"
    )

    # Compute expected keys if axis is across-files, otherwise we treat filename index as 0
    # We normalize across-file indices to 0..N-1 in sorted order (important if indices don't start at 0).
    t_map = {v: i for i, v in enumerate(g.t_vals)} if across_t else {g.t_vals[0]: 0}
    z_map = {v: i for i, v in enumerate(g.z_vals)} if across_z else {g.z_vals[0]: 0}
    c_map = {v: i for i, v in enumerate(g.c_vals)} if across_c else {g.c_vals[0]: 0}

    # Missing keys (across-files only). Internal missing is not checked here.
    missing: List[FrameKey] = []
    expected = []
    for t in range(len(g.t_vals) if across_t else 1):
        for z in range(len(g.z_vals) if across_z else 1):
            for c in range(len(g.c_vals) if across_c else 1):
                expected.append((t, z, c))

    # Translate existing keys into normalized (t,z,c)
    present_norm = set()
    for k in g.frames:
        present_norm.add((t_map[k.t], z_map[k.z], c_map[k.c]))

    for t, z, c in expected:
        if (t, z, c) not in present_norm:
            # store "normalized" as FrameKey for reporting
            missing.append(FrameKey(t=t, z=z, c=c))

    safe = re.sub(r'[<>:"/\\|?*]+', "_", g.base).strip()
    out_path = out_dir / f"{safe}.{out_ext.lstrip('.')}"
    return (
        Plan(
            base=g.base,
            across_t=across_t,
            across_z=across_z,
            across_c=across_c,
            nt=nt,
            nz=nz,
            nc=nc,
            yx=(internal.y, internal.x),
            dtype=internal.dtype,
            n_files=len(g.frames),
            missing_keys=missing,
            out_path=out_path,
            note=note,
        ),
        internal,
    )


def read_file_as_tzc_yx(arr: np.ndarray, internal: InternalShape) -> np.ndarray:
    """
    Convert raw TIFF array into standardized (t_in, z_in, c_in, y, x).
    Uses internal.{t_in,z_in,c_in} and expects arr.shape[-2:]==(y,x).
    """
    y, x = internal.y, internal.x
    if arr.shape[-2:] != (y, x):
        raise ValueError(f"Unexpected YX shape {arr.shape[-2:]} (expected {(y,x)})")

    lead = int(np.prod(arr.shape[:-2])) if arr.ndim > 2 else 1
    t_in, z_in, c_in = internal.t_in, internal.z_in, internal.c_in
    if t_in * z_in * c_in != lead:
        raise ValueError(
            f"Internal reshape mismatch: lead_prod={lead} but t*z*c={t_in*z_in*c_in}. "
            f"Raw shape={arr.shape}"
        )

    flat = arr.reshape((t_in, z_in, c_in, y, x))
    return flat


def assemble_group(
    g: GroupInfo,
    plan: Plan,
    internal: InternalShape,
    allow_missing_files: bool,
    fill_value: Optional[int],
) -> np.ndarray:
    """
    Build final array (T, Z, Y, X, C) with axes "TZYXC".

    If an axis is across-files: we index by filename key and place internal data into that slot.
    If an axis is NOT across-files: we expect that axis to be inside-file, and we copy the whole internal axis.
    """
    fv = 0 if fill_value is None else fill_value
    nt, nz, nc = plan.nt, plan.nz, plan.nc
    y, x = plan.yx

    # Across-files normalized index maps
    t_vals = g.t_vals
    z_vals = g.z_vals
    c_vals = g.c_vals
    t_map = {v: i for i, v in enumerate(t_vals)} if plan.across_t else {t_vals[0]: 0}
    z_map = {v: i for i, v in enumerate(z_vals)} if plan.across_z else {z_vals[0]: 0}
    c_map = {v: i for i, v in enumerate(c_vals)} if plan.across_c else {c_vals[0]: 0}

    if plan.missing_keys and not allow_missing_files:
        first = plan.missing_keys[:10]
        more = "" if len(plan.missing_keys) <= 10 else f" (+{len(plan.missing_keys)-10} more)"
        raise RuntimeError(f"Missing {len(plan.missing_keys)} file(s) for '{plan.base}'{more}. First: {first}")

    out = np.full((nt, nz, y, x, nc), fv, dtype=plan.dtype)

    for key, path in g.frames.items():
        t_idx = t_map[key.t] if plan.across_t else 0
        z_idx = z_map[key.z] if plan.across_z else 0
        c_idx = c_map[key.c] if plan.across_c else 0

        # Load raw and normalize to (t_in, z_in, c_in, y, x)
        raw = tiff.imread(str(path))
        tzc_yx = read_file_as_tzc_yx(raw, internal)  # (t_in, z_in, c_in, y, x)

        # Now place into output depending on whether axes are across-files or inside-file.
        # For each axis:
        #  - across-files: output index fixed (t_idx/z_idx/c_idx), internal axis must be size 1
        #  - inside-file: output spans that axis, output index is a slice, internal axis provides values
        #
        # We allow internal size > 1 even if across-files, only if output size matches and across-files
        # dimension is 1 (common if filenames had only one index but internal has the axis).
        #
        # Concretely:
        #  - If across_t: we expect internal.t_in == 1, and place at out[t_idx, ...]
        #    Else: we copy internal t into out[0:internal.t_in, ...]
        #
        # Same for z and c.

        # Determine target slices
        t_sl = slice(t_idx, t_idx + 1) if plan.across_t else slice(0, min(nt, internal.t_in))
        z_sl = slice(z_idx, z_idx + 1) if plan.across_z else slice(0, min(nz, internal.z_in))
        c_sl = slice(c_idx, c_idx + 1) if plan.across_c else slice(0, min(nc, internal.c_in))

        # Validate sizes when across-files
        if plan.across_t and internal.t_in != 1:
            raise RuntimeError(
                f"For group '{plan.base}', T varies across filenames but file '{path.name}' "
                f"contains internal T={internal.t_in}. Use --file-nt/--file-nz/--file-nc to disambiguate, "
                f"or reconsider axis interpretation."
            )
        if plan.across_z and internal.z_in != 1:
            raise RuntimeError(
                f"For group '{plan.base}', Z varies across filenames but file '{path.name}' "
                f"contains internal Z={internal.z_in}. Likely you should treat Z as inside-file."
            )
        if plan.across_c and internal.c_in != 1:
            raise RuntimeError(
                f"For group '{plan.base}', C varies across filenames but file '{path.name}' "
                f"contains internal C={internal.c_in}. Likely you should treat C as inside-file."
            )

        # Extract the internal block we want to paste
        block = tzc_yx[
            0 if plan.across_t else t_sl,
            0 if plan.across_z else z_sl,
            0 if plan.across_c else c_sl,
            :,
            :,
        ]

        # Make block always 5D: (t, z, c, y, x)
        if plan.across_t:
            block = block[np.newaxis, ...]
        if plan.across_z:
            block = block[:, np.newaxis, ...]
        if plan.across_c:
            block = block[:, :, np.newaxis, ...]

        # Now block is (t_out, z_out, c_out, y, x). We need (t,z,y,x,c).
        block_tzyxc = np.moveaxis(block, 2, -1)  # (t, z, y, x, c)

        # Paste
        out[t_sl, z_sl, :, :, c_sl] = block_tzyxc

    return out


def write_ome_tiff(path: Path, data_tzyxc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bigtiff = data_tzyxc.nbytes >= (4 * 1024**3)
    tiff.imwrite(
        str(path),
        data_tzyxc,
        ome=True,
        metadata={"axes": "TZYXC"},
        bigtiff=bigtiff,
    )


def assemble_marianas_stack(
    input_dir: Path | str,
    *,
    out_dir: Path | str | None = None,
    recursive: bool = False,
    save: bool = True,
    allow_missing_files: bool = False,
    fill_value: int | None = None,
    # Override FINAL assembled dims
    nt: int | None = None,
    nz: int | None = None,
    nc: int | None = None,
    # Override INTERNAL per-file dims interpretation
    file_nt: int | None = None,
    file_nz: int | None = None,
    file_nc: int | None = None,
    out_ext: str = "ome.tif",
    # Optional selection
    bases: list[str] | None = None,
    return_data: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Function for assembling microscope TIFF exports into OME-TIFF stacks.

    Parameters
    ----------
    input_dir:
        Folder containing TIFFs named like <BASE>_Z00_T01_C0.tif
    out_dir:
        Output folder (default: input_dir / "Stacks")
    recursive:
        If True, search input_dir recursively.
    save:
        If True, write OME-TIFF(s). If False, dry-run planning only.
    allow_missing_files:
        If True, fill missing file-slots instead of raising.
    fill_value:
        Fill value for missing slots (default 0 if allow_missing_files=True).
    nt/nz/nc:
        Override FINAL assembled output dims.
    file_nt/file_nz/file_nc:
        Override INTERNAL per-file interpretation (disambiguates multi-page TIFF axes).
    out_ext:
        Output extension, e.g. "ome.tif".
    bases:
        If provided, only assemble groups whose base name is in this list.
    return_data:
        If True and save=True, also return the assembled numpy array(s) in-memory.
        (Beware: can be huge.)
    verbose:
        Print plan summaries like the CLI.

    Returns
    -------
    dict with keys:
        - "input_dir", "out_dir", "n_groups"
        - "plans": list[Plan]
        - "written": list[Path] (if save=True)
        - "data": dict[str, np.ndarray] (if return_data=True and save=True)
        - "notes": list[str]
    """
    inp = Path(input_dir)
    if not inp.exists():
        raise FileNotFoundError(f"Not found: {inp}")

    outp = Path(out_dir) if out_dir is not None else (inp / "Stacks")

    groups = collect_groups(inp, recursive=recursive)
    if bases is not None:
        bases_set = set(bases)
        groups = {b: g for b, g in groups.items() if b in bases_set}

    if not groups:
        if verbose:
            print(f"No matching TIFFs found in {inp}")
        return {
            "input_dir": inp,
            "out_dir": outp,
            "n_groups": 0,
            "plans": [],
            "written": [],
            "data": {} if return_data else None,
            "notes": [],
        }

    written: list[Path] = []
    data_out: dict[str, np.ndarray] = {}
    plans: list[Plan] = []
    notes: list[str] = []

    if verbose:
        mode = "SAVE" if save else "DRY-RUN"
        print(f"Found {len(groups)} group(s). Mode: {mode}")
        print(f"Output dir: {outp}")
        if any(v is not None for v in (nt, nz, nc)):
            print(f"Final overrides: nt={nt} nz={nz} nc={nc}")
        if any(v is not None for v in (file_nt, file_nz, file_nc)):
            print(f"Internal overrides: file_nt={file_nt} file_nz={file_nz} file_nc={file_nc}")
        print("")

    for i, (base, g) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower()), start=1):
        plan, internal = plan_group(
            g=g,
            out_dir=outp,
            nz_override=nz,
            nt_override=nt,
            nc_override=nc,
            file_nz=file_nz,
            file_nt=file_nt,
            file_nc=file_nc,
            out_ext=out_ext,
        )
        plans.append(plan)
        notes.append(plan.note)

        if verbose:
            print(f"[{i}/{len(groups)}] {plan.base}")
            print(
                f"  Files: {plan.n_files}  (T vals={g.t_vals[:5]}{'...' if len(g.t_vals)>5 else ''}, "
                f"Z vals={g.z_vals[:5]}{'...' if len(g.z_vals)>5 else ''}, "
                f"C vals={g.c_vals})"
            )
            print(f"  {plan.note}")
            print(
                f"  Output dims: (T,Z,Y,X,C)=({plan.nt},{plan.nz},{plan.yx[0]},{plan.yx[1]},{plan.nc}) "
                f"dtype={plan.dtype}"
            )
            print(f"  Output: {plan.out_path}")

            if plan.missing_keys:
                print(f"  Missing file-slots: {len(plan.missing_keys)}")
                for k in plan.missing_keys[:10]:
                    print(f"    missing {k}")
                if len(plan.missing_keys) > 10:
                    print(f"    ... (+{len(plan.missing_keys)-10} more)")
                if not allow_missing_files:
                    print("  NOTE: will ERROR on missing unless allow_missing_files=True")

        if save:
            arr = assemble_group(
                g=g,
                plan=plan,
                internal=internal,
                allow_missing_files=allow_missing_files,
                fill_value=fill_value,
            )
            write_ome_tiff(plan.out_path, arr)
            written.append(plan.out_path)
            if return_data:
                data_out[plan.base] = arr
            if verbose:
                print("  Wrote OME-TIFF.")
        else:
            if verbose:
                print("  Dry-run only (set save=True to write).")

        if verbose:
            print("")

    return {
        "input_dir": inp,
        "out_dir": outp,
        "n_groups": len(groups),
        "plans": plans,
        "written": written,
        "data": data_out if return_data else None,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--save", action="store_true", help="Write OME-TIFFs (default dry-run)")
    ap.add_argument("--allow-missing-files", action="store_true", help="Fill missing file-slots instead of erroring")
    ap.add_argument("--fill", type=int, default=None, help="Fill value for missing slots (default 0)")

    # Override FINAL assembled dims
    ap.add_argument("--nt", type=int, default=None)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--nc", type=int, default=None)

    # Override INTERNAL per-file dims interpretation
    ap.add_argument("--file-nt", type=int, default=None, help="Force internal T inside each TIFF")
    ap.add_argument("--file-nz", type=int, default=None, help="Force internal Z inside each TIFF")
    ap.add_argument("--file-nc", type=int, default=None, help="Force internal C inside each TIFF")

    ap.add_argument("--ext", type=str, default="ome.tif")
    args = ap.parse_args()

    inp = args.input_dir
    if not inp.exists():
        raise SystemExit(f"Not found: {inp}")

    out_dir = args.out_dir if args.out_dir is not None else (inp / "Stacks")
    groups = collect_groups(inp, recursive=args.recursive)

    if not groups:
        print(f"No matching TIFFs found in {inp}")
        return 0

    print(f"Found {len(groups)} group(s). Mode: {'SAVE' if args.save else 'DRY-RUN'}")
    print(f"Output dir: {out_dir}")
    if any(v is not None for v in (args.nt, args.nz, args.nc)):
        print(f"Final overrides: nt={args.nt} nz={args.nz} nc={args.nc}")
    if any(v is not None for v in (args.file_nt, args.file_nz, args.file_nc)):
        print(f"Internal overrides: file_nt={args.file_nt} file_nz={args.file_nz} file_nc={args.file_nc}")
    print("")

    for i, (base, g) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower()), start=1):
        plan, internal = plan_group(
            g=g,
            out_dir=out_dir,
            nz_override=args.nz,
            nt_override=args.nt,
            nc_override=args.nc,
            file_nz=args.file_nz,
            file_nt=args.file_nt,
            file_nc=args.file_nc,
            out_ext=args.ext,
        )

        print(f"[{i}/{len(groups)}] {plan.base}")
        print(f"  Files: {plan.n_files}  (T vals={g.t_vals[:5]}{'...' if len(g.t_vals)>5 else ''}, "
              f"Z vals={g.z_vals[:5]}{'...' if len(g.z_vals)>5 else ''}, "
              f"C vals={g.c_vals})")
        print(f"  {plan.note}")
        print(f"  Output dims: (T,Z,Y,X,C)=({plan.nt},{plan.nz},{plan.yx[0]},{plan.yx[1]},{plan.nc}) dtype={plan.dtype}")
        print(f"  Output: {plan.out_path}")

        if plan.missing_keys:
            print(f"  Missing file-slots: {len(plan.missing_keys)}")
            for k in plan.missing_keys[:10]:
                print(f"    missing {k}")
            if len(plan.missing_keys) > 10:
                print(f"    ... (+{len(plan.missing_keys)-10} more)")
            if not args.allow_missing_files:
                print("  NOTE: will ERROR on missing unless --allow-missing-files is set")

        if args.save:
            data = assemble_group(
                g=g,
                plan=plan,
                internal=internal,
                allow_missing_files=args.allow_missing_files,
                fill_value=args.fill,
            )
            write_ome_tiff(plan.out_path, data)
            print("  Wrote OME-TIFF.")

        else:
            print("  Dry-run only (use --save to write).")

        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())