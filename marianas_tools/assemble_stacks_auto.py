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
import xml.etree.ElementTree as ET
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
    # Core identity
    base: str
    anchor_path: Path

    # What the assembler inferred about whether axes are spread across filenames
    # (kept for backward-compatibility with earlier planning output)
    across_t: bool
    across_z: bool
    across_c: bool

    # Final output dimensions (always reported as T,Z,Y,X,C)
    nt: int
    nz: int
    nc: int
    yx: Tuple[int, int]
    dtype: np.dtype

    # Planning / diagnostics
    n_files: int
    missing_keys: List[FrameKey]          # legacy: missing (T,Z,C) slots in filename grid
    missing_files: List[str]              # missing files referenced by OME-XML (filenames)
    axes: str | None                      # axes string as reported by tifffile (e.g., "TCZYX", "TZYX")
    referenced_files: List[str]           # files referenced by OME-XML (filenames)
    physical_sizes_um: dict[str, float] | None   # {"X": um, "Y": um, "Z": um}
    time_increment_s: float | None

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
            anchor_path=rep_path,
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
            missing_files=[],
            axes=None,
            referenced_files=[],
            physical_sizes_um=None,
            time_increment_s=None,
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


def write_ome_tiff(path, data_tzyxc, physical_sizes_um=None, time_increment_s=None):
    import numpy as np
    import tifffile as tiff
    path.parent.mkdir(parents=True, exist_ok=True)  # <-- add this line
    # Convert T Z Y X C  ->  T Z C Y X
    if data_tzyxc.ndim != 5:
        raise ValueError(f"Expected 5D TZYXC array, got shape={data_tzyxc.shape}")
    data_tzcyx = np.moveaxis(data_tzyxc, -1, 2)

    md = {"axes": "TZCYX"}

    def _coerce_physical_sizes_um(v):
        if v is None:
            return None
        if isinstance(v, dict):
            for keyset in (("X", "Y", "Z"), ("x", "y", "z"), ("sx", "sy", "sz")):
                if all(k in v for k in keyset):
                    try:
                        return tuple(float(v[k]) for k in keyset)
                    except Exception:
                        return None
            return None
        if isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                return None
        return None

    sizes = _coerce_physical_sizes_um(physical_sizes_um)
    if sizes is not None:
        sx, sy, sz = sizes
        md["PhysicalSizeX"] = sx
        md["PhysicalSizeXUnit"] = "µm"
        md["PhysicalSizeY"] = sy
        md["PhysicalSizeYUnit"] = "µm"
        md["PhysicalSizeZ"] = sz
        md["PhysicalSizeZUnit"] = "µm"

    if time_increment_s is not None:
        md["TimeIncrement"] = float(time_increment_s)
        md["TimeIncrementUnit"] = "s"

    tiff.imwrite(str(path), data_tzcyx, ome=True, metadata=md)


def _parse_ome_xml(ome_xml: str) -> dict:
    """Parse a minimal subset of OME-XML needed for planning.

    Returns dict with:
      - sizes: dict with keys SizeT/SizeZ/SizeC/SizeY/SizeX (ints if present)
      - physical: dict with keys PhysicalSizeX/Y/Z (floats if present) and units (strings)
      - time_increment: float seconds if present, else None
      - referenced_files: list[str] (filenames referenced by OME, may be empty)
    """
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return {"sizes": {}, "physical": {}, "time_increment_s": None, "referenced_files": []}

    # Namespace handling
    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"ome": ns_uri} if ns_uri else {}

    pixels = root.find(".//ome:Pixels", ns) if ns else root.find(".//Pixels")
    sizes: dict[str, int] = {}
    physical: dict[str, object] = {}
    time_increment_s: float | None = None

    if pixels is not None:
        for k in ("SizeT", "SizeZ", "SizeC", "SizeY", "SizeX"):
            v = pixels.attrib.get(k)
            if v is not None:
                try:
                    sizes[k] = int(v)
                except Exception:
                    pass

        # Physical sizes + units
        for ax in ("X", "Y", "Z"):
            v = pixels.attrib.get(f"PhysicalSize{ax}")
            u = pixels.attrib.get(f"PhysicalSize{ax}Unit")
            if v is not None:
                try:
                    physical[f"PhysicalSize{ax}"] = float(v)
                except Exception:
                    pass
            if u is not None:
                physical[f"PhysicalSize{ax}Unit"] = u

        ti = pixels.attrib.get("TimeIncrement")
        tiu = pixels.attrib.get("TimeIncrementUnit", "s")
        if ti is not None:
            try:
                ti_val = float(ti)
                # Convert to seconds if needed
                if tiu.lower() in ("s", "sec", "second", "seconds"):
                    time_increment_s = ti_val
                elif tiu.lower() in ("ms", "millisecond", "milliseconds"):
                    time_increment_s = ti_val / 1000.0
                elif tiu.lower() in ("min", "minute", "minutes"):
                    time_increment_s = ti_val * 60.0
                else:
                    # Unknown units -> assume seconds
                    time_increment_s = ti_val
            except Exception:
                pass

    # Referenced files (multi-file OME)
    referenced_files: list[str] = []
    # Common pattern: <TiffData><UUID FileName="..."/>
    if ns:
        for uuid in root.findall(".//ome:TiffData/ome:UUID", ns):
            fn = uuid.attrib.get("FileName")
            if fn:
                referenced_files.append(fn)
    else:
        for uuid in root.findall(".//TiffData/UUID"):
            fn = uuid.attrib.get("FileName")
            if fn:
                referenced_files.append(fn)

    # Deduplicate while preserving order
    seen = set()
    referenced_files = [f for f in referenced_files if not (f in seen or seen.add(f))]

    return {
        "sizes": sizes,
        "physical": physical,
        "time_increment_s": time_increment_s,
        "referenced_files": referenced_files,
    }


def _physical_um_from_ome(physical: dict) -> dict[str, float] | None:
    """Return physical sizes in µm if available; heuristically fix bad 'm' units."""
    if not physical:
        return None

    out: dict[str, float] = {}
    for ax in ("X", "Y", "Z"):
        v = physical.get(f"PhysicalSize{ax}")
        u = physical.get(f"PhysicalSize{ax}Unit")
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        u_norm = (u or "").strip()

        # Heuristic: some exports incorrectly mark unit as meters ("m") while values are in microns.
        # If unit is 'm' and value is "small-ish" (e.g., < 10), treat it as microns.
        if u_norm in ("m", "meter", "metre"):
            if v < 10:
                out[ax] = v  # interpret as µm
            else:
                out[ax] = v * 1e6  # meters -> µm
        elif u_norm in ("µm", "um", "micrometer", "micrometre", "micron", "microns"):
            out[ax] = v
        elif u_norm in ("nm", "nanometer", "nanometre"):
            out[ax] = v / 1000.0
        else:
            # Unknown unit: assume µm if value is plausible
            out[ax] = v
    return out or None


def _parse_log_metadata_ms(log_path: Path) -> dict:
    """Parse key-value header lines from a Marianas .log file."""
    info: dict[str, object] = {}
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return info

    # Simple header parsing (stop before the big table header)
    for line in txt.splitlines():
        if line.startswith("IFD"):
            break
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        info[k] = v

    # Extract average timelapse interval (ms)
    # Example: "Average Timelapse Interval: 60000.17 ms (0.0 Hz) (+/- 0.6 ms)"
    at = info.get("Average Timelapse Interval")
    if isinstance(at, str):
        m = re.search(r"([0-9]*\.?[0-9]+)\s*ms\b", at)
        if m:
            try:
                info["TimeIncrement_ms"] = float(m.group(1))
            except Exception:
                pass

    # Convenience numeric fields
    for k in ("Z Planes", "Time Points", "Channels"):
        if k in info and isinstance(info[k], str):
            try:
                info[k] = int(str(info[k]).strip())
            except Exception:
                pass
    for k in ("Microns Per Pixel", "Z Step Size Microns"):
        if k in info and isinstance(info[k], str):
            try:
                info[k] = float(str(info[k]).strip())
            except Exception:
                pass

    return info


def _pick_log_file_for_base(input_dir: Path, base: str) -> Path | None:
    """Pick a log file for a base, if present."""
    # Common: "<BASE>..._Z00_T00_C0.log" or "<BASE>.log" etc.
    # We'll pick the first matching by name sort.
    candidates = sorted(input_dir.glob(f"{base}*.log"))
    return candidates[0] if candidates else None


def _ensure_tzyxc(arr: np.ndarray, axes: str | None) -> np.ndarray:
    """Convert an array with known axes into TZYXC."""
    if axes is None:
        # Best-effort based on ndim
        if arr.ndim == 2:  # YX
            arr = arr[None, None, :, :, None]
        elif arr.ndim == 3:  # ZYX
            arr = arr[None, :, :, :, None]
        elif arr.ndim == 4:  # TZYX (assume)
            arr = arr[:, :, :, :, None]
        elif arr.ndim == 5:
            return arr
        else:
            raise ValueError(f"Unsupported array ndim={arr.ndim}")
        return arr

    ax = axes.upper()
    # Normalize common variants
    # We want order: T Z Y X C
    # Handle: TZYX, TCZYX, TZCYX, ZCYX, CZYX, etc.
    wanted = "TZYXC"
    # Build mapping from current axes to indices
    idx = {a: i for i, a in enumerate(ax)}
    # If no C, treat as C=1 at end
    has_c = "C" in idx
    has_t = "T" in idx
    has_z = "Z" in idx

    # Create view with missing dims added
    out = arr
    cur_axes = ax
    if not has_t:
        out = out[None, ...]
        cur_axes = "T" + cur_axes
        idx = {a: i for i, a in enumerate(cur_axes)}
    if not has_z:
        # if Z missing, treat as single plane
        out = out[:, None, ...] if cur_axes[0] == "T" else out[None, ...]
        # After insertion, easiest is to recompute with explicit insertion at correct place
        # We'll just fall back to best-effort reorder below by padding.
        cur_axes = cur_axes.replace("T", "TZ", 1) if cur_axes.startswith("T") else "Z" + cur_axes
        idx = {a: i for i, a in enumerate(cur_axes)}
    if "Y" not in idx or "X" not in idx:
        raise ValueError(f"Cannot coerce axes {axes} to TZYXC")

    if "C" not in idx:
        # Add channel dim at end
        out = out[..., None]
        cur_axes = cur_axes + "C"
        idx = {a: i for i, a in enumerate(cur_axes)}

    # Now reorder to T Z Y X C
    order = [idx["T"], idx["Z"], idx["Y"], idx["X"], idx["C"]]
    out = np.moveaxis(out, order, range(5))
    return out


def _ome_plan_for_anchor(
    anchor_path: Path,
    *,
    out_dir: Path,
    out_ext: str,
    log_files: bool,
) -> tuple[Plan, dict]:
    """Create a Plan using OME-XML + tifffile series metadata."""
    with tiff.TiffFile(str(anchor_path)) as tif:
        axes = getattr(tif.series[0], "axes", None)
        shape = tif.series[0].shape
        dtype = np.dtype(tif.series[0].dtype)

        ome = tif.ome_metadata
        parsed = _parse_ome_xml(ome) if ome else {"sizes": {}, "physical": {}, "time_increment_s": None, "referenced_files": []}
        sizes = parsed["sizes"]
        physical_um = _physical_um_from_ome(parsed["physical"])
        time_inc_s = parsed["time_increment_s"]
        referenced_files = parsed["referenced_files"]

    # Determine T, Z, C, Y, X from shape + axes
    # We'll coerce based on axes string if present.
    # If axes missing, fall back to common ordering used by tifffile: TZYX or TCZYX.
    if axes:
        ax = axes.upper()
        idx = {a: i for i, a in enumerate(ax)}
        nt = shape[idx["T"]] if "T" in idx else 1
        nz = shape[idx["Z"]] if "Z" in idx else 1
        nc = shape[idx["C"]] if "C" in idx else 1
        ny = shape[idx["Y"]] if "Y" in idx else shape[-2]
        nx = shape[idx["X"]] if "X" in idx else shape[-1]
    else:
        # shape could be (T,Z,Y,X) or (T,C,Z,Y,X)
        if len(shape) == 4:
            nt, nz, ny, nx = shape
            nc = 1
        elif len(shape) == 5:
            nt, nc, nz, ny, nx = shape
        else:
            raise ValueError(f"Unexpected series shape {shape} for {anchor_path.name}")

    # OME-referenced files: check existence relative to anchor folder
    missing_files: list[str] = []
    if referenced_files:
        for fn in referenced_files:
            if not (anchor_path.parent / fn).exists():
                missing_files.append(fn)

    # Supplement metadata from log file (dt + phys sizes) if requested
    if log_files:
        lp = _pick_log_file_for_base(anchor_path.parent, base=_base_from_filename(anchor_path.name) or anchor_path.stem)
        if lp is not None:
            log_info = _parse_log_metadata_ms(lp)
            if time_inc_s is None and "TimeIncrement_ms" in log_info:
                time_inc_s = float(log_info["TimeIncrement_ms"]) / 1000.0
            # Fill physical sizes if missing
            if physical_um is None:
                px = log_info.get("Microns Per Pixel")
                dz = log_info.get("Z Step Size Microns")
                if isinstance(px, (int, float)):
                    physical_um = {"X": float(px), "Y": float(px)}
                    if isinstance(dz, (int, float)):
                        physical_um["Z"] = float(dz)

    base = _base_from_filename(anchor_path.name) or anchor_path.stem
    out_path = (out_dir / f"{base}.{out_ext}") if out_ext else (out_dir / f"{base}.ome.tif")

    note_parts = []
    if referenced_files:
        note_parts.append(f"OME multi-file series ({len(referenced_files)} files referenced)")
    else:
        note_parts.append("OME metadata present")
    if missing_files:
        note_parts.append(f"MISSING referenced files: {len(missing_files)}")
    if time_inc_s is not None:
        note_parts.append(f"dt={time_inc_s:.6g}s")
    if physical_um:
        note_parts.append(
            "px_um="
            + ",".join(f"{k}:{v:.6g}" for k, v in physical_um.items() if k in ("X", "Y"))
            + (f" dz_um:{physical_um['Z']:.6g}" if "Z" in physical_um else "")
        )
    note = " | ".join(note_parts)

    # across_* are now legacy; set based on filename variability heuristics if desired.
    # Here, since OME defines the series, treat them as False.
    plan = Plan(
        base=base,
        anchor_path=anchor_path,
        across_t=False,
        across_z=False,
        across_c=False,
        nt=int(sizes.get("SizeT", nt)),
        nz=int(sizes.get("SizeZ", nz)),
        nc=int(sizes.get("SizeC", nc)),
        yx=(int(sizes.get("SizeY", ny)), int(sizes.get("SizeX", nx))),
        dtype=dtype,
        n_files=len(referenced_files) if referenced_files else 1,
        missing_keys=[],
        missing_files=missing_files,
        axes=axes,
        referenced_files=referenced_files,
        physical_sizes_um=physical_um,
        time_increment_s=time_inc_s,
        out_path=out_path,
        note=note,
    )

    internal = {
        "axes": axes,
        "shape": shape,
    }
    return plan, internal


def _base_from_filename(name: str) -> str | None:
    m = TIFF_RE.match(name)
    return m.group("base") if m else None


def assemble_marianas_stack(
    input_dir: Path | str,
    *,
    out_dir: Path | str | None = None,
    recursive: bool = False,
    save: bool = True,
    verbose: bool = True,
    # OME-first behavior
    use_ome: bool = True,
    # Metadata supplement
    log_files: bool = True,
    # Missing-data behavior (applies to OME-referenced files too)
    allow_missing_files: bool = False,
    fill_value: int | None = None,  # legacy; OME mode uses tifffile's zero-fill
    # Legacy override knobs (kept for backward-compatibility; ignored in OME mode)
    nt: int | None = None,
    nz: int | None = None,
    nc: int | None = None,
    file_nt: int | None = None,
    file_nz: int | None = None,
    file_nc: int | None = None,
    out_ext: str = "ome.tif",
    bases: list[str] | None = None,
    return_data: bool = False,
) -> dict:
    """Batch assemble Marianas/SlideBook TIFF exports into one OME-TIFF per base.

    This function is now **OME-first**:

    - If a dataset is an OME multi-file series, we rely on the OME-XML to determine
      the full stack shape and to stitch the referenced files.
    - If OME metadata is missing, we fall back to the legacy filename/grid heuristics.

    Log files are optional and used only as a *metadata supplement* (e.g. fill missing dt,
    and correct/confirm physical pixel sizes). They do not affect stacking logic in OME mode.

    Parameters
    ----------
    input_dir:
        Folder containing exported TIFFs (and optionally .log files).
    out_dir:
        Output folder (default: input_dir / "Stacks")
    recursive:
        If True, search input_dir recursively for TIFFs.
    save:
        If True, write OME-TIFF(s). If False, dry-run planning only.
    verbose:
        Print plan summaries.
    use_ome:
        If True (default), prefer OME-XML + tifffile series stitching when OME metadata is present.
    log_files:
        If True (default), use Marianas .log files to supplement missing metadata (dt, physical sizes).
    allow_missing_files:
        If False (default), raise if OME-XML references files that are missing.
        If True, proceed; tifffile will zero-fill missing planes and emit warnings.
    fill_value:
        Legacy: fill value for missing slots when using filename/grid assembly.
        (OME mode uses tifffile's behavior; fill_value is ignored there.)
    nt/nz/nc, file_nt/file_nz/file_nc:
        Legacy overrides for filename/grid assembly; ignored in OME mode.
    out_ext:
        Output extension, e.g. "ome.tif".
    bases:
        Optional list of base names to assemble.
    return_data:
        If True and save=True, also return assembled numpy array(s) in-memory (can be huge).

    Returns
    -------
    dict with keys:
        - "input_dir", "out_dir", "n_groups"
        - "plans": list[Plan]
        - "written": list[Path] (if save=True)
        - "data": dict[str, np.ndarray] (if return_data=True and save=True)
        - "notes": list[str]
        - "summary": str
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
        msg = f"No matching TIFFs found in {inp}"
        if verbose:
            print(msg)
        return {
            "input_dir": inp,
            "out_dir": outp,
            "n_groups": 0,
            "plans": [],
            "written": [],
            "data": {} if return_data else None,
            "notes": [],
            "summary": msg,
        }

    written: list[Path] = []
    data_out: dict[str, np.ndarray] = {}
    plans: list[Plan] = []
    notes: list[str] = []

    if verbose:
        mode = "SAVE" if save else "DRY-RUN"
        print(f"Found {len(groups)} group(s). Mode: {mode}")
        print(f"Output dir: {outp}")
        print(f"OME-first: use_ome={use_ome} (log_files={log_files})")
        if not use_ome:
            if any(v is not None for v in (nt, nz, nc)):
                print(f"Final overrides: nt={nt} nz={nz} nc={nc}")
            if any(v is not None for v in (file_nt, file_nz, file_nc)):
                print(f"Internal overrides: file_nt={file_nt} file_nz={file_nz} file_nc={file_nc}")
        print("")

    for i, (base, g) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower()), start=1):
        # Pick an anchor file for this base (prefer lowest T, then Z, then C).
        anchor_key = sorted(g.frames.keys(), key=lambda k: (k.t, k.z, k.c))[0]
        anchor_path = g.frames[anchor_key]

        # Determine whether this group has OME metadata
        ome_available = False
        if use_ome:
            try:
                with tiff.TiffFile(str(anchor_path)) as tif:
                    ome_available = tif.ome_metadata is not None
            except Exception:
                ome_available = False

        if ome_available and use_ome:
            plan, internal = _ome_plan_for_anchor(
                anchor_path=anchor_path,
                out_dir=outp,
                out_ext=out_ext,
                log_files=log_files,
            )
        else:
            # Legacy fallback
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
            print(f"  Anchor: {anchor_path.name}")
            if getattr(plan, "axes", None):
                print(f"  Series axes/shape: {plan.axes} / ({plan.nt},{plan.nz},{plan.yx[0]},{plan.yx[1]},{plan.nc})")
            print(f"  {plan.note}")
            print(f"  Output: {plan.out_path}")

            if getattr(plan, "missing_files", None):
                mf = plan.missing_files
                if mf:
                    print(f"  Missing referenced files: {len(mf)}")
                    for fn in mf[:10]:
                        print(f"    missing {fn}")
                    if len(mf) > 10:
                        print(f"    ... (+{len(mf)-10} more)")
                    if not allow_missing_files:
                        print("  NOTE: will ERROR on missing unless allow_missing_files=True")

            if plan.missing_keys:
                print(f"  Missing file-slots (legacy grid): {len(plan.missing_keys)}")
                if not allow_missing_files:
                    print("  NOTE: will ERROR on missing unless allow_missing_files=True")

        if save:
            # OME-first save path
            if ome_available and use_ome:
                if getattr(plan, "missing_files", None) and plan.missing_files and not allow_missing_files:
                    # Fail early with a clear error
                    sample = "\n".join(f"  - {fn}" for fn in plan.missing_files[:25])
                    more = "" if len(plan.missing_files) <= 25 else f"\n  ... (+{len(plan.missing_files)-25} more)"
                    raise FileNotFoundError(
                        f"OME series for base '{plan.base}' references missing file(s):\n{sample}{more}"
                    )

                # Read the full OME series (may emit warnings + zero-fill if allow_missing_files=True)
                with tiff.TiffFile(str(anchor_path)) as tif:
                    axes = getattr(tif.series[0], "axes", None)
                arr = tiff.imread(str(anchor_path))
                arr_tzyxc = _ensure_tzyxc(arr, axes)

                write_ome_tiff(
                    plan.out_path,
                    arr_tzyxc,
                    physical_sizes_um=getattr(plan, "physical_sizes_um", None),
                    time_increment_s=getattr(plan, "time_increment_s", None),
                )
                written.append(plan.out_path)
                if return_data:
                    data_out[plan.base] = arr_tzyxc
                if verbose:
                    print("  Wrote OME-TIFF.")
            else:
                # Legacy save path
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
                    print("  Wrote OME-TIFF (legacy).")
        else:
            if verbose:
                print("  Dry-run only (set save=True to write).")

        if verbose:
            print("")

    summary = f"Assembled {len(plans)} plan(s). Wrote {len(written)} file(s)." if save else f"Planned {len(plans)} stack(s) (dry-run)."
    return {
        "input_dir": inp,
        "out_dir": outp,
        "n_groups": len(groups),
        "plans": plans,
        "written": written,
        "data": data_out if return_data else None,
        "notes": notes,
        "summary": summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())