from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import gzip


def _open_text(path: Union[str, Path], mode: str = "rt"):
    """Open plain text or .gz transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _coerce_field(token: str) -> Any:
    """Best-effort coercion for 'extra' columns: int -> float -> str."""
    if _is_int(token):
        return int(token)
    if _is_float(token):
        return float(token)
    return token


@dataclass(frozen=True)
class IntervalRecord:
    """
    Represents one BED/bedGraph line.
    Minimal required fields: chrom, start, end.
    Optional: name, score, strand.
    Extras: any remaining columns (typed).
    """
    chrom: str
    start: int
    end: int
    name: Optional[str] = None
    score: Optional[Union[int, float]] = None
    strand: Optional[str] = None
    extras: Optional[Tuple[Any, ...]] = None

    def __init__(
        self,
        chrom: str,
        start: int,
        end: int,
        name: Optional[str] = None,
        score: Optional[Union[int, float]] = None,
        strand: Optional[str] = None,
        extras: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        object.__setattr__(self, "chrom", chrom)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "strand", strand)
        object.__setattr__(self, "extras", extras)

        if self.start < 0 or self.end < 0 or self.end < self.start:
            raise ValueError(f"Invalid interval coordinates: {self}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        """Half-open intervals: [start, end)."""
        return self.chrom == chrom and (self.start < end) and (start < self.end)

    def to_bed_fields(
        self,
        columns: Sequence[str] = ("chrom", "start", "end", "name", "score", "strand"),
        missing: str = ".",
    ) -> List[str]:
        out: List[str] = []
        for col in columns:
            if col == "chrom":
                out.append(str(self.chrom))
            elif col == "start":
                out.append(str(self.start))
            elif col == "end":
                out.append(str(self.end))
            elif col == "name":
                out.append(self.name if self.name is not None else missing)
            elif col == "score":
                out.append(str(self.score) if self.score is not None else missing)
            elif col == "strand":
                out.append(self.strand if self.strand is not None else missing)
            elif col == "extras":
                if self.extras is None:
                    out.append(missing)
                else:
                    out.extend(str(x) if x is not None else missing for x in self.extras)
            else:
                raise ValueError(f"Unknown column spec: {col}")

        return out

    def get_col_1based(self, col: int):
        """
        Return the value of a BED column using 1-based indexing.

        BED column mapping:
          1  chrom
          2  start
          3  end
          4  name
          5  score
          6  strand
          7+ extras[0], extras[1], ...

        Returns None if the column does not exist.
        """
        if col < 1:
            raise ValueError("BED columns are 1-based and must be >= 1")

        if col == 1:
            return self.chrom
        elif col == 2:
            return self.start
        elif col == 3:
            return self.end
        elif col == 4:
            return self.name
        elif col == 5:
            return self.score
        elif col == 6:
            return self.strand
        else:
            idx = col - 7
            if self.extras is None:
                return None
            if idx < 0 or idx >= len(self.extras):
                return None
            return self.extras[idx]



class BedTable:
    """
    Container for BED/BEDGRAPH-like interval tables.

    - Stores IntervalRecord rows
    - Keeps header/comment lines and optional track/browser lines
    - Remembers inferred format (bed vs bedGraph) and column count
    """

    def __init__(
        self,
        records: Optional[Iterable[IntervalRecord]] = None,
        *,
        comments: Optional[List[str]] = None,
        track_lines: Optional[List[str]] = None,
        browser_lines: Optional[List[str]] = None,
        source_path: Optional[Union[str, Path]] = None,
        inferred_kind: Optional[str] = None,  # "bed" | "bedgraph" | None
        inferred_ncols: Optional[int] = None,
    ) -> None:
        self._records: List[IntervalRecord] = list(records) if records is not None else []
        self.comments: List[str] = comments or []
        self.track_lines: List[str] = track_lines or []
        self.browser_lines: List[str] = browser_lines or []
        self.source_path: Optional[Path] = Path(source_path) if source_path else None
        self.inferred_kind: Optional[str] = inferred_kind
        self.inferred_ncols: Optional[int] = inferred_ncols

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[IntervalRecord]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> IntervalRecord:
        return self._records[idx]

    @property
    def records(self) -> List[IntervalRecord]:
        """Direct access if you want to mutate list in-place."""
        return self._records

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        delimiter: str = "\t",
        allow_spaces: bool = True,
        coerce_extras: bool = True,
        strict: bool = False,
    ) -> BedTable:
        """
        Read BED / bedGraph.

        Parsing rules:
        - Skip blank lines
        - Preserve comment lines (#...), track, browser lines
        - Expect at least 3 columns: chrom, start, end
        - If 4 cols and last is numeric => likely bedGraph (value)
        - Otherwise interpret 4th/5th/6th as name/score/strand (BED6)
        """
        path = Path(path)
        comments: List[str] = []
        track_lines: List[str] = []
        browser_lines: List[str] = []
        records: List[IntervalRecord] = []

        inferred_kind: Optional[str] = None
        inferred_ncols: Optional[int] = None

        with _open_text(path, "rt") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue

                lstripped = line.lstrip()
                if lstripped.startswith("#"):
                    comments.append(line)
                    continue
                if lstripped.startswith("track"):
                    track_lines.append(line)
                    continue
                if lstripped.startswith("browser"):
                    browser_lines.append(line)
                    continue

                # Split
                if delimiter == "\t":
                    parts = line.split("\t")
                    if allow_spaces and len(parts) == 1:
                        parts = line.split()  # fallback to any whitespace
                else:
                    parts = line.split(delimiter)

                if len(parts) < 3:
                    if strict:
                        raise ValueError(f"Expected >=3 columns, got {len(parts)}: {line}")
                    continue

                chrom = parts[0]
                if not _is_int(parts[1]) or not _is_int(parts[2]):
                    if strict:
                        raise ValueError(f"start/end must be integers: {line}")
                    continue

                start = int(parts[1])
                end = int(parts[2])

                ncols = len(parts)
                if inferred_ncols is None:
                    inferred_ncols = ncols

                # Heuristic: bedGraph is 4 cols and col4 is numeric (value).
                name: Optional[str] = None
                score: Optional[Union[int, float]] = None
                strand: Optional[str] = None
                extras: Tuple[Any, ...] = ()

                if ncols == 4 and _is_float(parts[3]):
                    inferred_kind = inferred_kind or "bedgraph"
                    extras = (float(parts[3]),) if coerce_extras else (parts[3],)
                else:
                    inferred_kind = inferred_kind or "bed"
                    if ncols >= 4:
                        name = parts[3] if parts[3] != "." else None
                    if ncols >= 5:
                        tok = parts[4]
                        if tok != "." and _is_float(tok):
                            # BED score typically int, but allow float too.
                            score = int(tok) if _is_int(tok) else float(tok)
                        elif tok != ".":
                            # if non-numeric but present, keep as extra
                            score = None
                    if ncols >= 6:
                        strand = parts[5] if parts[5] != "." else None

                    if ncols > 6:
                        tail = parts[6:]
                        if coerce_extras:
                            extras = tuple(_coerce_field(t) for t in tail)
                        else:
                            extras = tuple(tail)

                records.append(
                    IntervalRecord(
                        chrom=chrom,
                        start=start,
                        end=end,
                        name=name,
                        score=score,
                        strand=strand,
                        extras=extras,
                    )
                )

        return cls(
            records,
            comments=comments,
            track_lines=track_lines,
            browser_lines=browser_lines,
            source_path=path,
            inferred_kind=inferred_kind,
            inferred_ncols=inferred_ncols,
        )

    def to_path(
        self,
        path: Union[str, Path],
        *,
        delimiter: str = "\t",
        write_comments: bool = True,
        write_track_browser: bool = True,
        columns: Optional[Sequence[str]] = None,
        include_extras: bool = True,
        missing: str = ".",
        sort_key: Optional[Callable[[IntervalRecord], Any]] = None,
    ) -> None:
        """
        Write out to BED-like text. If you want a canonical BED6, pass columns.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(columns) if columns is not None else ["chrom", "start", "end"]

        # Include name/score/strand automatically if present in any record
        if columns is None:
            any_name = any(r.name is not None for r in self._records)
            any_score = any(r.score is not None for r in self._records)
            any_strand = any(r.strand is not None for r in self._records)
            if any_name:
                cols.append("name")
            if any_score:
                cols.append("score")
            if any_strand:
                cols.append("strand")
            if include_extras and any(len(r.extras) for r in self._records):
                cols.append("extras")
        else:
            if include_extras and "extras" not in cols:
                # user chose explicit columns, respect that; do nothing
                pass

        recs = self._records if sort_key is None else sorted(self._records, key=sort_key)

        with _open_text(path, "wt") as out:
            if write_track_browser:
                for ln in self.browser_lines:
                    out.write(ln.rstrip("\n") + "\n")
                for ln in self.track_lines:
                    out.write(ln.rstrip("\n") + "\n")
            if write_comments:
                for ln in self.comments:
                    out.write(ln.rstrip("\n") + "\n")

            for r in recs:
                fields = r.to_bed_fields(columns=cols, missing=missing)
                out.write(delimiter.join(fields) + "\n")

    def filter(self, predicate: Callable[[IntervalRecord], bool]) -> BedTable:
        """Return a new BedTable with records where predicate(record) is True."""
        return BedTable(
            (r for r in self._records if predicate(r)),
            comments=list(self.comments),
            track_lines=list(self.track_lines),
            browser_lines=list(self.browser_lines),
            source_path=self.source_path,
            inferred_kind=self.inferred_kind,
            inferred_ncols=self.inferred_ncols,
        )

    def subset_region(self, chrom: str, start: int, end: int) -> BedTable:
        """Return records overlapping [start, end) on chrom."""
        return self.filter(lambda r: r.overlaps(chrom, start, end))

    def sort(self, *, by: str = "coord", reverse: bool = False) -> BedTable:
        """
        by:
          - "coord": chrom, start, end
          - "start": start
          - "end": end
          - "length": length
        """
        if by == "coord":
            key = lambda r: (r.chrom, r.start, r.end)
        elif by == "start":
            key = lambda r: r.start
        elif by == "end":
            key = lambda r: r.end
        elif by == "length":
            key = lambda r: r.length
        else:
            raise ValueError(f"Unknown sort key: {by}")

        return BedTable(
            sorted(self._records, key=key, reverse=reverse),
            comments=list(self.comments),
            track_lines=list(self.track_lines),
            browser_lines=list(self.browser_lines),
            source_path=self.source_path,
            inferred_kind=self.inferred_kind,
            inferred_ncols=self.inferred_ncols,
        )

    def groupby_chrom(self) -> Dict[str, BedTable]:
        """Return dict chrom -> BedTable for that chrom."""
        out: Dict[str, List[IntervalRecord]] = {}
        for r in self._records:
            out.setdefault(r.chrom, []).append(r)
        return {c: BedTable(v, inferred_kind=self.inferred_kind) for c, v in out.items()}

    def as_tuples(self) -> List[Tuple[Any, ...]]:
        """Convenient export for pandas/DataFrame creation."""
        rows = []
        for r in self._records:
            rows.append((r.chrom, r.start, r.end, r.name, r.score, r.strand, *r.extras))
        return rows

    def get_column_1based(self, col: int, *, as_float: bool = False):
        """
        Extract a BED column (1-based) across all records.

        If as_float=True, values are converted to float with NaN for missing.
        """
        out = []
        for r in self._records:
            v = r.get_col_1based(col)
            if as_float:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(float("nan"))
            else:
                out.append(v)
        return out
